import os, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from tap import Tap
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch_geometric.nn import GCNConv, SAGEConv

from modules.data import get_data
from modules.gcn import GCN
from modules.utils import (TensorMap, get_logger, get_neighborhoods,
                            sample_neighborhoods_from_probs, slice_adjacency)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OOF_CACHE_DIR = os.environ.get('RAHAS_OOF_CACHE_DIR', 'oof_cache')
ASGCN_ROW_NORM = False          # set from --asgcn_row_norm in __main__


class Arguments(Tap):
    dataset: str = 'cora'
    sampling_hops: int = 2
    num_samples: int = 4            # GFN per-node samples (v6)
    use_indicators: bool = True
    lr_gf: float = 1e-4
    lr_gc: float = 1e-3
    loss_coef: float = 1.0
    log_z_init: float = 0.
    reg_param: float = 0.
    dropout: float = 0.0
    hidden_dim: int = 256
    max_epochs: int = 50
    batch_size: int = 512
    eval_frequency: int = 10
    runs: int = 3
    split_id: int = 0
    seed: int = None
    config_file: str = None
    use_best_val_checkpoint: bool = True
    expert_mode: str = 'v6'
    # ASGCN (original run_pubmed.py defaults)
    asgcn_hidden: int = 16
    asgcn_rank: int = 128
    asgcn_var: float = 0.5
    asgcn_wd: float = 5e-4
    asgcn_batch_size: int = 256
    asgcn_max_degree: int = 32       # <=0 disables train-time neighbour truncation
    asgcn_train_graph: str = 'full'  # 'full' | 'induced' (original: subgraph of train nodes)
    asgcn_dup_cols_zero: bool = True # original gather_columns quirk
    asgcn_row_norm: bool = False
    # router
    gnn_hidden: int = 512
    gnn_out: int = 512
    router_epochs: int = 300
    router_lr: float = 1e-3
    router_dropout: float = 0.0
    use_jaccard: bool = True
    jaccard_top_k: int = 5
    # analysis
    homophily_threshold: float = 0.5   # node homophily >= thr -> homophilous


# =============================================================================
# Helpers
# =============================================================================
def build_adjacency(data):
    return sp.csr_matrix((np.ones(data.num_edges, dtype=bool), data.edge_index.cpu()),
                         shape=(data.num_nodes, data.num_nodes))


def build_a_hat(adjacency):
    """D^-1/2 (A+I) D^-1/2 (AS-GCN's `adj`)."""
    n = adjacency.shape[0]
    A = (adjacency.astype(np.float32).tocsr() + sp.eye(n, dtype=np.float32, format='csr')).tocsr()
    A.data[:] = 1.0
    dinv = np.power(np.asarray(A.sum(1)).ravel(), -0.5)
    return (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr().astype(np.float32)


def log_dataset_info(data, num_features, num_classes, logger, tag=''):
    n_train = int(data.train_mask.sum())
    n_val = int(data.val_mask.sum())
    n_test = int(data.test_mask.sum())
    msg = (f'[Dataset{" " + tag if tag else ""}] n_nodes={data.num_nodes} '
           f'n_features={num_features} n_classes={num_classes} | '
           f'train_mask={n_train} val_mask={n_val} test_mask={n_test}')
    logger.info(msg)
    print(msg)


def _summarize(target_logits, labels, node_ids):
    probs = F.softmax(target_logits, dim=-1)
    true_prob = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return (target_logits.argmax(-1).cpu(), labels.cpu(), node_ids.cpu(), true_prob.cpu(),
            probs.max(dim=-1).values.cpu(), ent.cpu())


def _pack(parts):
    names = ['preds', 'labels', 'node_ids', 'true_prob', 'confidence', 'entropy']
    return {n: torch.cat([p[i] for p in parts]).numpy() for i, n in enumerate(names)}


def _align_by_node_id(d):
    order = np.argsort(d['node_ids'])
    return {k: v[order] for k, v in d.items()}


# =============================================================================
# Node properties (for studying WHY a node goes to ASGCN or GFN)  -- from V9
# =============================================================================
def _core_numbers(A):
    N = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel().copy()
    core = np.zeros(N, dtype=np.float32); alive = np.ones(N, dtype=bool); k = 0
    while alive.any():
        rem = alive & (deg <= k)
        if not rem.any():
            k += 1; continue
        core[rem] = k; alive[rem] = False
        deg -= A @ rem.astype(np.float32)
    return core


def _pagerank(A, deg, d=0.85, iters=50):
    N = A.shape[0]
    dinv = 1.0 / np.maximum(deg, 1)
    r = np.full(N, 1.0 / N, dtype=np.float64); dang = deg == 0
    for _ in range(iters):
        r = d * (A.T @ (r * dinv)) + (1 - d) / N + d * r[dang].sum() / N
    return (r * N).astype(np.float32)


def _dist_to_train(A, train_mask):
    N = A.shape[0]
    dist = np.full(N, np.inf); front = train_mask.copy(); visited = front.copy()
    dist[front] = 0; d = 0
    while front.any():
        d += 1
        nxt = ((A @ front.astype(np.float32)) > 0) & ~visited
        dist[nxt] = d; visited |= nxt; front = nxt
    fin = np.isfinite(dist)
    dist[~fin] = (dist[fin].max() if fin.any() else 0) + 1
    return dist.astype(np.float32)


_STRUCT = {}
def get_node_props(data, adjacency):
    N = data.num_nodes
    key = (N, data.num_edges)
    if key not in _STRUCT:
        A = adjacency.astype(np.float32).tocsr()
        A = A.maximum(A.T).tocsr()
        A = (A - sp.diags(A.diagonal())).tocsr(); A.eliminate_zeros(); A.data[:] = 1.0
        deg = np.asarray(A.sum(1)).ravel().astype(np.float64)
        dn = 1.0 / np.maximum(deg, 1)
        y = data.y.cpu().numpy()
        Ac = A.tocoo(); r, c = Ac.row.astype(np.int64), Ac.col.astype(np.int64)
        h = np.bincount(r, weights=(y[r] == y[c]).astype(np.float64), minlength=N) * dn
        x = data.x.cpu().float()
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        rt, ct = torch.as_tensor(r), torch.as_tensor(c)
        cos = np.bincount(r, weights=(xn[rt] * xn[ct]).sum(1).numpy(), minlength=N) * dn
        Y = np.eye(int(y.max()) + 1, dtype=np.float32)[y]
        cnt = A @ Y
        p = cnt * dn[:, None]
        lab_ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
        nbr_major = cnt.argmax(1); nbr_major[deg == 0] = -1
        A2 = (A @ A).tocsr()
        tri = np.asarray(A2.multiply(A).sum(1)).ravel() / 2.0
        clust = np.where(deg >= 2, tri / np.maximum(deg * (deg - 1) / 2.0, 1), 0.0)
        B = (A2 + A).tocsr()
        B = (B - sp.diags(B.diagonal())).tocsr(); B.eliminate_zeros()
        two_hop = np.diff(B.indptr).astype(np.float32)
        X = x.numpy().astype(np.float64)
        mx = (A @ X) * dn[:, None]
        sq = (A @ (X ** 2).sum(1)) * dn
        nbr_var = np.maximum(sq - (mx ** 2).sum(1), 0)
        _STRUCT[key] = dict(A=A, props=dict(deg=deg, h=h, cos=cos, lab_ent=lab_ent, two_hop=two_hop,
                                            clust=clust, pagerank=_pagerank(A, deg),
                                            kcore=_core_numbers(A), nbr_var=nbr_var),
                            nbr_major=nbr_major)
    S = _STRUCT[key]
    out = dict(S['props'])
    out['dist_train'] = _dist_to_train(S['A'], data.train_mask.cpu().numpy())
    out['nbr_major'] = S['nbr_major']
    return out


def build_rec(props, ids, pg, pa, cg, ca, eg, ea, jac):
    """Per-node record: structural props + policy-specific stats (pg/pa = predictions)."""
    nm = props['nbr_major'][ids]
    rec = {k: v[ids] for k, v in props.items() if k != 'nbr_major'}
    iso = rec['deg'] == 0
    rec.update(conf_gfn=cg, conf_asgcn=ca, ent_gfn=eg, ent_asgcn=ea,
               conf_diff=cg - ca, disagree=(pg != pa).astype(np.float32),
               agree_gfn=np.where(iso, np.nan, (pg == nm).astype(np.float32)),
               agree_asgcn=np.where(iso, np.nan, (pa == nm).astype(np.float32)),
               jac=jac)
    return rec


def _cat(recs):
    return {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}


def _nm(a):
    a = a[~np.isnan(a)]
    return a.mean() if a.size else float('nan')


def report_groups(title, groups, rec, thr):
    """Counts per group split homophilous / heterophilous, then mean node properties."""
    deg = rec['deg']
    homo = (deg > 0) & (rec['h'] >= thr)
    het = (deg > 0) & (rec['h'] < thr)
    iso = deg == 0
    n = len(deg)
    print(f'\n{"-"*100}\n{title}  (N={n}, homophilous if node-homophily >= {thr})\n{"-"*100}')
    print(f'{"Group":<38}{"N":>7}{"%all":>7}{"homophilous":>16}{"heterophilous":>16}{"isolated":>10}')
    for name, m in groups.items():
        c = int(m.sum())
        def f(sub):
            k = int((m & sub).sum()); return f'{k} ({100*k/max(c,1):.0f}%)'
        print(f'{name:<38}{c:>7}{100*c/max(n,1):>6.1f}%{f(homo):>16}{f(het):>16}{int((m&iso).sum()):>10}')
    keys = list(rec.keys())
    for i in range(0, len(keys), 9):
        chunk = keys[i:i + 9]
        print(f'\nMean node properties per group:' if i == 0 else '')
        print(f'{"Group":<38}' + ''.join(f'{k:>12}' for k in chunk))
        for name, m in groups.items():
            print(f'{name:<38}' + ''.join(
                f'{(_nm(rec[k][m].astype(np.float64)) if m.any() else float("nan")):>12.3f}' for k in chunk))


def report_classes(title, labels, cgroups):
    print(f'\n{"-"*100}\n{title}\n{"-"*100}')
    names = list(cgroups.keys())
    print(f'{"Class":<8}{"N":>7}' + ''.join(f'{n[:19]:>21}' for n in names))
    for c in range(int(labels.max()) + 1):
        mc = labels == c
        k = int(mc.sum())
        print(f'{c:<8}{k:>7}' + ''.join(f'{100*(m & mc).sum()/max(k,1):>20.1f}%' for m in cgroups.values()))


def one_line(tag, m, rec, thr):
    deg = rec['deg']
    homo = int((m & (deg > 0) & (rec['h'] >= thr)).sum())
    het = int((m & (deg > 0) & (rec['h'] < thr)).sum())
    return f'{tag}={int(m.sum())} (homo {homo} / hetero {het})'


def assignment_summary(title, to_a, to_g, rec, thr):
    """How many homophilous nodes go to ASGCN and how many heterophilous go to GFN.
    to_a / to_g: boolean masks of nodes assigned to ASGCN / GFN (oracle: only-X-right; router: argmax)."""
    deg, h = rec['deg'], rec['h']
    homo = (deg > 0) & (h >= thr)
    het = (deg > 0) & (h < thr)
    print(f'\n[{title}]  (homophilous: node-homophily >= {thr}; isolated nodes excluded)')
    for name, grp in (('homophilous', homo), ('heterophilous', het)):
        tot = int(grp.sum()); a = int((grp & to_a).sum()); g = int((grp & to_g).sum()); asg = a + g
        print(f'  {name:<14} total={tot:<6} '
              f'-> ASGCN={a:<6} ({100*a/max(tot,1):5.1f}% of all, {100*a/max(asg,1):5.1f}% of assigned) | '
              f'-> GFN={g:<6} ({100*g/max(tot,1):5.1f}% of all, {100*g/max(asg,1):5.1f}% of assigned)')
    hit = int((homo & to_a).sum() + (het & to_g).sum())
    asg_all = int(((homo | het) & (to_a | to_g)).sum())
    print(f'  hypothesis (homophilous->ASGCN, heterophilous->GFN) holds for {hit}/{asg_all} '
          f'({100*hit/max(asg_all,1):.1f}%) of assigned non-isolated nodes')


# =============================================================================
# GraphConv layer for the router (PyG) -- unchanged from V6
# =============================================================================
class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, base='sage', agg='mean'):
        super().__init__()
        assert base in ('gcn', 'sage') and agg in ('mean', 'max', 'sum')
        self.base = base
        pyg_aggr = 'add' if agg == 'sum' else agg
        if base == 'sage':
            self.conv = SAGEConv(in_dim, out_dim, aggr=pyg_aggr)
        else:
            self.conv = GCNConv(in_dim, out_dim, aggr=pyg_aggr)

    def forward(self, x, edge_index, num_nodes):
        return F.relu(self.conv(x, edge_index))


# =============================================================================
# POLICY 1: GFN -- unchanged from V6 (also evaluated on train_mask)
# =============================================================================
def train_gfn_mainstyle(args, seed, logger=None):
    logger = logger or get_logger()
    if args.loss_coef < 10:
        logger.info(f'[V6][GFN] WARNING: loss_coef={args.loss_coef} looks too small for the '
                    f'TB loss (log_z + log_prob + loss_coef*CE)**2 -- consider --loss_coef 10000.')

    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset, seed=seed, split_id=args.split_id)
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    log_dataset_info(data, num_features, num_classes, logger, tag=f'GFN seed={seed}')

    node_map = TensorMap(size=data.num_nodes)
    num_indicators = (args.sampling_hops + 1) if args.use_indicators else 0

    gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)
    gcn_gf = GCN(data.num_features + num_indicators, hidden_dims=[args.hidden_dim, 1]).to(device)
    gcn_z = GCN(data.num_features, hidden_dims=[args.hidden_dim, 1]).to(device)

    optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)
    optimizer_gf = Adam(list(gcn_gf.parameters()) + list(gcn_z.parameters()), lr=args.lr_gf)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(data.train_mask.nonzero().squeeze(1)), batch_size=args.batch_size)
    val_loader = DataLoader(TensorDataset(data.val_mask.nonzero().squeeze(1)), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(data.test_mask.nonzero().squeeze(1)), batch_size=args.batch_size)
    adjacency = build_adjacency(data)

    prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    indicator_features = torch.zeros((data.num_nodes, num_indicators))

    best_val_acc, best_state, best_epoch = -1.0, None, -1
    logger.info(f'[V6][GFN main.py-style][seed={seed}] training {args.max_epochs} epochs')

    for epoch in range(1, args.max_epochs + 1):
        with tqdm(total=len(train_loader), desc=f'[GFN] Epoch {epoch}') as bar:
            for batch in train_loader:
                target_nodes = batch[0]
                previous_nodes = target_nodes.clone()
                all_nodes_mask = torch.zeros_like(prev_nodes_mask)
                all_nodes_mask[target_nodes] = True
                indicator_features.zero_(); indicator_features[target_nodes, -1] = 1.0
                global_edge_indices, log_probs = [], []
                log_z = torch.tensor([0.0], device=device)

                for hop in range(args.sampling_hops):
                    neighborhoods = get_neighborhoods(previous_nodes, adjacency)
                    prev_nodes_mask.zero_(); batch_nodes_mask.zero_()
                    prev_nodes_mask[previous_nodes] = True
                    batch_nodes_mask[neighborhoods.view(-1)] = True
                    neighbor_nodes_mask = batch_nodes_mask & ~prev_nodes_mask
                    batch_nodes = node_map.values[batch_nodes_mask]
                    neighbor_nodes = node_map.values[neighbor_nodes_mask]
                    indicator_features[neighbor_nodes, hop] = 1.0
                    node_map.update(batch_nodes)
                    local_neighborhoods = node_map.map(neighborhoods).to(device)

                    x = torch.cat([data.x[batch_nodes], indicator_features[batch_nodes]], dim=1).to(device) \
                        if args.use_indicators else data.x[batch_nodes].to(device)
                    node_logits, *_ = gcn_gf(x, local_neighborhoods)
                    node_logits = node_logits[node_map.map(neighbor_nodes)]

                    sampled_neighboring_nodes, log_prob, _ = sample_neighborhoods_from_probs(
                        node_logits, neighbor_nodes, args.num_samples)
                    all_nodes_mask[sampled_neighboring_nodes] = True

                    if hop == 0:
                        pred_gcn_z = gcn_z(data.x[batch_nodes].to(device), local_neighborhoods)[0].squeeze()
                        log_z = pred_gcn_z.mean() - args.log_z_init

                    log_probs.append(log_prob)
                    batch_nodes = torch.cat([target_nodes, sampled_neighboring_nodes], dim=0)
                    global_edge_indices.append(slice_adjacency(adjacency, rows=batch_nodes, cols=previous_nodes))
                    previous_nodes = batch_nodes.clone()

                all_nodes = node_map.values[all_nodes_mask]
                node_map.update(all_nodes)
                edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]
                logits, *_ = gcn_c(data.x[all_nodes].to(device), edge_indices)
                local_target_ids = node_map.map(target_nodes)
                loss_c = loss_fn(logits[local_target_ids], data.y[target_nodes].to(device)) \
                         + args.reg_param * torch.sum(torch.var(logits, dim=1))

                optimizer_c.zero_grad(); loss_c.backward(); optimizer_c.step()

                optimizer_gf.zero_grad()
                cost_gfn = loss_c.detach()
                tot_log_prob = torch.sum(torch.cat(log_probs, dim=0))
                loss_gfn = (log_z + tot_log_prob + args.loss_coef * cost_gfn) ** 2
                loss_gfn.backward(); optimizer_gf.step()

                bar.set_postfix({'loss_c': f'{loss_c.item():.4f}', 'loss_gfn': f'{loss_gfn.item():.3f}'})
                bar.update()

        if epoch % args.eval_frequency == 0:
            val_out = evaluate_gfn_mainstyle(gcn_c, gcn_gf, data, args, adjacency, num_indicators, val_loader)
            val_acc = float((val_out['preds'] == val_out['labels']).mean())
            logger.info(f'[V6][GFN][seed={seed}] epoch={epoch} val_acc={val_acc:.4f}')
            if args.use_best_val_checkpoint and val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                best_state = {'gcn_c': {k: v.detach().cpu().clone() for k, v in gcn_c.state_dict().items()},
                               'gcn_gf': {k: v.detach().cpu().clone() for k, v in gcn_gf.state_dict().items()},
                               'gcn_z': {k: v.detach().cpu().clone() for k, v in gcn_z.state_dict().items()}}

    if args.use_best_val_checkpoint and best_state is not None:
        gcn_c.load_state_dict(best_state['gcn_c']); gcn_gf.load_state_dict(best_state['gcn_gf'])
        gcn_z.load_state_dict(best_state['gcn_z'])
        logger.info(f'[V6][GFN][seed={seed}] restored best ckpt (epoch {best_epoch}, val_acc={best_val_acc:.4f})')

    train_out = evaluate_gfn_mainstyle(gcn_c, gcn_gf, data, args, adjacency, num_indicators, train_loader)
    val_out = evaluate_gfn_mainstyle(gcn_c, gcn_gf, data, args, adjacency, num_indicators, val_loader)
    test_out = evaluate_gfn_mainstyle(gcn_c, gcn_gf, data, args, adjacency, num_indicators, test_loader)
    logger.info(f'[V6][GFN][seed={seed}] FINAL train_acc={float((train_out["preds"]==train_out["labels"]).mean()):.4f} '
                f'val_acc={float((val_out["preds"]==val_out["labels"]).mean()):.4f} '
                f'test_acc={float((test_out["preds"]==test_out["labels"]).mean()):.4f}')
    return train_out, val_out, test_out, gcn_c, gcn_gf, gcn_z


@torch.no_grad()
def evaluate_gfn_mainstyle(gcn_c, gcn_gf, data, args, adjacency, num_indicators, loader):
    gcn_c.eval(); gcn_gf.eval()
    node_map = TensorMap(size=data.num_nodes)
    prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    indicator_features = torch.zeros((data.num_nodes, num_indicators))
    all_preds, all_labels, all_node_ids, all_true_prob, all_conf, all_ent = [], [], [], [], [], []

    for batch in loader:
        target_nodes = batch[0]
        previous_nodes = target_nodes.clone()
        all_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        all_nodes_mask[target_nodes] = True
        indicator_features.zero_(); indicator_features[target_nodes, -1] = 1.0
        global_edge_indices = []

        for hop in range(args.sampling_hops):
            neighborhoods = get_neighborhoods(previous_nodes, adjacency)
            prev_nodes_mask.zero_(); batch_nodes_mask.zero_()
            prev_nodes_mask[previous_nodes] = True
            batch_nodes_mask[neighborhoods.view(-1)] = True
            neighbor_nodes_mask = batch_nodes_mask & ~prev_nodes_mask
            batch_nodes = node_map.values[batch_nodes_mask]
            neighbor_nodes = node_map.values[neighbor_nodes_mask]
            indicator_features[neighbor_nodes, hop] = 1.0
            node_map.update(batch_nodes)
            local_neighborhoods = node_map.map(neighborhoods).to(device)

            x = torch.cat([data.x[batch_nodes], indicator_features[batch_nodes]], dim=1).to(device) \
                if args.use_indicators else data.x[batch_nodes].to(device)
            node_logits, *_ = gcn_gf(x, local_neighborhoods)
            node_logits = node_logits[node_map.map(neighbor_nodes)]
            sampled_neighboring_nodes, _, _ = sample_neighborhoods_from_probs(
                node_logits, neighbor_nodes, args.num_samples)
            all_nodes_mask[sampled_neighboring_nodes] = True
            batch_nodes = torch.cat([target_nodes, sampled_neighboring_nodes], dim=0)
            global_edge_indices.append(slice_adjacency(adjacency, rows=batch_nodes, cols=previous_nodes))
            previous_nodes = batch_nodes.clone()

        all_nodes = node_map.values[all_nodes_mask]
        node_map.update(all_nodes)
        edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]
        logits, *_ = gcn_c(data.x[all_nodes].to(device), edge_indices)
        local_target_ids = node_map.map(target_nodes).to(device)
        target_logits = logits[local_target_ids]
        probs = F.softmax(target_logits, dim=-1)
        labels = data.y[target_nodes].to(device)
        true_prob = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        all_preds.append(target_logits.argmax(-1).cpu()); all_labels.append(labels.cpu())
        all_node_ids.append(target_nodes.cpu()); all_true_prob.append(true_prob.cpu())
        all_conf.append(probs.max(dim=-1).values.cpu())
        all_ent.append((-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).cpu())

    gcn_c.train(); gcn_gf.train()
    return dict(preds=torch.cat(all_preds).numpy(), labels=torch.cat(all_labels).numpy(),
                node_ids=torch.cat(all_node_ids).numpy(), true_prob=torch.cat(all_true_prob).numpy(),
                confidence=torch.cat(all_conf).numpy(), entropy=torch.cat(all_ent).numpy())

def asgcn_features(data):
    x = data.x.float()
    if (not ASGCN_ROW_NORM) or float(x.min()) < 0:
        return x
    rs = x.sum(1, keepdim=True)
    return x * torch.where(rs > 0, 1.0 / rs, torch.zeros_like(rs))


def asgcn_a_hat(args, adjacency):
    A = build_a_hat(adjacency)
    if args.dataset.lower() == 'pubmed':
        A = (A + sp.eye(A.shape[0], dtype=np.float32, format='csr')).tocsr()
    return A.astype(np.float32)


def truncate_rows(A, max_deg, rng):
    A = A.tocsr()
    if max_deg is None or max_deg <= 0:
        return A
    deg = np.diff(A.indptr)
    keep = np.ones(A.nnz, dtype=bool)
    for v in np.nonzero(deg > max_deg)[0]:
        s, e = A.indptr[v], A.indptr[v + 1]
        keep[rng.choice(np.arange(s, e), e - s - max_deg, replace=False)] = False
    rows = np.repeat(np.arange(A.shape[0]), deg)
    return sp.csr_matrix((A.data[keep], (rows[keep], A.indices[keep])), shape=A.shape)


class ASGCNSampler(nn.Module):
    """w_s [d,2], shared between sampler and model, as in the original."""
    def __init__(self, feat_dim):
        super().__init__()
        self.w_s = nn.Parameter(torch.empty(feat_dim, 2))
        nn.init.xavier_uniform_(self.w_s)

    def sample_probs(self, x_v, x_u, colsq, k):
        h_v = (x_v @ self.w_s[:, 1]).sum()
        h_u = x_u @ self.w_s[:, 0]
        attention = (1.0 / k) * (F.relu(h_v + h_u) + 1.0)
        g_u = F.relu(h_u) + 1.0
        p = colsq.sqrt() * attention * g_u
        return p / p.sum()

    def edge_support(self, x_v, x_u, rows, cols, vals, p_u, n_u):
        h_v = x_v @ self.w_s[:, 0]
        h_u = x_u @ self.w_s[:, 1]
        att = (F.relu(h_v[rows] + h_u[cols]) + 1.0) / n_u
        return vals * att / p_u[cols]


def _spmm(rows, cols, w, dense, n_rows):
    return torch.zeros(n_rows, dense.shape[1], device=dense.device, dtype=dense.dtype) \
        .index_add_(0, rows, w.unsqueeze(1) * dense[cols])


class ASGCNNet(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.0):
        super().__init__()
        self.sampler = ASGCNSampler(in_dim)
        self.W0 = nn.Parameter(torch.empty(in_dim, hidden)); nn.init.xavier_uniform_(self.W0)
        self.W1 = nn.Parameter(torch.empty(hidden, n_cls)); nn.init.xavier_uniform_(self.W1)
        self.dropout = dropout

    def forward_sampled(self, x0, x1, xt, E1, p0, E2, p1):
        r1, c1, v1 = E1; r2, c2, v2 = E2
        n0, n1, B = x0.shape[0], x1.shape[0], xt.shape[0]
        s1 = self.sampler.edge_support(x1, x0, r1, c1, v1, p0, n0)
        s2 = self.sampler.edge_support(xt, x1, r2, c2, v2, p1, n1)
        pre0 = F.dropout(x0, self.dropout, self.training) @ self.W0
        h1 = F.relu(_spmm(r1, c1, s1, pre0, n1))
        pre1 = F.dropout(h1, self.dropout, self.training) @ self.W1
        out = _spmm(r2, c2, s2, pre1, B)
        ms = torch.zeros(n1, device=x0.device).index_add_(0, c2, s2) / B
        diff = ms.unsqueeze(1) * pre1 - out.mean(0, keepdim=True)
        reg = (diff ** 2).sum() / (n1 * out.shape[1])
        return out, reg


def asgcn_sample_layer(sampler, x_dev, a_hat, v_ids, k, dup_first_only=True):
    sub = a_hat[v_ids.cpu().numpy()]
    cand = np.unique(sub.indices)
    sub_c = sub[:, cand].tocsc()
    colsq = torch.as_tensor(np.asarray(sub_c.multiply(sub_c).sum(0)).ravel(),
                            dtype=torch.float32, device=device)
    cand_t = torch.as_tensor(cand, dtype=torch.long, device=device)
    with torch.no_grad():
        p = sampler.sample_probs(x_dev[v_ids.to(device)], x_dev[cand_t], colsq, k)
        idx = torch.multinomial(p, k, replacement=True)
    idx_np = idx.cpu().numpy()
    S = sub_c[:, idx_np].tocoo()
    r, c, d = S.row, S.col, S.data
    if dup_first_only:
        first = np.zeros(k, dtype=bool)
        first[np.unique(idx_np, return_index=True)[1]] = True
        keep = first[c]
        r, c, d = r[keep], c[keep], d[keep]
    E = (torch.as_tensor(r, dtype=torch.long, device=device),
         torch.as_tensor(c, dtype=torch.long, device=device),
         torch.as_tensor(d, dtype=torch.float32, device=device))
    return cand_t[idx], p[idx], E


def train_asgcn_native(args, seed, logger=None):
    logger = logger or get_logger()
    assert args.sampling_hops == 2, 'ASGCN port is a fixed 2-layer GCNAdapt'
    assert args.asgcn_train_graph in ('full', 'induced')
    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset, seed=seed, split_id=args.split_id)
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    log_dataset_info(data, num_features, num_classes, logger, tag=f'ASGCN seed={seed}')

    net = ASGCNNet(data.num_features, args.asgcn_hidden, num_classes, args.dropout).to(device)
    opt = Adam(net.parameters(), lr=args.lr_gc)
    loss_fn = nn.CrossEntropyLoss()

    adjacency = build_adjacency(data)
    x_full = asgcn_features(data).to(device)
    a_full = asgcn_a_hat(args, adjacency)

    tr_global = data.train_mask.nonzero().squeeze(1)
    if args.asgcn_train_graph == 'induced':
        t = tr_global.numpy()
        a_tr = asgcn_a_hat(args, adjacency[t][:, t])
        x_tr = x_full[tr_global.to(device)]
        y_tr = data.y[tr_global]
        train_ids = torch.arange(len(tr_global))
    else:
        a_tr, x_tr, y_tr, train_ids = a_full, x_full, data.y, tr_global
    a_tr = truncate_rows(a_tr, args.asgcn_max_degree,
                         np.random.RandomState(None if seed is None else int(seed)))

    train_loader = DataLoader(TensorDataset(train_ids), batch_size=args.asgcn_batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(tr_global), batch_size=args.batch_size)   # global ids, ordered
    val_loader = DataLoader(TensorDataset(data.val_mask.nonzero().squeeze(1)), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(data.test_mask.nonzero().squeeze(1)), batch_size=args.batch_size)

    best_val_acc, best_state, best_epoch = -1.0, None, -1
    logger.info(f'[V6][ASGCN port][seed={seed}] training {args.max_epochs} epochs '
                f'(rank={args.asgcn_rank}, hidden={args.asgcn_hidden}, bs={args.asgcn_batch_size}, '
                f'max_deg={args.asgcn_max_degree}, train_graph={args.asgcn_train_graph}, '
                f'row_norm={args.asgcn_row_norm}, n_train={len(train_ids)})')

    for epoch in range(1, args.max_epochs + 1):
        net.train()
        with tqdm(total=len(train_loader), desc=f'[ASGCN] Epoch {epoch}') as bar:
            for batch in train_loader:
                tgt = batch[0]
                u1, p1, E2 = asgcn_sample_layer(net.sampler, x_tr, a_tr, tgt, args.asgcn_rank,
                                                args.asgcn_dup_cols_zero)
                u0, p0, E1 = asgcn_sample_layer(net.sampler, x_tr, a_tr, u1, args.asgcn_rank,
                                                args.asgcn_dup_cols_zero)
                out, reg = net.forward_sampled(x_tr[u0], x_tr[u1], x_tr[tgt.to(device)], E1, p0, E2, p1)
                loss = loss_fn(out, y_tr[tgt].to(device)) \
                       + args.asgcn_wd * 0.5 * net.W0.pow(2).sum() + args.asgcn_var * reg
                opt.zero_grad(); loss.backward(); opt.step()
                bar.set_postfix({'loss': f'{loss.item():.4f}', 'reg': f'{reg.item():.4f}'}); bar.update()

        if epoch % args.eval_frequency == 0:
            val_out = evaluate_asgcn_native(net, data, a_full, x_full, val_loader)
            val_acc = float((val_out['preds'] == val_out['labels']).mean())
            logger.info(f'[V6][ASGCN][seed={seed}] epoch={epoch} val_acc={val_acc:.4f}')
            if args.use_best_val_checkpoint and val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if args.use_best_val_checkpoint and best_state is not None:
        net.load_state_dict(best_state)
        logger.info(f'[V6][ASGCN][seed={seed}] restored best ckpt (epoch {best_epoch}, val_acc={best_val_acc:.4f})')

    train_out = evaluate_asgcn_native(net, data, a_full, x_full, train_eval_loader)
    val_out = evaluate_asgcn_native(net, data, a_full, x_full, val_loader)
    test_out = evaluate_asgcn_native(net, data, a_full, x_full, test_loader)
    logger.info(f'[V6][ASGCN][seed={seed}] FINAL train_acc={float((train_out["preds"]==train_out["labels"]).mean()):.4f} '
                f'val_acc={float((val_out["preds"]==val_out["labels"]).mean()):.4f} '
                f'test_acc={float((test_out["preds"]==test_out["labels"]).mean()):.4f}')
    return train_out, val_out, test_out, net


@torch.no_grad()
def evaluate_asgcn_native(net, data, a_hat, x, loader):
    """Original AS-GCN test: full (untruncated) neighbourhood, p_u=1, learned attention."""
    net.eval()
    N = data.num_nodes
    full = a_hat.tocoo()
    R = torch.as_tensor(full.row, dtype=torch.long, device=device)
    C = torch.as_tensor(full.col, dtype=torch.long, device=device)
    V = torch.as_tensor(full.data, dtype=torch.float32, device=device)
    ones = torch.ones(N, device=device)
    s1 = net.sampler.edge_support(x, x, R, C, V, ones, 1.0)
    h1 = F.relu(_spmm(R, C, s1, x @ net.W0, N))
    pre1 = h1 @ net.W1
    parts = []
    for batch in loader:
        tgt = batch[0]
        S = a_hat[tgt.numpy()].tocoo()
        r = torch.as_tensor(S.row, dtype=torch.long, device=device)
        c = torch.as_tensor(S.col, dtype=torch.long, device=device)
        v = torch.as_tensor(S.data, dtype=torch.float32, device=device)
        s2 = net.sampler.edge_support(x[tgt.to(device)], x, r, c, v, ones, 1.0)
        out = _spmm(r, c, s2, pre1, tgt.numel())
        parts.append(_summarize(out, data.y[tgt].to(device), tgt))
    net.train()
    return _pack(parts)


# =============================================================================
# Cache-aware fetch. GFN key unchanged (v6b); ASGCN key is new (v6c) -- every ASGCN setting is in it.
# =============================================================================
def _v6_cache_path(policy, args, seed):
    if policy == 'asgcn':
        key = (f'v6c_asgcn_port_{args.dataset}_split{args.split_id}_seed{seed}_ep{args.max_epochs}_'
               f'ah{args.asgcn_hidden}_rk{args.asgcn_rank}_bs{args.asgcn_batch_size}_'
               f'md{args.asgcn_max_degree}_tg{args.asgcn_train_graph}_dup{int(args.asgcn_dup_cols_zero)}_'
               f'var{args.asgcn_var}_wd{args.asgcn_wd}_lr{args.lr_gc}_do{args.dropout}_'
               f'rn{int(args.asgcn_row_norm)}')
    else:
        key = (f'v6b_{policy}_mainstyle_{args.dataset}_split{args.split_id}_seed{seed}_ep{args.max_epochs}_'
               f'h{args.hidden_dim}_ns{args.num_samples}_hops{args.sampling_hops}')
    os.makedirs(OOF_CACHE_DIR, exist_ok=True)
    return os.path.join(OOF_CACHE_DIR, key + '.pkl')


def get_policy_predictions_and_models(args, policy, seed, logger=None):
    logger = logger or get_logger()
    cache_path = _v6_cache_path(policy, args, seed)
    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset, seed=seed, split_id=args.split_id)
    num_indicators = (args.sampling_hops + 1) if args.use_indicators else 0

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            train_out, val_out, test_out, state_dicts = pickle.load(f)
        logger.info(f'[V6] cache HIT policy={policy} seed={seed}')
    else:
        if policy == 'gfn':
            train_out, val_out, test_out, gcn_c, gcn_gf, gcn_z = train_gfn_mainstyle(args, seed, logger=logger)
            state_dicts = {'gcn_gf': {k: v.detach().cpu() for k, v in gcn_gf.state_dict().items()}}
        else:
            train_out, val_out, test_out, net = train_asgcn_native(args, seed, logger=logger)
            state_dicts = {'sampler': {k: v.detach().cpu() for k, v in net.sampler.state_dict().items()}}
        with open(cache_path, 'wb') as f:
            pickle.dump((train_out, val_out, test_out, state_dicts), f)
        logger.info(f'[V6] cache SAVED policy={policy} seed={seed}')

    if policy == 'gfn':
        gcn_gf = GCN(num_features + num_indicators, hidden_dims=[args.hidden_dim, 1]).to(device)
        gcn_gf.load_state_dict(state_dicts['gcn_gf']); gcn_gf.eval()
        models = {'gcn_gf': gcn_gf}
    else:
        sampler = ASGCNSampler(feat_dim=num_features).to(device)
        sampler.load_state_dict(state_dicts['sampler']); sampler.eval()
        models = {'sampler': sampler}
    return train_out, val_out, test_out, models


# =============================================================================
# Jaccard overlap between GFN and ASGCN top-k neighbour choices.
# GFN side: V6 (star graph around v). ASGCN side: new sampler's sample_probs on a_hat
# (single-v approximation: original h_v sums over the whole layer).
# =============================================================================
@torch.no_grad()
def compute_jaccard_overlap(gcn_gf, sampler, data, adjacency, a_hat, node_ids, top_k=5,
                             use_indicators=False, num_indicators=0):
    gcn_gf.eval(); sampler.eval()
    n = len(node_ids)
    jaccard = np.zeros(n, dtype=np.float32)
    data_x = data.x.to(device)
    x_as = asgcn_features(data).to(device)

    for i, v in enumerate(node_ids):
        v = int(v)
        nbrs = adjacency.getrow(v).indices
        if nbrs.size == 0:
            continue
        nbrs_t = torch.as_tensor(nbrs, dtype=torch.long, device=device)
        k = min(top_k, len(nbrs))

        # ASGCN side
        a_vu = torch.as_tensor(a_hat.getrow(v)[:, nbrs].toarray().ravel(), dtype=torch.float32, device=device)
        p = sampler.sample_probs(x_as[v:v + 1], x_as[nbrs_t], a_vu ** 2, len(nbrs))
        top_asgcn = set(nbrs[np.argsort(-p.cpu().numpy())[:k]].tolist())

        # GFN side (V6)
        local_nodes = torch.cat([torch.tensor([v], device=device), nbrs_t])
        local_x = data_x[local_nodes]
        if use_indicators:
            ind = torch.zeros(local_nodes.shape[0], num_indicators, device=device)
            ind[0, -1] = 1.0
            local_x = torch.cat([local_x, ind], dim=1)
        local_edges = torch.stack([
            torch.zeros(len(nbrs), dtype=torch.long, device=device),
            torch.arange(1, len(nbrs) + 1, device=device)])
        node_logits, *_ = gcn_gf(local_x, local_edges)
        gfn_scores = node_logits[1:].squeeze(-1)
        top_gfn = set(nbrs[np.argsort(-gfn_scores.cpu().numpy())[:k]].tolist())

        union = top_asgcn | top_gfn
        jaccard[i] = len(top_asgcn & top_gfn) / len(union) if union else 0.0
    return jaccard


_JAC = {}
def get_test_jaccard(args, seed, data, adjacency, a_hat, model_objs, ids):
    """Test-set jaccard, computed once per (seed, policy-cache) and shared by oracle + router."""
    key = (args.dataset, args.split_id, seed, args.jaccard_top_k,
           _v6_cache_path('gfn', args, seed), _v6_cache_path('asgcn', args, seed))
    if key not in _JAC:
        num_ind = (args.sampling_hops + 1) if args.use_indicators else 0
        _JAC[key] = compute_jaccard_overlap(model_objs['gfn']['gcn_gf'], model_objs['asgcn']['sampler'],
                                            data, adjacency, a_hat, ids, top_k=args.jaccard_top_k,
                                            use_indicators=args.use_indicators, num_indicators=num_ind)
    return _JAC[key]


# =============================================================================
# ORACLE (V6 definition, on TEST) + homophilous->ASGCN / heterophilous->GFN statistics
# =============================================================================
def run_oracle_experiment(args, seeds):
    rows, recs, grp_list, labs = [], [], [], []
    thr = args.homophily_threshold
    for s in seeds:
        print(f'\n{"="*70}\n[Oracle] seed={s}: GFN(main.py-style) vs ASGCN(AS-GCN port)\n{"="*70}')
        logger = get_logger()
        path = os.path.join(os.getcwd(), 'data', args.dataset)
        data0, nf0, nc0 = get_data(root=path, name=args.dataset, seed=s, split_id=args.split_id)
        log_dataset_info(data0, nf0, nc0, logger, tag=f'Oracle seed={s}')

        _, _, out_a, m_a = get_policy_predictions_and_models(args, 'asgcn', s)
        _, _, out_g, m_g = get_policy_predictions_and_models(args, 'gfn', s)
        adjacency = build_adjacency(data0); a_hat = build_a_hat(adjacency)
        props = get_node_props(data0, adjacency)

        common, ia, ig = np.intersect1d(out_a['node_ids'], out_g['node_ids'], return_indices=True)
        assert len(common) == len(out_a['node_ids']) == len(out_g['node_ids'])
        pa, la = out_a['preds'][ia], out_a['labels'][ia]
        pg = out_g['preds'][ig]
        labels = la

        ca, cg = (pa == labels), (pg == labels)
        oracle = ca | cg
        both_c, only_a, only_g, both_w = (ca & cg), (ca & ~cg), (~ca & cg), (~ca & ~cg)

        jac = get_test_jaccard(args, s, data0, adjacency, a_hat, {'gfn': m_g, 'asgcn': m_a}, common)
        rec = build_rec(props, common, pg, pa, out_g['confidence'][ig], out_a['confidence'][ia],
                        out_g['entropy'][ig], out_a['entropy'][ia], jac)
        groups = {'oracle->ASGCN (only ASGCN right)': only_a, 'oracle->GFN (only GFN right)': only_g,
                  'either (both right)': both_c, 'neither (both wrong)': both_w}
        recs.append(rec); grp_list.append(groups); labs.append(labels)

        def frac(m): return float(m.mean()) if m.size else float('nan')
        row = dict(seed=s, acc_asgcn=frac(ca), acc_gfn=frac(cg), acc_oracle=frac(oracle),
                   pct_both_correct=frac(both_c), pct_only_asgcn=frac(only_a),
                   pct_only_gfn=frac(only_g), pct_both_wrong=frac(both_w))
        rows.append(row)
        print(f"[Oracle][seed={s}] ASGCN={row['acc_asgcn']*100:.2f} GFN={row['acc_gfn']*100:.2f} "
              f"Oracle={row['acc_oracle']*100:.2f} | both_correct={row['pct_both_correct']*100:.2f}% "
              f"only_ASGCN={row['pct_only_asgcn']*100:.2f}% only_GFN={row['pct_only_gfn']*100:.2f}% "
              f"both_wrong={row['pct_both_wrong']*100:.2f}%")
        print('   ' + ' | '.join(one_line(k.split(' (')[0], m, rec, thr) for k, m in groups.items()))
        assignment_summary(f'Oracle seed={s}: only-ASGCN-right -> ASGCN, only-GFN-right -> GFN',
                           only_a, only_g, rec, thr)

    def m(k):
        arr = np.array([r[k] for r in rows], dtype=float); return arr.mean(), arr.std()
    print(f'\n{"="*70}\n[Oracle] FINAL (mean±std over seeds={seeds})\n{"="*70}')
    for k, lbl in [('acc_asgcn', 'ASGCN'), ('acc_gfn', 'GFN'), ('acc_oracle', 'Oracle')]:
        mean, std = m(k); print(f'{lbl:<10} {100*mean:6.2f} ± {100*std:.2f}')
    best_single = max(m('acc_asgcn')[0], m('acc_gfn')[0])
    print(f'\n[Oracle] Headroom over best single policy: {100*(m("acc_oracle")[0]-best_single):.2f} pts')

    all_rec = _cat(recs)
    all_groups = {k: np.concatenate([g[k] for g in grp_list]) for k in grp_list[0]}
    all_labels = np.concatenate(labs)
    report_groups(f'ORACLE node assignment, pooled over seeds {seeds}', all_groups, all_rec, thr)
    report_classes('ORACLE per-class assignment (% of class)', all_labels, all_groups)
    assignment_summary(f'ORACLE POOLED over seeds {seeds}: homophilous->ASGCN / heterophilous->GFN',
                       all_groups['oracle->ASGCN (only ASGCN right)'],
                       all_groups['oracle->GFN (only GFN right)'], all_rec, thr)
    return rows


# =============================================================================
# ROUTER -- V6: single-logit head, BCE, Oracle-style target.
# TRAIN on FULL train_mask disagreement set; VALIDATE on FULL val_mask disagreement set;
# DEPLOY hard (0.5) on ALL test_mask nodes.
# =============================================================================
class MixtureRouter(nn.Module):
    def __init__(self, in_dim, gnn_hidden=512, gnn_out=512, dropout=0.0,
                 extra_feat_dim=0, base='sage', agg='mean'):
        super().__init__()
        self.layer1 = GraphConvLayer(in_dim, gnn_hidden, base=base, agg=agg)
        self.layer2 = GraphConvLayer(gnn_hidden, gnn_out, base=base, agg=agg)
        self.dropout = dropout
        self.extra_feat_dim = extra_feat_dim
        self.head = nn.Linear(gnn_out + extra_feat_dim, 1)

    def embed(self, x, edge_index, num_nodes):
        h1 = self.layer1(x, edge_index, num_nodes)
        if self.dropout > 0:
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
        return self.layer2(h1, edge_index, num_nodes)

    def logits(self, node_ids, cached_embeddings, extra_feats=None):
        emb = cached_embeddings[node_ids]
        if self.extra_feat_dim > 0:
            emb = torch.cat([emb, extra_feats], dim=-1)
        return self.head(emb).squeeze(-1)


MIN_DISAGREE = 10
HEAVY_KEYS = ('data', 'ids', 'chosen', 'correct', 'labels', 'preds', 'conf', 'ent', 'jac')


def train_mixture_router(args, seed, policies=('gfn', 'asgcn'), base='sage', agg='mean',
                          use_jaccard=True, logger=None):
    logger = logger or get_logger()
    policies = list(policies)
    assert len(policies) == 2, "BCE router is binary -- exactly 2 policies required"
    torch.manual_seed(seed); np.random.seed(seed)       # reproducible reruns of the best config
    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset, seed=seed, split_id=args.split_id)
    log_dataset_info(data, num_features, num_classes, logger, tag=f'Router seed={seed}')
    adjacency = build_adjacency(data)
    a_hat = build_a_hat(adjacency)
    num_indicators = (args.sampling_hops + 1) if args.use_indicators else 0

    cached, model_objs = {}, {}
    for pol in policies:
        train_out, val_out, test_out, models = get_policy_predictions_and_models(args, pol, seed, logger=logger)
        cached[pol] = {'train': _align_by_node_id(train_out),
                       'val': _align_by_node_id(val_out),
                       'test': _align_by_node_id(test_out)}
        model_objs[pol] = models

    for split in ('train', 'val', 'test'):
        ref_ids = cached[policies[0]][split]['node_ids']
        for pol in policies[1:]:
            assert np.array_equal(cached[pol][split]['node_ids'], ref_ids), f"{split} node_id mismatch: {pol}"

    ref_train_ids = cached[policies[0]]['train']['node_ids']
    ref_val_ids = cached[policies[0]]['val']['node_ids']
    ref_test_ids = cached[policies[0]]['test']['node_ids']

    # ---- Oracle-style binary target: TRAIN split, FULL set ----
    labels_train = cached[policies[0]]['train']['labels']
    c0_train = (cached[policies[0]]['train']['preds'] == labels_train)
    c1_train = (cached[policies[1]]['train']['preds'] == labels_train)
    disagree_train = c0_train != c1_train
    n_disagree_train = int(disagree_train.sum())
    logger.info(f'[V6][seed={seed}] TRAIN disagreement nodes: {n_disagree_train}/{len(labels_train)} '
                f'(only_{policies[0]}={int((c0_train & ~c1_train).sum())} '
                f'only_{policies[1]}={int((~c0_train & c1_train).sum())})')
    if n_disagree_train < MIN_DISAGREE:
        raise RuntimeError(f"[V6][seed={seed}] only {n_disagree_train} TRAIN disagreement nodes -- too few.")
    target_train_all = c1_train.astype(np.float32)

    # ---- Oracle-style binary target: VAL split, FULL set ----
    labels_val = cached[policies[0]]['val']['labels']
    c0_val = (cached[policies[0]]['val']['preds'] == labels_val)
    c1_val = (cached[policies[1]]['val']['preds'] == labels_val)
    disagree_val = c0_val != c1_val
    n_disagree_val = int(disagree_val.sum())
    logger.info(f'[V6][seed={seed}] VAL disagreement nodes: {n_disagree_val}/{len(labels_val)} '
                f'(only_{policies[0]}={int((c0_val & ~c1_val).sum())} '
                f'only_{policies[1]}={int((~c0_val & c1_val).sum())})')
    if n_disagree_val < MIN_DISAGREE:
        raise RuntimeError(f"[V6][seed={seed}] only {n_disagree_val} VAL disagreement nodes -- too few.")
    target_val_all = c1_val.astype(np.float32)

    train_disagree_idx = np.where(disagree_train)[0]
    val_disagree_idx = np.where(disagree_val)[0]

    train_ids_t = torch.as_tensor(ref_train_ids[train_disagree_idx], dtype=torch.long, device=device)
    val_ids_t = torch.as_tensor(ref_val_ids[val_disagree_idx], dtype=torch.long, device=device)
    target_train_t = torch.as_tensor(target_train_all[train_disagree_idx], dtype=torch.float32, device=device)
    target_val_t = torch.as_tensor(target_val_all[val_disagree_idx], dtype=torch.float32, device=device)

    # test jaccard: always computed once (shared via memo) -> used for analysis; feature only if use_jaccard
    jac_test_np = get_test_jaccard(args, seed, data, adjacency, a_hat, model_objs, ref_test_ids)

    extra_dim = 0
    jac_train = jac_val = jac_test = None
    if use_jaccard and 'gfn' in model_objs and 'asgcn' in model_objs:
        logger.info(f'[V6][seed={seed}] computing jaccard-overlap feature '
                    f'(train_disagree={len(train_disagree_idx)}, val_disagree={len(val_disagree_idx)}, '
                    f'test_all={len(ref_test_ids)})')
        kw = dict(top_k=args.jaccard_top_k, use_indicators=args.use_indicators, num_indicators=num_indicators)
        jac_train = compute_jaccard_overlap(model_objs['gfn']['gcn_gf'], model_objs['asgcn']['sampler'],
                                             data, adjacency, a_hat, ref_train_ids[train_disagree_idx], **kw)
        jac_val = compute_jaccard_overlap(model_objs['gfn']['gcn_gf'], model_objs['asgcn']['sampler'],
                                           data, adjacency, a_hat, ref_val_ids[val_disagree_idx], **kw)
        jac_train = torch.as_tensor(jac_train, dtype=torch.float32, device=device).unsqueeze(-1)
        jac_val = torch.as_tensor(jac_val, dtype=torch.float32, device=device).unsqueeze(-1)
        jac_test = torch.as_tensor(jac_test_np, dtype=torch.float32, device=device).unsqueeze(-1)
        extra_dim = 1

    router = MixtureRouter(in_dim=data.num_features, gnn_hidden=args.gnn_hidden, gnn_out=args.gnn_out,
                            dropout=args.router_dropout, extra_feat_dim=extra_dim,
                            base=base, agg=agg).to(device)
    opt = Adam(router.parameters(), lr=args.router_lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    x_dev, ei_dev = data.x.to(device), data.edge_index.to(device)

    best_val_loss, best_state, best_epoch = float('inf'), None, -1
    logger.info(f'[V6][seed={seed}] router base={base} agg={agg} jaccard={use_jaccard} '
                f'gnn_hidden={args.gnn_hidden} gnn_out={args.gnn_out} -- BCE, Oracle-style target, '
                f'TRAIN full disagreement set (n={len(train_disagree_idx)}) -> '
                f'VAL full disagreement set (n={len(val_disagree_idx)})')
    for epoch in range(1, args.router_epochs + 1):
        router.train(); opt.zero_grad()
        emb = router.embed(x_dev, ei_dev, data.num_nodes)
        logit_train = router.logits(train_ids_t, emb, jac_train)
        loss = bce(logit_train, target_train_t)
        loss.backward(); opt.step()

        router.eval()
        with torch.no_grad():
            emb_e = router.embed(x_dev, ei_dev, data.num_nodes)
            logit_val = router.logits(val_ids_t, emb_e, jac_val)
            val_loss = bce(logit_val, target_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}
        if epoch % 50 == 0 or epoch == args.router_epochs:
            logger.info(f'[V6][seed={seed}] epoch={epoch} train_bce={loss.item():.4f} '
                        f'val_bce={val_loss:.4f} (best_val_bce={best_val_loss:.4f} @ep{best_epoch})')

    router.load_state_dict(best_state)
    logger.info(f'[V6][seed={seed}] restored best router (epoch {best_epoch}, val_bce={best_val_loss:.4f})')

    test_ids_t = torch.as_tensor(ref_test_ids, dtype=torch.long, device=device)
    router.eval()
    with torch.no_grad():
        emb_t = router.embed(x_dev, ei_dev, data.num_nodes)
        p1 = torch.sigmoid(router.logits(test_ids_t, emb_t, jac_test))
    chosen = (p1.cpu().numpy() > 0.5).astype(int)

    labels = cached[policies[0]]['test']['labels']
    stacked = np.stack([cached[pol]['test']['preds'] for pol in policies], axis=1)
    final_pred = stacked[np.arange(len(chosen)), chosen]
    acc_router = float((final_pred == labels).mean())
    acc_single = {pol: float((cached[pol]['test']['preds'] == labels).mean()) for pol in policies}
    correctness = np.stack([(cached[pol]['test']['preds'] == labels) for pol in policies], axis=1)
    acc_oracle = float(correctness.any(axis=1).mean())
    usage = {pol: float((chosen == i).mean()) for i, pol in enumerate(policies)}

    test_disagree = correctness[:, 0] != correctness[:, 1]
    router_acc_on_disagree = float((final_pred[test_disagree] == labels[test_disagree]).mean()) \
        if test_disagree.any() else float('nan')

    logger.info(f'[V6][seed={seed}] base={base} agg={agg} jac={use_jaccard} -> Router={100*acc_router:.2f} '
                f'usage={ {p: f"{100*usage[p]:.1f}%" for p in policies} } '
                f'test_disagree_n={int(test_disagree.sum())} router_acc_on_disagree={100*router_acc_on_disagree:.2f}')
    return dict(seed=seed, acc_single=acc_single, acc_router=acc_router, acc_oracle=acc_oracle,
                usage=usage, router_acc_on_disagree=router_acc_on_disagree,
                # extras for node-assignment analysis
                ids=ref_test_ids, chosen=chosen, correct=correctness, labels=labels, jac=jac_test_np,
                preds={p: cached[p]['test']['preds'] for p in policies},
                conf={p: cached[p]['test']['confidence'] for p in policies},
                ent={p: cached[p]['test']['entropy'] for p in policies},
                data=data)


# =============================================================================
# SWEEP
# =============================================================================
def run_sweep(args, seeds, policies=('gfn', 'asgcn')):
    bases = ['gcn', 'sage']
    aggs = ['mean', 'max', 'sum']
    jac_opts = [False, True]

    results = {}
    for base in bases:
        for agg in aggs:
            for use_jac in jac_opts:
                key = (base, agg, use_jac)
                rows = [train_mixture_router(args, seed=s, policies=policies,
                                              base=base, agg=agg, use_jaccard=use_jac)
                        for s in seeds]
                acc = np.array([r['acc_router'] for r in rows], dtype=float)
                results[key] = dict(mean=acc.mean(), std=acc.std(),
                                    rows=[{k: v for k, v in r.items() if k not in HEAVY_KEYS} for r in rows])

    any_rows = results[('gcn', 'mean', False)]['rows']
    single_means = {pol: float(np.mean([r['acc_single'][pol] for r in any_rows])) for pol in policies}
    oracle_mean = float(np.mean([r['acc_oracle'] for r in any_rows]))
    best_single = max(single_means.values())

    print(f'\n{"="*78}\n[V6 SWEEP] 6 architectures x {{no-jaccard, jaccard}} x seeds={seeds}\n{"="*78}')
    for pol in policies:
        print(f'{pol.upper()} (fixed baseline): {100*single_means[pol]:.2f}%')
    print(f'Oracle (fixed baseline): {100*oracle_mean:.2f}%\n')

    hdr = f'{"Base":<8}{"Agg":<8}{"Jaccard":<10}{"Router Acc":>14}{"Gain vs best-single":>22}{"Disagree Acc":>16}'
    print(hdr); print('-' * len(hdr))
    for base in bases:
        for agg in aggs:
            for use_jac in jac_opts:
                r = results[(base, agg, use_jac)]
                gain = r['mean'] - best_single
                dis_mean = np.mean([row['router_acc_on_disagree'] for row in r['rows']])
                print(f'{base:<8}{agg:<8}{str(use_jac):<10}'
                      f'{100*r["mean"]:>10.2f}±{100*r["std"]:<4.2f}{100*gain:>+18.2f} pts'
                      f'{100*dis_mean:>14.2f}%')

    best_key = max(results, key=lambda k: results[k]['mean'])
    print(f'\n[V6 SWEEP] Best config: base={best_key[0]} agg={best_key[1]} jaccard={best_key[2]} '
          f'-> {100*results[best_key]["mean"]:.2f}% ± {100*results[best_key]["std"]:.2f}%')
    return results, best_key


# =============================================================================
# Best-config rerun: how many homophilous nodes the ROUTER sends to ASGCN and
# how many heterophilous nodes it sends to GFN.
# =============================================================================
def run_best_router_report(args, seeds, policies, best_key):
    base, agg, use_jac = best_key
    thr = args.homophily_threshold
    ia, ig = policies.index('asgcn'), policies.index('gfn')
    print(f'\n{"="*78}\n[ROUTER ANALYSIS] rerun best config base={base} agg={agg} jaccard={use_jac}\n{"="*78}')
    recs, grp_list, cgrp_list, labs = [], [], [], []
    for s in seeds:
        d = train_mixture_router(args, seed=s, policies=policies, base=base, agg=agg, use_jaccard=use_jac)
        data = d['data']
        props = get_node_props(data, build_adjacency(data))
        ids, ch, cor = d['ids'], d['chosen'], d['correct']
        rec = build_rec(props, ids, d['preds']['gfn'], d['preds']['asgcn'],
                        d['conf']['gfn'], d['conf']['asgcn'], d['ent']['gfn'], d['ent']['asgcn'], d['jac'])
        to_a, to_g = ch == ia, ch == ig
        ca, cg = cor[:, ia], cor[:, ig]
        router_ok = cor[np.arange(len(ch)), ch]
        cats = {'only ASGCN right': ca & ~cg, 'only GFN right': ~ca & cg,
                'both right': ca & cg, 'both wrong': ~ca & ~cg}
        groups = {'router->ASGCN': to_a, 'router->GFN': to_g}
        for cn, cm in cats.items():
            groups[f'[{cn}] -> ASGCN'] = cm & to_a
            groups[f'[{cn}] -> GFN'] = cm & to_g
        cgroups = {'router->ASGCN': to_a, 'router->GFN': to_g, 'router correct': router_ok,
                   'ASGCN correct': ca, 'GFN correct': cg,
                   'oracle only-ASGCN': cats['only ASGCN right'], 'oracle only-GFN': cats['only GFN right']}
        recs.append(rec); grp_list.append(groups); cgrp_list.append(cgroups); labs.append(d['labels'])
        print(f'[Router][seed={s}] acc={100*d["acc_router"]:.2f} | '
              f'{one_line("->ASGCN", to_a, rec, thr)} | {one_line("->GFN", to_g, rec, thr)}')
        assignment_summary(f'Router seed={s}: ALL test nodes', to_a, to_g, rec, thr)
        dec = ca != cg
        assignment_summary(f'Router seed={s}: DECISIVE nodes only (exactly one policy right)',
                           to_a & dec, to_g & dec, rec, thr)

    all_rec = _cat(recs)
    all_groups = {k: np.concatenate([g[k] for g in grp_list]) for k in grp_list[0]}
    all_cgroups = {k: np.concatenate([g[k] for g in cgrp_list]) for k in cgrp_list[0]}
    all_labels = np.concatenate(labs)
    report_groups(f'ROUTER node assignment (base={base}, agg={agg}, jaccard={use_jac}), pooled over seeds {seeds}',
                  all_groups, all_rec, thr)
    report_classes('ROUTER per-class assignment / accuracy (% of class)', all_labels, all_cgroups)

    to_a_all, to_g_all = all_groups['router->ASGCN'], all_groups['router->GFN']
    dec_all = all_cgroups['ASGCN correct'] != all_cgroups['GFN correct']
    assignment_summary(f'ROUTER POOLED over seeds {seeds}: ALL test nodes (homophilous->ASGCN / heterophilous->GFN)',
                       to_a_all, to_g_all, all_rec, thr)
    assignment_summary(f'ROUTER POOLED over seeds {seeds}: DECISIVE nodes only',
                       to_a_all & dec_all, to_g_all & dec_all, all_rec, thr)
    print('\nReading guide: compare the ROUTER summary with the ORACLE summary above. If the router '
          'follows the oracle, the same homophilous->ASGCN / heterophilous->GFN pattern should appear '
          'in the DECISIVE-nodes block; the [category]->policy rows show where it is right vs wrong.')


if __name__ == '__main__':
    args = Arguments(explicit_bool=True).parse_args()
    if args.config_file is not None:
        args = Arguments(explicit_bool=True, config_files=[args.config_file]).parse_args()
    SEEDS = list(range(args.runs))
    POLICIES = ['gfn', 'asgcn']
    ASGCN_ROW_NORM = args.asgcn_row_norm

    print(f"\n{'='*70}\n[V6] Stage 0: Oracle (seeds={SEEDS}, epochs={args.max_epochs})\n{'='*70}")
    run_oracle_experiment(args, SEEDS)
    print(f"\n{'='*70}\n[V6] Stage 1: GFN(main.py-style) -> ASGCN(AS-GCN port) "
          f"-> Router architecture sweep (12 configs)\n{'='*70}")
    _, best_key = run_sweep(args, SEEDS, policies=POLICIES)
    run_best_router_report(args, SEEDS, POLICIES, best_key)
