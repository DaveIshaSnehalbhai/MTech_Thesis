"""
main.py -- V9. GFN (grapes main.py original) and ASGCN (torch port of huangwb/AS-GCN
GCNAdapt + SamplerAdapt) are two independent frozen policies. A router
(base in {gcn,sage} x agg in {mean,max,sum} x {no-jaccard, jaccard}) is trained per-node on
val_mask (soft weighted-CE) and deployed hard (argmax) on test_mask.
Analysis: oracle / router node-assignment stats split by homophily, class and node properties.

ASGCN port is aligned with the original TF code (run_pubmed.py / sampler.py / models.py /
layers.py / utils.prepare_pubmed / sparse_tensor_utils.py). See the "ASGCN" section below for the
list of things that are replicated and the (documented) deliberate deviations.
"""
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

from modules.data import get_data
from modules.gcn import GCN
from modules.utils import (TensorMap, get_logger, get_neighborhoods,
                            sample_neighborhoods_from_probs, slice_adjacency)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OOF_CACHE_DIR = os.environ.get('RAHAS_OOF_CACHE_DIR', 'oof_cache')


class Arguments(Tap):
    dataset: str = 'cora'
    sampling_hops: int = 2
    num_samples: int = 16           # GFN: per node (grapes default)
    use_indicators: bool = True
    lr_gf: float = 1e-4
    lr_gc: float = 1e-3
    loss_coef: float = 1e4          # GFN TB loss coefficient
    log_z_init: float = 0.
    reg_param: float = 0.
    dropout: float = 0.0
    hidden_dim: int = 256
    max_epochs: int = 50
    batch_size: int = 512
    eval_frequency: int = 10
    eval_full_batch: bool = True    # grapes default; False = sampled GFN eval
    runs: int = 3
    split_id: int = 0
    seed: int = None
    config_file: str = None
    use_best_val_checkpoint: bool = True
    # ASGCN (original run_pubmed.py defaults: hidden1=16, rank=128, var=0.5, weight_decay=5e-4,
    #        max_degree=32, minibatch=256, lr=1e-3, dropout=0.0)
    asgcn_hidden: int = 16
    asgcn_rank: int = 128
    asgcn_var: float = 0.5
    asgcn_wd: float = 5e-4
    asgcn_batch_size: int = 256     # original uses 256 (GFN keeps batch_size=512)
    asgcn_max_degree: int = 32      # original: <=32 random neighbours kept per node at TRAIN time (fixed once); <=0 disables
    asgcn_train_graph: str = 'full' # 'full' = sample on full graph (transductive); 'induced' = original: subgraph of train nodes
    asgcn_dup_cols_zero: bool = True  # original gather_columns quirk: repeated draw keeps support only in its first column
    # router
    gnn_hidden: int = 64
    gnn_out: int = 32
    router_epochs: int = 300
    router_lr: float = 1e-3
    router_dropout: float = 0.0
    val_frac_for_selection: float = 0.2
    jaccard_top_k: int = 5
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
# Node properties (for studying WHY a node goes to ASGCN or GFN)
# =============================================================================
def _core_numbers(A):
    """k-core by iterative peeling (A: symmetric binary CSR, no self loops)."""
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
    return (r * N).astype(np.float32)          # 1.0 = average node


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
        # homophily + feature cosine from the SAME cleaned graph as deg
        h = np.bincount(r, weights=(y[r] == y[c]).astype(np.float64), minlength=N) * dn
        x = data.x.cpu().float()
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        rt, ct = torch.as_tensor(r), torch.as_tensor(c)
        cos = np.bincount(r, weights=(xn[rt] * xn[ct]).sum(1).numpy(), minlength=N) * dn
        # neighbour label distribution -> entropy + majority label
        Y = np.eye(int(y.max()) + 1, dtype=np.float32)[y]
        cnt = A @ Y
        p = cnt * dn[:, None]
        lab_ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
        nbr_major = cnt.argmax(1); nbr_major[deg == 0] = -1
        # clustering coefficient and TRUE 2-hop size (within 2 hops, excluding self)
        A2 = (A @ A).tocsr()
        tri = np.asarray(A2.multiply(A).sum(1)).ravel() / 2.0
        clust = np.where(deg >= 2, tri / np.maximum(deg * (deg - 1) / 2.0, 1), 0.0)
        B = (A2 + A).tocsr()
        B = (B - sp.diags(B.diagonal())).tocsr(); B.eliminate_zeros()
        two_hop = np.diff(B.indptr).astype(np.float32)
        # neighbour feature variance (float64): E||x_u||^2 - ||E x_u||^2
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
            print(f'{name:<38}' + ''.join(f'{(_nm(rec[k][m].astype(np.float64)) if m.any() else float("nan")):>12.3f}' for k in chunk))


def report_classes(title, labels, cgroups):
    """Per node class: N and % of the class falling in each group."""
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


# =============================================================================
# ASGCN -- torch port of AS-GCN (sampler.py SamplerAdapt + models.py GCNAdapt)
#
# Replicated from the original:
#   * features are row-normalised (utils.nontuple_preprocess_features)
#   * adjacency = D^-1/2 (I+A) D^-1/2 (+ extra identity for pubmed, utils.prepare_pubmed)
#   * TRAIN-time neighbour lists are truncated to `max_degree` random neighbours, fixed once
#     (utils.compute_adjlist); values stay as in the normalised adjacency (not renormalised)
#   * optional train-induced subgraph (utils.prepare_pubmed: adj[train][:,train])
#   * sampler: p ∝ sqrt(sum_v A_vu^2) * attention * g_u, draws WITH replacement (tf.multinomial)
#   * gather_columns quirk: a repeatedly drawn node keeps its support only in its FIRST column
#   * model: relu(S1 X W0) -> S2 H W1, S = A_vu * (relu(h_v+h_u)+1)/n_u / p_u, no bias,
#            weight decay only on W0, variance regulariser only from the last layer
#   * one Adam over W0, W1, w_s ; sampling is a constant w.r.t. the loss (placeholders in TF)
#   * TEST: full graph, p_u = 1, learned attention
# Deliberate deviation (kept): at test time the 1/n_u factor (n_u = N in the original) is skipped.
#   It is a positive rescale of the logits (no bias => identical argmax / accuracy), and it keeps
#   the logits on the training scale (E[sum_j A att/(k p)] = sum_u A att), which the router needs
#   because it consumes true_prob / confidence / entropy. With 1/N^2 the softmax would be ~uniform.
# =============================================================================
def asgcn_features(data):
    """Row-normalise features like the original (r_inv = 1/rowsum, inf -> 0). Idempotent on already
    row-normalised input. Skipped if features contain negatives (e.g. standardised features)."""
    x = data.x.float()
    if float(x.min()) < 0:
        return x
    rs = x.sum(1, keepdim=True)
    return x * torch.where(rs > 0, 1.0 / rs, torch.zeros_like(rs))


def asgcn_a_hat(args, adjacency):
    """Original adjacency preprocessing; pubmed gets an extra identity (prepare_pubmed)."""
    A = build_a_hat(adjacency)
    if args.dataset.lower() == 'pubmed':
        A = (A + sp.eye(A.shape[0], dtype=np.float32, format='csr')).tocsr()
    return A.astype(np.float32)


def truncate_rows(A, max_deg, rng):
    """compute_adjlist: rows with > max_deg entries keep a random max_deg of them (no replacement),
    values untouched. Done once, before training."""
    A = A.tocsr()
    if max_deg is None or max_deg <= 0:
        return A
    deg = np.diff(A.indptr)
    keep = np.ones(A.nnz, dtype=bool)
    for v in np.nonzero(deg > max_deg)[0]:
        s, e = A.indptr[v], A.indptr[v + 1]
        keep[rng.choice(np.arange(s, e), e - s - max_deg, replace=False)] = False
    rows = np.repeat(np.arange(A.shape[0]), deg)
    out = sp.csr_matrix((A.data[keep], (rows[keep], A.indices[keep])), shape=A.shape)
    return out


class ASGCNSampler(nn.Module):
    """w_s [d,2], shared between sampler and model, as in the original."""
    def __init__(self, feat_dim):
        super().__init__()
        self.w_s = nn.Parameter(torch.empty(feat_dim, 2))
        nn.init.xavier_uniform_(self.w_s)

    def sample_probs(self, x_v, x_u, colsq, k):
        """Sampler.one_layer_sampling: p ∝ sqrt(sum_v A_vu^2) * attention * g_u, normalised.
        h_v is a SCALAR sum over all v in the layer; sampler uses w_s[:,1] for v, w_s[:,0] for u."""
        h_v = (x_v @ self.w_s[:, 1]).sum()
        h_u = x_u @ self.w_s[:, 0]
        attention = (1.0 / k) * (F.relu(h_v + h_u) + 1.0)
        g_u = F.relu(h_u) + 1.0
        p = colsq.sqrt() * attention * g_u
        return p / p.sum()

    def edge_support(self, x_v, x_u, rows, cols, vals, p_u, n_u):
        """GCNAdapt._attention on the nonzeros: A_vu * (1/n_u)(relu(h_v+h_u)+1) / p_u.
        The model uses w_s[:,0] for v and w_s[:,1] for u (swapped vs the sampler, as in the original)."""
        h_v = x_v @ self.w_s[:, 0]
        h_u = x_u @ self.w_s[:, 1]
        att = (F.relu(h_v[rows] + h_u[cols]) + 1.0) / n_u
        return vals * att / p_u[cols]


def _spmm(rows, cols, w, dense, n_rows):
    return torch.zeros(n_rows, dense.shape[1], device=dense.device, dtype=dense.dtype) \
        .index_add_(0, rows, w.unsqueeze(1) * dense[cols])


class ASGCNNet(nn.Module):
    """GCNAdapt: relu(S1 X W0) -> S2 H W1 (+ variance regulariser), no bias."""
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
        # GraphSampleConvolutionReg
        ms = torch.zeros(n1, device=x0.device).index_add_(0, c2, s2) / B
        diff = ms.unsqueeze(1) * pre1 - out.mean(0, keepdim=True)
        reg = (diff ** 2).sum() / (n1 * out.shape[1])
        return out, reg


def asgcn_sample_layer(sampler, x_dev, a_hat, v_ids, k, dup_first_only=True):
    """Layer-wise sampling, k draws WITH replacement. Returns sampled global ids,
    p_u of the draws, and the column-gathered support (rows, cols, vals) [|v| x k].
    dup_first_only=True reproduces sparse_tensor_utils.gather_columns: if the same u is drawn several
    times, only its first column carries the support (the repeated columns are empty)."""
    sub = a_hat[v_ids.cpu().numpy()]                    # duplicate v rows kept, as in the original
    cand = np.unique(sub.indices)
    sub_c = sub[:, cand].tocsc()
    colsq = torch.as_tensor(np.asarray(sub_c.multiply(sub_c).sum(0)).ravel(),
                            dtype=torch.float32, device=device)
    cand_t = torch.as_tensor(cand, dtype=torch.long, device=device)
    with torch.no_grad():                               # original feeds samples/probs as constants
        p = sampler.sample_probs(x_dev[v_ids.to(device)], x_dev[cand_t], colsq, k)
        idx = torch.multinomial(p, k, replacement=True)
    idx_np = idx.cpu().numpy()
    S = sub_c[:, idx_np].tocoo()
    r, c, d = S.row, S.col, S.data
    if dup_first_only:
        first = np.zeros(k, dtype=bool)
        first[np.unique(idx_np, return_index=True)[1]] = True   # first occurrence of every drawn node
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

    net = ASGCNNet(data.num_features, args.asgcn_hidden, num_classes, args.dropout).to(device)
    opt = Adam(net.parameters(), lr=args.lr_gc)      # one optimiser over W0, W1, w_s (as original)
    loss_fn = nn.CrossEntropyLoss()

    adjacency = build_adjacency(data)
    x_full = asgcn_features(data).to(device)         # row-normalised, used for test-time propagation
    a_full = asgcn_a_hat(args, adjacency)            # full graph, never truncated (original test adj)

    tr_global = data.train_mask.nonzero().squeeze(1)
    if args.asgcn_train_graph == 'induced':          # original: adj_train = adj[train][:, train]
        t = tr_global.numpy()
        a_tr = asgcn_a_hat(args, adjacency[t][:, t])
        x_tr = x_full[tr_global.to(device)]
        y_tr = data.y[tr_global]
        train_ids = torch.arange(len(tr_global))
    else:
        a_tr, x_tr, y_tr, train_ids = a_full, x_full, data.y, tr_global
    a_tr = truncate_rows(a_tr, args.asgcn_max_degree,
                         np.random.RandomState(None if seed is None else int(seed)))

    train_loader = DataLoader(TensorDataset(train_ids), batch_size=args.asgcn_batch_size,
                              shuffle=True)          # original shuffles, minibatch 256
    val_loader = DataLoader(TensorDataset(data.val_mask.nonzero().squeeze(1)), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(data.test_mask.nonzero().squeeze(1)), batch_size=args.batch_size)

    best_val_acc, best_state, best_epoch = -1.0, None, -1
    logger.info(f'[V9][ASGCN][seed={seed}] training {args.max_epochs} epochs '
                f'(rank={args.asgcn_rank}, hidden={args.asgcn_hidden}, bs={args.asgcn_batch_size}, '
                f'max_deg={args.asgcn_max_degree}, train_graph={args.asgcn_train_graph}, '
                f'n_train={len(train_ids)})')

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
            logger.info(f'[V9][ASGCN][seed={seed}] epoch={epoch} val_acc={val_acc:.4f}')
            if args.use_best_val_checkpoint and val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if args.use_best_val_checkpoint and best_state is not None:
        net.load_state_dict(best_state)
        logger.info(f'[V9][ASGCN][seed={seed}] restored best ckpt (epoch {best_epoch}, val_acc={best_val_acc:.4f})')

    val_out = evaluate_asgcn_native(net, data, a_full, x_full, val_loader)
    test_out = evaluate_asgcn_native(net, data, a_full, x_full, test_loader)
    logger.info(f'[V9][ASGCN][seed={seed}] FINAL test_acc={float((test_out["preds"]==test_out["labels"]).mean()):.4f}')
    return val_out, test_out, net


@torch.no_grad()
def evaluate_asgcn_native(net, data, a_hat, x, loader):
    """Original AS-GCN test: full (untruncated) neighbourhood, p_u=1, learned attention, row-normalised
    features. The 1/n_u factor is skipped (positive rescale of logits; no bias => argmax unchanged,
    true_prob stays on the train scale)."""
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
# ROUTER -- base {gcn, sage} x aggregator {mean, max, sum}
# =============================================================================
class GraphConvLayer(nn.Module):
    """base='sage': separate W_self / W_neigh. base='gcn': single shared W over (x_self + agg(neighbours)).
    agg is swappable independently of base."""
    def __init__(self, in_dim, out_dim, base='gcn', agg='mean'):
        super().__init__()
        assert base in ('gcn', 'sage') and agg in ('mean', 'max', 'sum')
        self.base, self.agg = base, agg
        if base == 'sage':
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim)
        else:
            self.lin = nn.Linear(in_dim, out_dim)

    def _aggregate(self, x, src, dst, num_nodes):
        d = x.shape[1]
        msg = x[src]
        if self.agg == 'sum':
            out = torch.zeros(num_nodes, d, device=x.device, dtype=x.dtype)
            out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d), msg)
            return out
        if self.agg == 'mean':
            out = torch.zeros(num_nodes, d, device=x.device, dtype=x.dtype)
            out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d), msg)
            deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            return out / deg.clamp(min=1).unsqueeze(-1)
        out = torch.full((num_nodes, d), float('-inf'), device=x.device, dtype=x.dtype)
        out.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, d), msg, reduce='amax', include_self=False)
        return torch.where(torch.isinf(out), torch.zeros_like(out), out)

    def forward(self, x, edge_index, num_nodes):
        agg = self._aggregate(x, edge_index[0], edge_index[1], num_nodes)
        if self.base == 'sage':
            return F.relu(self.lin_self(x) + self.lin_neigh(agg))
        return F.relu(self.lin(x + agg))


class MixtureRouter(nn.Module):
    """2-layer GraphConv(base, agg) -> [+ optional jaccard] -> Linear(K). TRAIN soft, DEPLOY hard."""
    def __init__(self, in_dim, n_policies=2, gnn_hidden=64, gnn_out=32, dropout=0.0,
                 extra_feat_dim=0, base='gcn', agg='mean'):
        super().__init__()
        self.layer1 = GraphConvLayer(in_dim, gnn_hidden, base=base, agg=agg)
        self.layer2 = GraphConvLayer(gnn_hidden, gnn_out, base=base, agg=agg)
        self.dropout = dropout
        self.extra_feat_dim = extra_feat_dim
        self.head = nn.Linear(gnn_out + extra_feat_dim, n_policies)

    def embed(self, x, edge_index, num_nodes):
        h1 = self.layer1(x, edge_index, num_nodes)
        if self.dropout > 0:
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
        return self.layer2(h1, edge_index, num_nodes)

    def weights(self, node_ids, cached_embeddings, extra_feats=None):
        emb = cached_embeddings[node_ids]
        if self.extra_feat_dim > 0:
            emb = torch.cat([emb, extra_feats], dim=-1)
        return F.softmax(self.head(emb), dim=-1)


# =============================================================================
# POLICY 1: GFN -- grapes main.py ORIGINAL (gcn_gf + gcn_z, trajectory balance)
# =============================================================================
def train_gfn_mainstyle(args, seed, logger=None):
    logger = logger or get_logger()
    if args.loss_coef < 10:
        logger.info(f'[V9][GFN] WARNING: loss_coef={args.loss_coef} looks too small for the TB loss.')

    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset, seed=seed, split_id=args.split_id)
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

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
    logger.info(f'[V9][GFN][seed={seed}] training {args.max_epochs} epochs')

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
            logger.info(f'[V9][GFN][seed={seed}] epoch={epoch} val_acc={val_acc:.4f}')
            if args.use_best_val_checkpoint and val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                best_state = {'gcn_c': {k: v.detach().cpu().clone() for k, v in gcn_c.state_dict().items()},
                               'gcn_gf': {k: v.detach().cpu().clone() for k, v in gcn_gf.state_dict().items()},
                               'gcn_z': {k: v.detach().cpu().clone() for k, v in gcn_z.state_dict().items()}}

    if args.use_best_val_checkpoint and best_state is not None:
        gcn_c.load_state_dict(best_state['gcn_c']); gcn_gf.load_state_dict(best_state['gcn_gf'])
        gcn_z.load_state_dict(best_state['gcn_z'])
        logger.info(f'[V9][GFN][seed={seed}] restored best ckpt (epoch {best_epoch}, val_acc={best_val_acc:.4f})')

    val_out = evaluate_gfn_mainstyle(gcn_c, gcn_gf, data, args, adjacency, num_indicators, val_loader)
    test_out = evaluate_gfn_mainstyle(gcn_c, gcn_gf, data, args, adjacency, num_indicators, test_loader)
    logger.info(f'[V9][GFN][seed={seed}] FINAL test_acc={float((test_out["preds"]==test_out["labels"]).mean()):.4f}')
    return val_out, test_out, gcn_c, gcn_gf, gcn_z


@torch.no_grad()
def evaluate_gfn_mainstyle(gcn_c, gcn_gf, data, args, adjacency, num_indicators, loader):
    gcn_c.eval(); gcn_gf.eval()
    parts = []

    if args.eval_full_batch:       # grapes default: classifier on the full graph, no sampling
        ei = data.edge_index.to(device)
        logits, *_ = gcn_c(data.x.to(device), [ei] * args.sampling_hops)
        for batch in loader:
            tgt = batch[0]
            parts.append(_summarize(logits[tgt.to(device)], data.y[tgt].to(device), tgt))
        gcn_c.train(); gcn_gf.train()
        return _pack(parts)

    node_map = TensorMap(size=data.num_nodes)
    prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    indicator_features = torch.zeros((data.num_nodes, num_indicators))
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
        parts.append(_summarize(logits[local_target_ids], data.y[target_nodes].to(device), target_nodes))

    gcn_c.train(); gcn_gf.train()
    return _pack(parts)


# =============================================================================
# Cache-aware fetch (predictions + sampler weights, so jaccard works on cache hits)
# =============================================================================
def _cache_path(policy, args, seed):
    if policy == 'asgcn':
        # every setting that changes the ASGCN policy is part of the key (v9a = corrected port;
        # old v8_asgcn_* pickles were produced by the previous port and are never loaded)
        key = (f'v9a_asgcn_{args.dataset}_split{args.split_id}_seed{seed}_ep{args.max_epochs}_'
               f'ah{args.asgcn_hidden}_rk{args.asgcn_rank}_bs{args.asgcn_batch_size}_'
               f'md{args.asgcn_max_degree}_tg{args.asgcn_train_graph}_dup{int(args.asgcn_dup_cols_zero)}_'
               f'var{args.asgcn_var}_wd{args.asgcn_wd}_lr{args.lr_gc}_do{args.dropout}')
    else:
        key = (f'v8_{policy}_{args.dataset}_split{args.split_id}_seed{seed}_ep{args.max_epochs}_'
               f'h{args.hidden_dim}_ns{args.num_samples}_hops{args.sampling_hops}_lc{args.loss_coef}_'
               f'fb{int(args.eval_full_batch)}_ah{args.asgcn_hidden}_rk{args.asgcn_rank}')
    os.makedirs(OOF_CACHE_DIR, exist_ok=True)
    return os.path.join(OOF_CACHE_DIR, key + '.pkl')


def get_policy_predictions_and_models(args, policy, seed, logger=None):
    logger = logger or get_logger()
    cache_path = _cache_path(policy, args, seed)
    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, _, num_classes = get_data(root=path, name=args.dataset, seed=seed, split_id=args.split_id)
    num_indicators = (args.sampling_hops + 1) if args.use_indicators else 0

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            val_out, test_out, state_dicts = pickle.load(f)
        logger.info(f'[V9] cache HIT policy={policy} seed={seed}')
    else:
        if policy == 'gfn':
            val_out, test_out, gcn_c, gcn_gf, gcn_z = train_gfn_mainstyle(args, seed, logger=logger)
            state_dicts = {'gcn_gf': {k: v.detach().cpu() for k, v in gcn_gf.state_dict().items()}}
        else:
            val_out, test_out, net = train_asgcn_native(args, seed, logger=logger)
            state_dicts = {'sampler': {k: v.detach().cpu() for k, v in net.sampler.state_dict().items()}}
        with open(cache_path, 'wb') as f:
            pickle.dump((val_out, test_out, state_dicts), f)

    if policy == 'gfn':
        gcn_gf = GCN(data.num_features + num_indicators, hidden_dims=[args.hidden_dim, 1]).to(device)
        gcn_gf.load_state_dict(state_dicts['gcn_gf']); gcn_gf.eval()
        models = {'gcn_gf': gcn_gf}
    else:
        sampler = ASGCNSampler(feat_dim=data.num_features).to(device)
        sampler.load_state_dict(state_dicts['sampler']); sampler.eval()
        models = {'sampler': sampler}
    return val_out, test_out, models


# =============================================================================
# Jaccard overlap between GFN and ASGCN top-k neighbour choices (diagnostic feature).
# GFN side = exact first-hop scoring pipeline of training (get_neighborhoods + indicators).
# ASGCN side = sampler probs for a single v (approximation: original h_v sums over the layer),
#              computed on the same row-normalised features the sampler was trained with.
# =============================================================================
@torch.no_grad()
def compute_jaccard_overlap(gcn_gf, sampler, data, adjacency, a_hat, node_ids, top_k=5,
                             use_indicators=False, num_indicators=0):
    gcn_gf.eval(); sampler.eval()
    jac = np.zeros(len(node_ids), dtype=np.float32)
    x_all = asgcn_features(data).to(device)

    for i, v in enumerate(node_ids):
        v = int(v)
        nbrs = adjacency.getrow(v).indices
        nbrs = nbrs[nbrs != v]
        if nbrs.size == 0:
            continue
        k = min(top_k, len(nbrs))
        nbrs_t = torch.as_tensor(nbrs, dtype=torch.long)

        a_vu = torch.as_tensor(a_hat.getrow(v)[:, nbrs].toarray().ravel(), dtype=torch.float32, device=device)
        p = sampler.sample_probs(x_all[v:v + 1], x_all[nbrs_t.to(device)], a_vu ** 2, len(nbrs))
        top_a = set(nbrs[np.argsort(-p.cpu().numpy())[:k]].tolist())

        nb = get_neighborhoods(torch.tensor([v]), adjacency)
        nodes = torch.unique(nb.reshape(-1))
        local = torch.searchsorted(nodes, nb).to(device)
        x = data.x[nodes]
        if use_indicators:
            ind = torch.zeros(len(nodes), num_indicators)
            ind[nodes != v, 0] = 1.0
            ind[nodes == v, -1] = 1.0
            x = torch.cat([x, ind], dim=1)
        logits, *_ = gcn_gf(x.to(device), local)
        gs = logits.squeeze(-1)[torch.searchsorted(nodes, nbrs_t).to(device)].cpu().numpy()
        top_g = set(nbrs[np.argsort(-gs)[:k]].tolist())

        union = top_a | top_g
        jac[i] = len(top_a & top_g) / len(union) if union else 0.0
    return jac


_JAC = {}
def get_jaccard(args, seed, split, data, adjacency, a_hat, model_objs, ids):
    key = (args.dataset, args.split_id, seed, split, args.jaccard_top_k,
           _cache_path('gfn', args, seed), _cache_path('asgcn', args, seed))
    if key not in _JAC:
        num_ind = (args.sampling_hops + 1) if args.use_indicators else 0
        _JAC[key] = compute_jaccard_overlap(model_objs['gfn']['gcn_gf'], model_objs['asgcn']['sampler'],
                                            data, adjacency, a_hat, ids, top_k=args.jaccard_top_k,
                                            use_indicators=args.use_indicators, num_indicators=num_ind)
    return _JAC[key]


# =============================================================================
# ORACLE (+ node-assignment statistics)
# =============================================================================
def run_oracle_experiment(args, seeds):
    rows, recs, grp_list, labs = [], [], [], []
    thr = args.homophily_threshold
    for s in seeds:
        print(f'\n{"="*70}\n[Oracle] seed={s}: GFN(main.py) vs ASGCN(AS-GCN port)\n{"="*70}')
        _, out_a, m_a = get_policy_predictions_and_models(args, 'asgcn', s)
        _, out_g, m_g = get_policy_predictions_and_models(args, 'gfn', s)
        path = os.path.join(os.getcwd(), 'data', args.dataset)
        data, _, _ = get_data(root=path, name=args.dataset, seed=s, split_id=args.split_id)
        adjacency = build_adjacency(data); a_hat = build_a_hat(adjacency)
        props = get_node_props(data, adjacency)

        common, ia, ig = np.intersect1d(out_a['node_ids'], out_g['node_ids'], return_indices=True)
        assert len(common) == len(out_a['node_ids']) == len(out_g['node_ids'])
        labels = out_a['labels'][ia]
        pa, pg = out_a['preds'][ia], out_g['preds'][ig]
        ca, cg = (pa == labels), (pg == labels)
        oracle = ca | cg
        both_c, only_a, only_g, both_w = (ca & cg), (ca & ~cg), (~ca & cg), (~ca & ~cg)

        jac = get_jaccard(args, s, 'test', data, adjacency, a_hat, {'gfn': m_g, 'asgcn': m_a}, common)
        rec = build_rec(props, common, pg, pa, out_g['confidence'][ig], out_a['confidence'][ia],
                        out_g['entropy'][ig], out_a['entropy'][ia], jac)
        groups = {'oracle->ASGCN (only ASGCN right)': only_a, 'oracle->GFN (only GFN right)': only_g,
                  'either (both right)': both_c, 'neither (both wrong)': both_w}
        recs.append(rec); grp_list.append(groups); labs.append(labels)

        def frac(m): return float(m.mean()) if m.size else float('nan')
        row = dict(seed=s, acc_asgcn=frac(ca), acc_gfn=frac(cg), acc_oracle=frac(oracle))
        rows.append(row)
        print(f"[Oracle][seed={s}] ASGCN={row['acc_asgcn']*100:.2f} GFN={row['acc_gfn']*100:.2f} "
              f"Oracle={row['acc_oracle']*100:.2f}")
        print('   ' + ' | '.join(one_line(k.split(' (')[0], m, rec, thr) for k, m in groups.items()))

    def m(k):
        arr = np.array([r[k] for r in rows], dtype=float); return arr.mean(), arr.std()
    print(f'\n{"="*70}\n[Oracle] FINAL (mean±std over seeds={seeds})\n{"="*70}')
    for k, lbl in [('acc_asgcn', 'ASGCN'), ('acc_gfn', 'GFN'), ('acc_oracle', 'Oracle')]:
        mean, std = m(k); print(f'{lbl:<10} {100*mean:6.2f} ± {100*std:.2f}')
    print(f'Headroom over best single policy: {100*(m("acc_oracle")[0]-max(m("acc_asgcn")[0], m("acc_gfn")[0])):.2f} pts')

    all_rec = _cat(recs)
    all_groups = {k: np.concatenate([g[k] for g in grp_list]) for k in grp_list[0]}
    all_labels = np.concatenate(labs)
    report_groups(f'ORACLE node assignment, pooled over seeds {seeds}', all_groups, all_rec, thr)
    report_classes('ORACLE per-class assignment (% of class)', all_labels, all_groups)
    return rows


# =============================================================================
# Router training (returns per-node details for analysis)
# =============================================================================
def train_mixture_router(args, seed, policies=('gfn', 'asgcn'), base='gcn', agg='mean',
                          use_jaccard=True, logger=None):
    logger = logger or get_logger()
    policies = list(policies)
    torch.manual_seed(seed); np.random.seed(seed)
    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, _, _ = get_data(root=path, name=args.dataset, seed=seed, split_id=args.split_id)
    adjacency = build_adjacency(data)
    a_hat = build_a_hat(adjacency)

    cached, model_objs = {}, {}
    for pol in policies:
        val_out, test_out, models = get_policy_predictions_and_models(args, pol, seed, logger=logger)
        cached[pol] = {'val': _align_by_node_id(val_out), 'test': _align_by_node_id(test_out)}
        model_objs[pol] = models

    ref_val_ids = cached[policies[0]]['val']['node_ids']
    ref_test_ids = cached[policies[0]]['test']['node_ids']
    for pol in policies[1:]:
        assert np.array_equal(cached[pol]['val']['node_ids'], ref_val_ids)
        assert np.array_equal(cached[pol]['test']['node_ids'], ref_test_ids)

    # test jaccard is always computed (for analysis); used as router feature only if use_jaccard
    jac_test_np = get_jaccard(args, seed, 'test', data, adjacency, a_hat, model_objs, ref_test_ids)
    extra_dim, jac_val, jac_test = 0, None, None
    if use_jaccard:
        jac_val_np = get_jaccard(args, seed, 'val', data, adjacency, a_hat, model_objs, ref_val_ids)
        jac_val = torch.as_tensor(jac_val_np, dtype=torch.float32, device=device).unsqueeze(-1)
        jac_test = torch.as_tensor(jac_test_np, dtype=torch.float32, device=device).unsqueeze(-1)
        extra_dim = 1

    val_ids_t = torch.as_tensor(ref_val_ids, dtype=torch.long, device=device)
    ce_val = torch.stack([
        torch.as_tensor(-np.log(np.clip(cached[pol]['val']['true_prob'], 1e-8, 1.0)), dtype=torch.float32)
        for pol in policies], dim=1).to(device)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(val_ids_t.shape[0], generator=g)
    n_sel = max(1, int(round(args.val_frac_for_selection * val_ids_t.shape[0])))
    select_idx, fit_idx = perm[:n_sel], perm[n_sel:]

    router = MixtureRouter(in_dim=data.num_features, n_policies=len(policies),
                            gnn_hidden=args.gnn_hidden, gnn_out=args.gnn_out,
                            dropout=args.router_dropout, extra_feat_dim=extra_dim,
                            base=base, agg=agg).to(device)
    opt = Adam(router.parameters(), lr=args.router_lr, weight_decay=1e-4)
    x_dev, ei_dev = data.x.to(device), data.edge_index.to(device)

    best_sel_loss, best_state, best_epoch = float('inf'), None, -1
    logger.info(f'[V9][seed={seed}] router base={base} agg={agg} jaccard={use_jaccard}')
    for epoch in range(1, args.router_epochs + 1):
        router.train(); opt.zero_grad()
        emb = router.embed(x_dev, ei_dev, data.num_nodes)
        extra_fit = jac_val[fit_idx] if extra_dim else None
        w_fit = router.weights(val_ids_t[fit_idx], emb, extra_fit)
        loss = (w_fit * ce_val[fit_idx]).sum(dim=1).mean()
        loss.backward(); opt.step()

        router.eval()
        with torch.no_grad():
            emb_e = router.embed(x_dev, ei_dev, data.num_nodes)
            extra_sel = jac_val[select_idx] if extra_dim else None
            w_sel = router.weights(val_ids_t[select_idx], emb_e, extra_sel)
            sel_loss = (w_sel * ce_val[select_idx]).sum(dim=1).mean().item()
        if sel_loss < best_sel_loss:
            best_sel_loss, best_epoch = sel_loss, epoch
            best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}

    if best_state is None:          # sel_loss never finite/improved -> keep the last weights
        best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}
    router.load_state_dict(best_state)

    test_ids_t = torch.as_tensor(ref_test_ids, dtype=torch.long, device=device)
    router.eval()
    with torch.no_grad():
        emb_t = router.embed(x_dev, ei_dev, data.num_nodes)
        w_test = router.weights(test_ids_t, emb_t, jac_test)
    chosen = w_test.argmax(dim=1).cpu().numpy()

    labels = cached[policies[0]]['test']['labels']
    stacked = np.stack([cached[pol]['test']['preds'] for pol in policies], axis=1)
    final_pred = stacked[np.arange(len(chosen)), chosen]
    acc_router = float((final_pred == labels).mean())
    acc_single = {pol: float((cached[pol]['test']['preds'] == labels).mean()) for pol in policies}
    correctness = np.stack([(cached[pol]['test']['preds'] == labels) for pol in policies], axis=1)
    acc_oracle = float(correctness.any(axis=1).mean())

    logger.info(f'[V9][seed={seed}] base={base} agg={agg} jac={use_jaccard} -> '
                f'Router={100*acc_router:.2f} (best_sel_loss={best_sel_loss:.4f} @ep{best_epoch})')
    return dict(seed=seed, acc_single=acc_single, acc_router=acc_router, acc_oracle=acc_oracle,
                ids=ref_test_ids, chosen=chosen, policies=policies, correct=correctness,
                labels=labels, jac=jac_test_np,
                preds={p: cached[p]['test']['preds'] for p in policies},
                conf={p: cached[p]['test']['confidence'] for p in policies},
                ent={p: cached[p]['test']['entropy'] for p in policies},
                data=data)


# =============================================================================
# SWEEP: 2 bases x 3 aggregators x {no-jaccard, jaccard} x seeds = 12 configs
# =============================================================================
def run_sweep(args, seeds, policies=('gfn', 'asgcn')):
    bases = ['gcn', 'sage']
    aggs = ['mean', 'max', 'sum']
    jac_opts = [False, True]

    results = {}
    for base in bases:
        for agg in aggs:
            for use_jac in jac_opts:
                rows = [train_mixture_router(args, seed=s, policies=policies, base=base, agg=agg,
                                             use_jaccard=use_jac) for s in seeds]
                acc = np.array([r['acc_router'] for r in rows], dtype=float)
                results[(base, agg, use_jac)] = dict(
                    mean=acc.mean(), std=acc.std(),
                    rows=[{k: v for k, v in r.items() if k != 'data'} for r in rows])

    any_rows = results[('gcn', 'mean', False)]['rows']
    single_means = {pol: float(np.mean([r['acc_single'][pol] for r in any_rows])) for pol in policies}
    oracle_mean = float(np.mean([r['acc_oracle'] for r in any_rows]))
    best_single = max(single_means.values())

    print(f'\n{"="*78}\n[V9 SWEEP] 2 bases x 3 aggregators x {{no-jaccard, jaccard}} x seeds={seeds}\n{"="*78}')
    for pol in policies:
        print(f'{pol.upper()} (fixed baseline): {100*single_means[pol]:.2f}%')
    print(f'Oracle (fixed baseline): {100*oracle_mean:.2f}%\n')

    hdr = f'{"Base":<8}{"Agg":<8}{"Jaccard":<10}{"Router Acc":>14}{"Gain vs best-single":>22}'
    print(hdr); print('-' * len(hdr))
    for base in bases:
        for agg in aggs:
            for use_jac in jac_opts:
                r = results[(base, agg, use_jac)]
                print(f'{base:<8}{agg:<8}{str(use_jac):<10}{100*r["mean"]:>10.2f}±{100*r["std"]:<4.2f}'
                      f'{100*(r["mean"]-best_single):>+18.2f} pts')

    best_key = max(results, key=lambda k: results[k]['mean'])
    print(f'\n[V9 SWEEP] Best config: base={best_key[0]} agg={best_key[1]} jaccard={best_key[2]} '
          f'-> {100*results[best_key]["mean"]:.2f}% ± {100*results[best_key]["std"]:.2f}%')
    return results, best_key


# =============================================================================
# Best-config rerun + what-is-the-router-learning report
# =============================================================================
def run_best_router_report(args, seeds, policies, best_key):
    base, agg, use_jac = best_key
    thr = args.homophily_threshold
    ia, ig = policies.index('asgcn'), policies.index('gfn')
    print(f'\n{"="*78}\n[ROUTER ANALYSIS] rerun best config base={base} agg={agg} jaccard={use_jac}\n{"="*78}')
    recs, grp_list, cgrp_list, labs = [], [], [], []
    for s in seeds:
        d = train_mixture_router(args, seed=s, policies=policies, base=base, agg=agg, use_jaccard=use_jac)
        props = get_node_props(d['data'], build_adjacency(d['data']))
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

    all_rec = _cat(recs)
    all_groups = {k: np.concatenate([g[k] for g in grp_list]) for k in grp_list[0]}
    all_cgroups = {k: np.concatenate([g[k] for g in cgrp_list]) for k in cgrp_list[0]}
    all_labels = np.concatenate(labs)
    report_groups(f'ROUTER node assignment (base={base}, agg={agg}, jaccard={use_jac}), pooled over seeds {seeds}',
                  all_groups, all_rec, thr)
    report_classes('ROUTER per-class assignment / accuracy (% of class)', all_labels, all_cgroups)
    print('\nReading guide: compare "router->ASGCN" vs "router->GFN" property means with the oracle table. '
          'If they track (e.g. low homophily/degree -> same policy), the router learned the oracle signal; '
          'the [category]->policy rows show where it is right vs wrong on decisive nodes.')


if __name__ == '__main__':
    args = Arguments(explicit_bool=True).parse_args()
    if args.config_file is not None:
        args = Arguments(explicit_bool=True, config_files=[args.config_file]).parse_args()
    SEEDS = list(range(args.runs))
    POLICIES = ['gfn', 'asgcn']

    print(f"\n{'='*70}\n[V9] Stage 0: Oracle (seeds={SEEDS}, epochs={args.max_epochs})\n{'='*70}")
    run_oracle_experiment(args, SEEDS)
    print(f"\n{'='*70}\n[V9] Stage 1: GFN + ASGCN -> router sweep (12 configs)\n{'='*70}")
    _, best_key = run_sweep(args, SEEDS, policies=POLICIES)
    run_best_router_report(args, SEEDS, POLICIES, best_key)