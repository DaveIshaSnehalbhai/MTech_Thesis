# From PyGAS, PyTorch Geometric Auto-Scale: https://github.com/rusty1s/pyg_autoscale/tree/master
from typing import Tuple

import os
import sys
import ssl
import time
import random
import socket
import shutil
import zipfile
import json
import csv
import urllib.error
import urllib.request

import scipy.io
import torch
import numpy as np
from torch.utils.data import random_split
import torch_geometric.transforms as T
import torch_geometric.utils as pygutils
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import (PPI, Amazon, Coauthor, Flickr,
                                      GNNBenchmarkDataset, Planetoid, Reddit2,
                                      WikiCS, Yelp, DBLP, WikipediaNetwork)
import torch.nn.functional as F
from .utils import gen_masks, index2mask
from .linkx.dataset import load_snap_patents_mat


# =============================================================================
# Retrying download_url -- fixes transient HTTP failures AND gives an honest,
# actionable message when the failure is really "no network access" (e.g.
# Kaggle/Colab with internet disabled), which looks completely different from
# a flaky CDN and was previously mislabeled as one.
#
# NOTE on URLs: graphmining.ai's domain registration has EXPIRED (confirmed
# via Namecheap's "domain has expired" landing page, not just DNS flakiness
# inside a sandbox) -- so it is now permanently unreachable from anywhere,
# not only from network-restricted notebooks. PyG's WikipediaNetwork with
# geom_gcn_preprocess=False has no fallback host for this. get_wikipedia_
# network_regression below auto-falls-back to an independent SNAP-hosted
# mirror of the same underlying MUSAE data when this host is unreachable.
# See RAHAS_WIKIPEDIA_NETWORK_MIRROR below for a manual-file escape hatch too.
# =============================================================================
def _is_dns_failure(exc: Exception) -> bool:
    """True if exc is (or wraps) a DNS resolution failure -- i.e. there is no
    route to the host at all, as opposed to the host responding with an
    HTTP error. socket.gaierror surfaces as urllib.error.URLError(reason=gaierror)."""
    reason = getattr(exc, 'reason', None)
    return isinstance(exc, socket.gaierror) or isinstance(reason, socket.gaierror)


def _retrying_download_url(url, folder, log=True, filename=None):
    """
    Drop-in replacement for torch_geometric.data.download.download_url,
    with retry + exponential backoff for transient HTTP failures, and a
    clear diagnosis (rather than a generic retry-then-fail) when the
    problem is actually "this environment has no network access to this
    host" -- which retrying does nothing to fix.
    """
    import os.path as osp
    if filename is None:
        filename = url.rpartition('/')[2].split('?')[0]
    path = osp.join(folder, filename)

    if osp.exists(path) and osp.getsize(path) > 0:
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    # Escape hatch: if the user has already fetched the file some other way,
    # point RAHAS_WIKIPEDIA_NETWORK_MIRROR at that local directory and we'll
    # copy it in instead of hitting the network.
    mirror_dir = os.environ.get('RAHAS_WIKIPEDIA_NETWORK_MIRROR')
    if mirror_dir:
        candidate = osp.join(mirror_dir, filename)
        if osp.exists(candidate) and osp.getsize(candidate) > 0:
            os.makedirs(folder, exist_ok=True)
            shutil.copyfile(candidate, path)
            if log:
                print(f'[download] Found {filename} in RAHAS_WIKIPEDIA_NETWORK_MIRROR '
                      f'({mirror_dir}) -- copied instead of downloading.', file=sys.stderr)
            return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)
    os.makedirs(folder, exist_ok=True)

    max_retries = 6
    backoff_base = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            context = ssl._create_unverified_context()
            resp = urllib.request.urlopen(url, timeout=30, context=context)
            content = resp.read()
            with open(path, 'wb') as f:
                f.write(content)
            return path
        except (urllib.error.HTTPError, urllib.error.URLError,
                 TimeoutError, ConnectionError) as e:
            code = getattr(e, 'code', None)
            dns_failure = _is_dns_failure(e)

            if dns_failure:
                # Retrying a DNS failure is almost always pointless -- there is
                # no route to the host, full stop. Fail fast with a diagnosis
                # instead of burning ~70s on backoff first.
                raise RuntimeError(
                    f"Could not resolve host for {url} -- this is a DNS/network-"
                    f"access failure. If this is graphmining.ai: that domain's "
                    f"registration has expired, so it is permanently unreachable, "
                    f"not just blocked in this sandbox. If you're on Kaggle and "
                    f"this is a different host: Notebook Settings -> Internet -> "
                    f"On (then restart the session). If internet is already on "
                    f"and this specific host is blocked or gone, download the "
                    f"file manually and set the RAHAS_WIKIPEDIA_NETWORK_MIRROR "
                    f"env var to the directory containing it, e.g.:\n"
                    f"  os.environ['RAHAS_WIKIPEDIA_NETWORK_MIRROR'] = '/kaggle/input/your-dataset'\n"
                    f"expected filename: {filename}, expected final path: {path}. "
                    f"Original error: {e}"
                ) from e

            transient = (code is None) or (code in (429, 500, 502, 503, 504))
            if not transient or attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download {url} after {attempt} attempt(s). "
                    f"Last error: {e}. If this persists, the remote host may be "
                    f"rate-limiting or temporarily down -- wait a few minutes "
                    f"and retry, or download the file manually to {path} (or "
                    f"set RAHAS_WIKIPEDIA_NETWORK_MIRROR to a directory containing it)."
                ) from e
            sleep_s = backoff_base ** attempt + random.uniform(0, 1.0)
            print(f"[download] {url} failed ({e}) -- retrying in {sleep_s:.1f}s "
                  f"(attempt {attempt}/{max_retries})", file=sys.stderr)
            time.sleep(sleep_s)


def _install_retrying_downloader():
    """
    Monkeypatch every torch_geometric dataset module in this file's
    download path to use _retrying_download_url instead of the stock
    single-shot version. Must run before any dataset class (Flickr,
    WikipediaNetwork, Planetoid, ...) is instantiated. Wrapped defensively
    per-module since attribute names/module layout can shift across PyG
    versions -- a failed patch on one module should never block the
    others or crash import of this file.
    """
    import importlib
    targets = [
        'torch_geometric.data.download',
        'torch_geometric.datasets.wikipedia_network',
        'torch_geometric.datasets.flickr',
        'torch_geometric.datasets.planetoid',
    ]
    for mod_name in targets:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, 'download_url'):
                mod.download_url = _retrying_download_url
        except Exception as e:
            print(f"[data.py] WARNING: could not patch retrying downloader "
                  f"into {mod_name}: {e}")


_install_retrying_downloader()


# =============================================================================
# Deterministic feature-dimension projection.
#
# The raw MUSAE/SNAP feature vocab (bag-of-informative-nouns) is much wider
# and NOT guaranteed to match the ~2.3k-dim vocab PyG's now-dead
# graphmining.ai preprocessing used -- e.g. the SNAP mirror produces ~13k
# dims for chameleon. Feeding that straight into ASGCN's pairwise sampler
# (which materializes a [num_pairs, feat_dim] tensor) causes CUDA OOM.
#
# Rather than trying to reverse-engineer the exact original vocab (which we
# can no longer verify since graphmining.ai is gone), we project every
# variant down to a FIXED width with a seeded random projection. This keeps
# GPU memory bounded and reproducible regardless of which upstream mirror
# or vocab size produced x, at the cost of not exactly matching whatever
# graphmining.ai historically served.
# =============================================================================
def _project_features(x: torch.Tensor, out_dim: int, seed: int = 0) -> torch.Tensor:
    """
    Deterministic (seeded) random projection down to `out_dim` columns.
    No-op if x already has <= out_dim features. Uses a fixed Gaussian
    random matrix scaled by 1/sqrt(in_dim) (Johnson-Lindenstrauss-style),
    so results are reproducible across runs/mirrors for the same seed.
    """
    in_dim = x.size(1)
    if in_dim <= out_dim:
        return x
    g = torch.Generator().manual_seed(int(seed))
    proj = torch.randn(in_dim, out_dim, generator=g) / (in_dim ** 0.5)
    print(f"[feature-projection] Projecting features from {in_dim} -> {out_dim} dims "
          f"(seed={seed}) to bound downstream memory use.")
    return x @ proj


# =============================================================================
# SNAP fallback mirror for chameleon/squirrel/crocodile regression data.
# PyG's WikipediaNetwork(geom_gcn_preprocess=False) only ever tries
# graphmining.ai and has no fallback host. graphmining.ai's domain
# registration has EXPIRED (confirmed via registrar), so it is permanently
# dead, not just DNS-flaky in a sandbox. snap.stanford.edu hosts the same
# underlying MUSAE data (Rozemberczki, Allen, Sarkar -- "Multi-scale
# Attributed Node Embedding") on a completely separate domain/CDN, as
# edges.csv / features.json / target.csv instead of a packed .npz.
# =============================================================================
def _download_snap_wikipedia_zip(root: str) -> str:
    extract_dir = os.path.join(root, 'SNAPWikipediaRaw')
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        return extract_dir
    zip_path = _retrying_download_url(
        'https://snap.stanford.edu/data/wikipedia.zip', root,
        filename='wikipedia_snap.zip')
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    return extract_dir

def _find_snap_files(extract_dir: str, name: str):
    """
    Locate the edges/features/target files for the requested MUSAE
    Wikipedia dataset.

    IMPORTANT:
    Match against the FILE'S BASENAME, not the full path.
    This prevents a root directory such as:
        .../data/squirrel/...
    from causing every file underneath it to match "squirrel".
    """
    name = name.lower().strip()

    all_files = []
    for dirpath, _, filenames in os.walk(extract_dir):
        for fn in filenames:
            all_files.append(os.path.join(dirpath, fn))

    print(f"[DEBUG] extract_dir={extract_dir}")
    print(f"[DEBUG] looking for name={name!r}, found {len(all_files)} files:")
    for f in all_files:
        print("   ", f)

    def pick(suffix):
        # Expected filename pattern:
        #   musae_<dataset>_<suffix>
        #
        # Example:
        #   musae_squirrel_edges.csv
        #   musae_chameleon_features.json
        #   musae_crocodile_target.csv

        expected_filename = f"musae_{name}_{suffix}"

        matches = []
        for f in all_files:
            filename = os.path.basename(f).lower()

            if filename == expected_filename:
                matches.append(f)

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise RuntimeError(
                f"[SNAP mirror] Multiple matches found for "
                f"'{expected_filename}': {matches}"
            )

        return None

    edges_f = pick('edges.csv')
    features_f = pick('features.json')
    target_f = pick('target.csv')

    print(
        f"[DEBUG] picked edges={edges_f} "
        f"features={features_f} "
        f"target={target_f}"
    )

    if not (edges_f and features_f and target_f):
        raise FileNotFoundError(
            f"[SNAP mirror] Could not find exact "
            f"edges/features/target files for '{name}'. "
            f"Expected:\n"
            f"  musae_{name}_edges.csv\n"
            f"  musae_{name}_features.json\n"
            f"  musae_{name}_target.csv"
        )

    return edges_f, features_f, target_f


def get_wikipedia_network_regression_snap(root: str, name: str, seed: int = None,
                                           split_id: int = 0,
                                           max_features: int = 128) -> Tuple[Data, int, int]:
    """
    chameleon/squirrel/crocodile node-regression data loaded directly from
    the SNAP mirror, bypassing PyG's WikipediaNetwork (graphmining.ai-only,
    no fallback host, and now permanently dead). Produces the same fields
    as get_wikipedia_network_regression: x, edge_index, standardized y
    (using train-split-only mean/std), train/val/test masks, y_mean/y_std.

    `max_features`: raw SNAP features are projected down to this width
    (see _project_features) to keep downstream memory use (e.g. ASGCN's
    pairwise sampler) bounded and independent of the mirror's raw vocab
    size. Pass None to disable projection and keep the raw width.
    """
    extract_dir = _download_snap_wikipedia_zip(root)
    edges_f, features_f, target_f = _find_snap_files(extract_dir, name)

    with open(features_f, 'r') as f:
        feat_dict = json.load(f)
    num_nodes = len(feat_dict)
    max_feat_id = max((max(v) for v in feat_dict.values() if v), default=-1)
    x = torch.zeros((num_nodes, max_feat_id + 1), dtype=torch.float)
    for node_id_str, feat_idxs in feat_dict.items():
        for fi in feat_idxs:
            x[int(node_id_str), fi] = 1.0

    with open(edges_f, 'r') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        c1 = next((c for c in cols if c.lower() in ('id1', 'node1', 'source')), cols[0])
        c2 = next((c for c in cols if c.lower() in ('id2', 'node2', 'target')), cols[1])
        edge_list = [(int(row[c1]), int(row[c2])) for row in reader]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    y = torch.zeros(num_nodes, dtype=torch.float)
    with open(target_f, 'r') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        id_col = next((c for c in cols if c.lower() in ('id', 'node_id', 'new_id')), cols[0])
        val_col = next((c for c in cols if c != id_col), cols[-1])
        for row in reader:
            y[int(row[id_col])] = float(row[val_col])

    data = Data(x=x, edge_index=edge_index, y=y)

    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        n_bad = (torch.isnan(data.x) | torch.isinf(data.x)).any(dim=1).sum().item()
        print(f"[SNAP-WikipediaNetwork-reg/{name}] WARNING: {n_bad} nodes had NaN/Inf "
              f"features -- zeroing them.")
        data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)

    seed_for_split = seed if seed is not None else split_id

    # Project BEFORE normalizing, so normalization operates on the final
    # feature space actually consumed by the model.
    if max_features is not None:
        data.x = _project_features(data.x, max_features, seed=seed_for_split)
    data.x = F.normalize(data.x, p=1)

    y = data.y.float()
    if torch.isnan(y).any() or torch.isinf(y).any():
        n_bad = (torch.isnan(y) | torch.isinf(y)).sum().item()
        print(f"[SNAP-WikipediaNetwork-reg/{name}] WARNING: {n_bad} NaN/Inf values in "
              f"regression target -- zeroing them.")
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask, val_mask, test_mask = _make_random_masks(data.num_nodes, seed=seed_for_split)

    y_mean, y_std = y[train_mask].mean(), y[train_mask].std().clamp_min(1e-8)
    data.y = (y - y_mean) / y_std
    data.y_mean, data.y_std = y_mean, y_std
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    data.edge_index = pygutils.to_undirected(data.edge_index, num_nodes=data.num_nodes)

    print(f"[SNAP-WikipediaNetwork-reg/{name}] Loaded: n={data.num_nodes} "
          f"e={data.edge_index.size(1)} split_seed={seed_for_split} "
          f"train={int(train_mask.sum())} val={int(val_mask.sum())} "
          f"test={int(test_mask.sum())} features={data.num_features} "
          f"target=standardized-continuous "
          f"(train-split mean={float(y_mean):.3f} std={float(y_std):.3f})")

    return data, data.num_features, 1  # out_dim=1 for regression


def get_blogcat(root: str, name: str, split_id: int=0) -> Tuple[Data, int, int]:
    dataset = torch.load(f'{root}/blogcatalog_0.6/split_{split_id}.pt')
    graph = scipy.io.loadmat(f'{root}/blogcatalog_0.6/blogcatalog.mat')
    edges = graph['network'].nonzero()
    edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)
    train_mask = torch.zeros(10312, dtype=torch.bool)
    test_mask = torch.zeros(10312, dtype=torch.bool)
    val_mask = torch.zeros(10312, dtype=torch.bool)
    train_mask[dataset['train_mask']] = True
    test_mask[dataset['test_mask']] = True
    val_mask[dataset['val_mask']] = True

    data = Data(y=torch.tensor(graph['group'].todense()), edge_index=edge_index,
                train_mask=train_mask, test_mask=test_mask, val_mask=val_mask,
                num_classes=39, num_nodes=10312)

    return data, data.num_features, data.num_classes


def get_dblp(root: str, name: str, seed: int = None) -> Tuple[Data, int, int]:
    features = []
    with open(f'{root}/features.txt', 'r') as file:
        content = file.read()
        values = content.strip().split('\n')
        for value in values:
            feature = value.split(',')
            number = [float(i) for i in feature]
            features.append(number)
    edges = []
    with open(f'{root}/dblp.edgelist', 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            source_node = int(parts[0])
            target_node = int(parts[1])
            edges.append((source_node, target_node))
            edges.append((source_node, target_node))

    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    labels = []
    with open(f'{root}/labels.txt', 'r') as file:
        content = file.read()
        values = content.strip().split('\n')
        for value in values:
            label = value.split(',')
            number = [1. if float(x)==1 else 0. for x in label]
            labels.append(number)
    # FIX: seed defaults to None, and torch.manual_seed(None) raises TypeError.
    # Fall back to a fixed seed so this is reproducible even when the caller
    # doesn't pass one explicitly (matches the fallback pattern already used
    # in get_wikipedia_network_regression's split seeding below).
    torch.manual_seed(seed if seed is not None else 0)
    num_nodes = 28702
    train_ratio = 0.8
    test_ratio = 0.1
    num_train = int(train_ratio * num_nodes)
    num_test = int(test_ratio * num_nodes)
    num_val = num_nodes - num_train - num_test

    train_idx, test_idx, val_idx = random_split(list(range(num_nodes)), [num_train, num_test, num_val])
    train_mask = torch.zeros(28702, dtype=torch.bool)
    test_mask = torch.zeros(28702, dtype=torch.bool)
    val_mask = torch.zeros(28702, dtype=torch.bool)
    train_mask[train_idx.indices] = True
    test_mask[test_idx.indices] = True
    val_mask[val_idx.indices] = True

    data = Data(edge_index=edge_index, x=torch.tensor(features),y=torch.tensor(labels), train_mask=train_mask,
                test_mask=test_mask, val_mask=val_mask, num_features=300)
    data.node_stores[0].num_classes = 4
    return data, data.num_features, data.num_classes


def get_synth(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = torch.load(f'{root}/Hyperspheres_10_10_0_0.6/split_0.pt')
    edge_index = torch.load(f'{root}/Hyperspheres_10_10_0/edge_index.pt')
    labels = torch.tensor(np.genfromtxt(f'{root}/Hyperspheres_10_10_0/labels.csv', skip_header=1,
                       dtype=np.dtype(float), delimiter=','))
    features = torch.tensor(np.genfromtxt(f'{root}/Hyperspheres_10_10_0/features.csv', skip_header=1,
                       dtype=np.dtype('float32'), delimiter=','))
    data = Data(x=features, y=labels, edge_index=edge_index, train_mask=dataset['train_mask'],
                test_mask=dataset['test_mask'], val_mask=dataset['val_mask'],
                num_classes=20)
    return data, data.num_features, data.num_classes


def get_synth_high(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = torch.load(f'{root}/Hyperspheres_10_10_0_0.6/split_0.pt')
    edge_index = torch.load(f'{root}/Hyperspheres_10_10_0/edge_index_high.pt')
    labels = torch.tensor(np.genfromtxt(f'{root}/Hyperspheres_10_10_0/labels.csv', skip_header=1,
                       dtype=np.dtype(float), delimiter=','))
    features = torch.tensor(np.genfromtxt(f'{root}/Hyperspheres_10_10_0/features.csv', skip_header=1,
                       dtype=np.dtype('float32'), delimiter=','))
    data = Data(x=features, y=labels, edge_index=edge_index, train_mask=dataset['train_mask'],
                test_mask=dataset['test_mask'], val_mask=dataset['val_mask'],
                num_classes=20)
    return data, data.num_features, data.num_classes


def get_planetoid(root: str, name: str) -> Tuple[Data, int, int]:
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = Planetoid(f'{root}/Planetoid', name, split='full') #, transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes


def _make_random_masks(num_nodes: int, seed: int = None, train_ratio: float = 0.6,
                        val_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deterministic (seeded) 60/20/20 random split, used for dataset variants
    that don't ship a predefined train/val/test mask (e.g. the raw,
    non geom_gcn_preprocess WikipediaNetwork regression targets).
    """
    g = torch.Generator().manual_seed(int(seed) if seed is not None else 0)
    perm = torch.randperm(num_nodes, generator=g)
    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return (index2mask(train_idx, num_nodes),
            index2mask(val_idx, num_nodes),
            index2mask(test_idx, num_nodes))


def get_wikipedia_network_regression(root: str, name: str, seed: int = None,
                                      split_id: int = 0,
                                      max_features: int = 128) -> Tuple[Data, int, int]:
    """
    Regression variant of chameleon/squirrel: the raw (geom_gcn_preprocess=
    False) WikipediaNetwork target, i.e. the continuous (log) monthly page
    traffic value per node -- the original node-regression benchmark, as
    opposed to the 5-bin classification target used by
    get_wikipedia_network(). This raw variant ships no predefined
    train/val/test split, so we generate a deterministic 60/20/20 split
    seeded by `seed` (falling back to `split_id` if seed is None), so the
    same split is reproduced across GFN/ASGCN/router calls for a given run.

    The target is standardized (zero mean / unit variance) using TRAIN-SPLIT
    statistics only (computed after the split, not before) so val/test targets
    don't leak into the standardization. data.y_mean / data.y_std are kept in
    case de-standardizing is ever needed.

    Tries PyG's graphmining.ai-hosted .npz first. graphmining.ai's domain
    registration has EXPIRED (confirmed via registrar), so it is now
    permanently unreachable -- not a transient or sandbox-only DNS issue.
    On that DNS failure this falls back automatically to an independent
    SNAP-hosted mirror of the same underlying MUSAE data
    (get_wikipedia_network_regression_snap).

    `max_features`: raw features (from either source) are projected down to
    this width (see _project_features) so downstream memory use (e.g.
    ASGCN's pairwise sampler) is bounded and consistent regardless of which
    source served the data. Pass None to disable projection.
    """
    canon = {'chameleon': 'chameleon', 'squirrel': 'squirrel'}[name.lower()]

    try:
        dataset = WikipediaNetwork(f'{root}/WikipediaNetwork', canon,
                                    geom_gcn_preprocess=False)
        data = dataset[0]
    except RuntimeError as e:
        if 'DNS/network-access failure' not in str(e):
            raise
        print(f"[WikipediaNetwork-reg/{canon}] graphmining.ai unreachable (domain "
              f"registration expired) -- falling back to SNAP mirror "
              f"(snap.stanford.edu).")
        return get_wikipedia_network_regression_snap(
            root, name, seed=seed, split_id=split_id, max_features=max_features)

    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        n_bad = (torch.isnan(data.x) | torch.isinf(data.x)).any(dim=1).sum().item()
        print(f"[WikipediaNetwork-reg/{canon}] WARNING: {n_bad} nodes had NaN/Inf "
              f"features -- zeroing them (known upstream PyG caching issue).")
        data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)

    seed_for_split = seed if seed is not None else split_id

    # Project BEFORE normalizing, so normalization operates on the final
    # feature space actually consumed by the model.
    if max_features is not None:
        data.x = _project_features(data.x, max_features, seed=seed_for_split)
    data.x = F.normalize(data.x, p=1)

    y = data.y.float()
    if torch.isnan(y).any() or torch.isinf(y).any():
        n_bad = (torch.isnan(y) | torch.isinf(y)).sum().item()
        print(f"[WikipediaNetwork-reg/{canon}] WARNING: {n_bad} NaN/Inf values in "
              f"regression target -- zeroing them.")
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Split FIRST, then standardize using train-only statistics, so
    # val/test targets can't influence the scale the model is trained/judged on.
    train_mask, val_mask, test_mask = _make_random_masks(data.num_nodes, seed=seed_for_split)

    y_mean, y_std = y[train_mask].mean(), y[train_mask].std().clamp_min(1e-8)
    data.y = (y - y_mean) / y_std
    data.y_mean, data.y_std = y_mean, y_std
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    data.edge_index = pygutils.to_undirected(data.edge_index, num_nodes=data.num_nodes)

    print(f"[WikipediaNetwork-reg/{canon}] Loaded: n={data.num_nodes} "
          f"e={data.edge_index.size(1)} split_seed={seed_for_split} "
          f"train={int(train_mask.sum())} val={int(val_mask.sum())} "
          f"test={int(test_mask.sum())} features={data.num_features} "
          f"target=standardized-continuous "
          f"(train-split mean={float(y_mean):.3f} std={float(y_std):.3f})")

    return data, data.num_features, 1  # out_dim=1 for regression


def get_wikipedia_network(root: str, name: str, split_id: int = 0, seed: int = None,
                           task: str = 'classification') -> Tuple[Data, int, int]:
    if task == 'regression':
        return get_wikipedia_network_regression(root, name, seed=seed, split_id=split_id)

    canon = {'chameleon': 'chameleon', 'squirrel': 'squirrel'}[name.lower()]
    dataset = WikipediaNetwork(f'{root}/WikipediaNetwork', canon,
                                geom_gcn_preprocess=True)
    data = dataset[0]

    # ---- Gotcha 4: sanitize features defensively before anything else touches them.
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        n_bad = (torch.isnan(data.x) | torch.isinf(data.x)).any(dim=1).sum().item()
        print(f"[WikipediaNetwork/{canon}] WARNING: {n_bad} nodes had NaN/Inf "
              f"features -- zeroing them (known upstream PyG caching issue for "
              f"this dataset). If this number is large, delete "
              f"{root}/WikipediaNetwork and re-download instead of trusting this.")
        data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)

    data.x = F.normalize(data.x, p=1)

    # ---- Gotcha 1: geom_gcn_preprocess masks are [N, 10] -- pick ONE split.
    if data.train_mask.dim() == 2:
        n_splits = data.train_mask.size(1)
        if not (0 <= split_id < n_splits):
            raise ValueError(
                f"[WikipediaNetwork/{canon}] --split_id={split_id} out of range -- "
                f"geom_gcn_preprocess provides {n_splits} predefined splits "
                f"(valid range: 0..{n_splits - 1})."
            )
        train_mask = data.train_mask[:, split_id].clone()
        val_mask   = data.val_mask[:, split_id].clone()
        test_mask  = data.test_mask[:, split_id].clone()
    else:
        # Older/non-geom_gcn cache: already 1D -- split_id is a no-op.
        if split_id != 0:
            print(f"[WikipediaNetwork/{canon}] WARNING: cached masks are already "
                  f"1D (single split available) -- --split_id={split_id} is being "
                  f"ignored.")
        train_mask = data.train_mask.clone()
        val_mask   = data.val_mask.clone()
        test_mask  = data.test_mask.clone()

    # ---- Gotcha 2: fail loudly HERE, not deep inside a DataLoader later.
    for split_name, m in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        assert m.dim() == 1 and m.numel() == data.num_nodes, (
            f"[WikipediaNetwork/{canon}] {split_name}_mask has unexpected shape "
            f"{tuple(m.shape)} for num_nodes={data.num_nodes} -- stale or "
            f"corrupted cache. Delete {root}/WikipediaNetwork and re-download."
        )
        assert m.sum().item() > 0, (
            f"[WikipediaNetwork/{canon}] {split_name}_mask is entirely False for "
            f"split_id={split_id} -- stale or corrupted cache. Delete "
            f"{root}/WikipediaNetwork and re-download."
        )
    if ((train_mask & val_mask).any() or (train_mask & test_mask).any()
            or (val_mask & test_mask).any()):
        print(f"[WikipediaNetwork/{canon}] WARNING: train/val/test masks overlap "
              f"for split_id={split_id} -- unexpected for the standard Geom-GCN "
              f"splits; double-check the cache before trusting results.")

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # ---- Gotcha 5: compute num_classes directly, don't trust dataset.num_classes.
    num_classes = int(data.y.max().item()) + 1
    if num_classes != 5:
        print(f"[WikipediaNetwork/{canon}] WARNING: expected 5 classes (the "
              f"standard binned target) but found {num_classes} -- check the "
              f"dataset cache / PyG version.")

    # ---- Gotcha 3: known duplicate-feature-row leakage, logged once, not "fixed".
    n_unique = torch.unique(data.x, dim=0).size(0)
    if n_unique < data.num_nodes:
        dup_frac = 100.0 * (1 - n_unique / data.num_nodes)
        print(f"[WikipediaNetwork/{canon}] NOTE: {dup_frac:.1f}% of node feature "
              f"rows are exact duplicates of another node's row (documented "
              f"leakage issue in raw Chameleon/Squirrel -- see Platonov et al., "
              f"'A critical look at evaluation of GNNs on heterophilous graphs'). "
              f"Absolute accuracy on this dataset will look inflated relative to "
              f"cleaner benchmarks; ASGCN-vs-GFN-vs-RA-HAS comparisons remain "
              f"valid since every expert sees the same data.")

    # ---- Gotcha 6: force undirected, matching this codebase's assumption
    # elsewhere that adjacency is symmetric.
    data.edge_index = pygutils.to_undirected(data.edge_index, num_nodes=data.num_nodes)

    print(f"[WikipediaNetwork/{canon}] Loaded: n={data.num_nodes} "
          f"e={data.edge_index.size(1)} split_id={split_id} "
          f"train={int(train_mask.sum())} val={int(val_mask.sum())} "
          f"test={int(test_mask.sum())} classes={num_classes}")

    return data, data.num_features, num_classes


def get_wikics(root: str) -> Tuple[Data, int, int]:
    dataset = WikiCS(f'{root}/WIKICS', transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes


def get_coauthor(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Coauthor(f'{root}/Coauthor', name, transform=T.ToSparseTensor(remove_edge_index = False))
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_amazon(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Amazon(f'{root}/Amazon', name, transform=T.ToSparseTensor(remove_edge_index = False))
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_arxiv(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB')#,
                                     #pre_transform=T.ToSparseTensor())
    data = dataset[0]
    #data.adj_t = data.adj_t.to_symmetric()
    # Instead of the commented line above use the following line to convert the edge_index to undirected
    data.edge_index = pygutils.to_undirected(data.edge_index)
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_products(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB')#,
                                     #pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_yelp(root: str) -> Tuple[Data, int, int]:
    dataset = Yelp(f'{root}/YELP')#, pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str) -> Tuple[Data, int, int]:
    dataset = Flickr(f'{root}/Flickr')#, pre_transform=T.ToSparseTensor())
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2')#, pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_ppi(root: str, split: str = 'train') -> Tuple[Data, int, int]:
    dataset = PPI(f'{root}/PPI', split=split)#, pre_transform=T.ToSparseTensor())
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    data[f'{split}_mask'] = torch.ones(data.num_nodes, dtype=torch.bool)
    return data, dataset.num_features, dataset.num_classes


def get_sbm(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train',
                                  pre_transform=T.ToSparseTensor())
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    return data, dataset.num_features, dataset.num_classes


def get_proteins(root: str):
    dataset = PygNodePropPredDataset('ogbn-proteins', root)
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)

    # This is a multi-label binary classification dataset, so we need
    # float targets for BCEWithLogitsLoss
    data.y = data.y.float()

    return data, dataset.num_features, data.y.shape[1]


def get_linkx_dataset(root: str, name: str, seed: int = None):
    if name.lower() == 'snap-patents':
        dataset = load_snap_patents_mat(root)
        split_idx = dataset.get_idx_split(seed=seed)
        num_nodes = dataset.graph['num_nodes']
        train_mask = index2mask(split_idx['train'], num_nodes)
        valid_mask = index2mask(split_idx['valid'], num_nodes)
        test_mask = index2mask(split_idx['test'], num_nodes)

        edge_index = dataset.graph['edge_index']
        edge_index = pygutils.to_undirected(edge_index, num_nodes=num_nodes)

        data = Data(x=dataset.graph['node_feat'],
                    edge_index=edge_index,
                    y=dataset.label,
                    train_mask=train_mask,
                    val_mask=valid_mask,
                    test_mask=test_mask)
        num_classes = len(data.y.unique())
    else:
        raise ValueError(f'Unknown dataset name: {name}')

    return data, data.num_features, num_classes

def get_data(root: str,
             name: str,
             seed: int = None,
             split_id: int = 0,
             task: str = 'classification',
             ) -> Tuple[Data, int, int]:
    """
    `task`: 'classification' (default, unchanged behavior for every dataset)
    or 'regression'. Regression is currently only supported for
    name in ['chameleon', 'squirrel'] -- it loads the raw (continuous,
    standardized) WikipediaNetwork target instead of the 5-bin
    classification target. Passing task='regression' for any other
    dataset raises NotImplementedError.
    """
    if task not in ('classification', 'regression'):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    if task == 'regression' and name.lower() not in ['chameleon', 'squirrel']:
        raise NotImplementedError(
            f"task='regression' is currently only supported for "
            f"name in ['chameleon', 'squirrel'], got {name!r}."
        )

    if name.lower() in ['blogcat']:
        return get_blogcat(root, name, split_id)
    elif name.lower() == 'dblp':
        return get_dblp(root, name, seed)
    elif name.lower() == 'synth':
        return get_synth(root, name)
    elif name.lower() == 'synth_high':
        return get_synth_high(root, name)
    elif name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(root, name)
    elif name.lower() in ['chameleon', 'squirrel']:
        return get_wikipedia_network(root, name, split_id, seed=seed, task=task)
    elif name.lower() in ['coauthorcs', 'coauthorphysics']:
        return get_coauthor(root, name[8:])
    elif name.lower() in ['amazoncomputers', 'amazonphoto']:
        return get_amazon(root, name[6:])
    elif name.lower() == 'wikics':
        return get_wikics(root)
    elif name.lower() in ['cluster', 'pattern']:
        return get_sbm(root, name)
    elif name.lower() == 'reddit':
        return get_reddit(root)
    elif name.lower() == 'ppi':
        return get_ppi(root)
    elif name.lower() == 'flickr':
        return get_flickr(root)
    elif name.lower() == 'yelp':
        return get_yelp(root)
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root)
    elif name.lower() in ['ogbn-products', 'products']:
        return get_products(root)
    elif name.lower() == 'ogbn-proteins':
        return get_proteins(root)
    elif name.lower() == 'snap-patents':
        return get_linkx_dataset(root, 'snap-patents', seed)
    else:
        raise NotImplementedError