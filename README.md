# RA-HAS V6: GFlowNet vs AS-GCN Router

Script: `rahas_v6.py` (run as `main.py` in the commands below).

## Run

```bash
!python main.py --config_file=configs/gflownet/chameleon.txt
!python main.py --config_file=configs/gflownet/squirrel.txt
```

---

## What this code does (simple words)

A graph neural network (GNN) classifies a node by looking at its neighbours. Real graphs are big, so you cannot look at all neighbours. You **sample** a few. *How* you sample matters.

This code builds **two different samplers**:

- **GFN**: learns which neighbours to pick using a GFlowNet (a learned policy).
- **AS-GCN**: a known method (adaptive sampling) that picks neighbours using attention.

Each one is right on some nodes and wrong on others. They do not fail on the same nodes.

Then the code trains a third model, the **router**. For each node, the router decides: "trust GFN here" or "trust AS-GCN here."

Finally, it tests an idea:

> Nodes whose neighbours share their label (**homophilous**) should go to **AS-GCN**.
> Nodes whose neighbours have different labels (**heterophilous**) should go to **GFN**.

The code measures whether this is true, first for a perfect router (the **Oracle**), then for the real learned router.

**The three stages, in one line each:**

1. **Oracle**: what is the best possible accuracy if we always picked the right policy per node?
2. **Router sweep**: train 12 router variants, find the best one.
3. **Analysis**: does the best router send homophilous nodes to AS-GCN and heterophilous nodes to GFN?

---

## What this code does (in detail)

### Step 1: Train policy 1, GFN (`train_gfn_mainstyle`)

Three small GCNs are trained together:

- `gcn_c`: the **classifier**. Predicts node labels from the sampled subgraph.
- `gcn_gf`: the **sampling policy**. Gives each candidate neighbour a score.
- `gcn_z`: estimates **log Z**, the normalising term the GFlowNet loss needs.

For each batch of target nodes, for each hop (2 hops):

1. Find all neighbours of the current frontier.
2. Run `gcn_gf` on them (with hop indicator features) to get a logit per neighbour.
3. Sample `num_samples` neighbours per node from those scores. Save the log-probabilities.
4. Build the sampled subgraph and run `gcn_c` on it.

Two losses:

- Classifier: normal cross-entropy on the target nodes.
- Sampler: **trajectory balance loss** `(log_z + sum(log_probs) + loss_coef * CE)^2`. The classifier's CE acts as the "cost," so the sampler learns to pick neighbourhoods that give low classification loss.

Every `eval_frequency` epochs it checks val accuracy and keeps the best checkpoint. At the end it outputs, for train, val and test nodes: predictions, labels, node ids, confidence, entropy and probability of the true class.

### Step 2: Train policy 2, AS-GCN (`train_asgcn_native`)

A PyTorch port of the original AS-GCN (`huangwb/AS-GCN`), fixed at 2 layers.

- **Sampler** (`ASGCNSampler`): one weight matrix `w_s`. It computes attention between a node and its candidate neighbours and turns that into sampling probabilities `p`. Sampling is layer-wise: `asgcn_rank` nodes are drawn per layer, shared across the whole batch.
- **Network** (`ASGCNNet`): a 2-layer GCN. Edge weights come from the normalised adjacency, multiplied by learned attention, divided by the sampling probability (importance-sampling correction).
- **Variance regulariser**: an extra loss term (`asgcn_var`) that reduces the variance caused by sampling.
- **Training graph**: either the full graph or the subgraph induced by train nodes (`asgcn_train_graph`). High-degree nodes can be truncated (`asgcn_max_degree`) to keep batches small.
- **Evaluation** (`evaluate_asgcn_native`): uses the **full** neighbourhood with no sampling, same as the original paper's test.

Same outputs as GFN: predictions, confidence, entropy for train/val/test.

### Step 3: Node properties (`get_node_props`)

For every node it computes structural facts used later to explain routing:

| Property | Meaning |
|---|---|
| `deg` | degree |
| `h` | **node homophily**: fraction of neighbours with the same label |
| `cos` | mean feature cosine similarity with neighbours |
| `lab_ent` | entropy of neighbours' labels |
| `two_hop` | size of 2-hop neighbourhood |
| `clust` | clustering coefficient |
| `pagerank`, `kcore` | importance / core number |
| `nbr_var` | variance of neighbour features |
| `dist_train` | shortest distance to any train node |

A node is **homophilous** if `h >= homophily_threshold` (default 0.5), **heterophilous** if below, and **isolated** if degree is 0 (excluded from the hypothesis check).

### Step 4: Oracle experiment (`run_oracle_experiment`)

For every seed, on the test set, compare the two policies node by node:

- **both right**
- **only AS-GCN right**
- **only GFN right**
- **both wrong**

Oracle accuracy = both right + only AS-GCN + only GFN. This is the upper bound for any router. The gap between Oracle and the best single policy is the **headroom**.

It then prints:

- Accuracy of GFN, AS-GCN, Oracle (mean ± std over seeds).
- Per-group table: how many nodes in each group are homophilous / heterophilous / isolated, plus mean node properties per group.
- Per-class table: what % of each class falls in each group.
- `assignment_summary`: of all homophilous nodes, how many does the oracle send to AS-GCN vs GFN, and the same for heterophilous. It reports whether the hypothesis holds.

### Step 5: Router (`MixtureRouter`, `train_mixture_router`)

**Model:** two PyG graph-conv layers (`GCNConv` or `SAGEConv`, 512-dim) on the original graph, then a linear head that outputs **one logit**. Output > 0.5 after sigmoid means "use AS-GCN," otherwise "use GFN."

**Target (Oracle-style):** only nodes where exactly one policy is correct are used for training. Label = 1 if AS-GCN is right, 0 if GFN is right. Nodes where both are right or both are wrong carry no signal and are skipped.

**Training and selection:**

- Train: BCE loss on the disagreement nodes of `train_mask`.
- Validate: BCE loss on the disagreement nodes of `val_mask`.
- The epoch with the lowest val BCE is restored.

**Deployment:** the hard 0.5 threshold is applied to **all** test nodes (not only disagreement ones). The final prediction is the chosen policy's prediction. Reported: router accuracy, how often each policy was chosen (usage), and router accuracy on test disagreement nodes.

**Optional Jaccard feature (`compute_jaccard_overlap`):** for node `v`, take the top-k neighbours GFN would pick and the top-k AS-GCN would pick, and compute their Jaccard overlap. High overlap means the two samplers agree on the neighbourhood. This one number is concatenated to the router embedding.

### Step 6: Sweep (`run_sweep`)

Runs the router for every combination of:

- base layer: `gcn`, `sage`
- aggregation: `mean`, `max`, `sum`
- Jaccard feature: off, on

That is 2 x 3 x 2 = **12 configs**, each over all seeds. A table prints router accuracy, gain over the best single policy, and accuracy on disagreement nodes. The best config by mean accuracy is selected.

### Step 7: Best-config analysis (`run_best_router_report`)

The best config is rerun on all seeds. For the **router's hard assignment**, it reports:

- How many homophilous nodes went to AS-GCN, and how many heterophilous nodes went to GFN.
- The same restricted to **decisive nodes** (exactly one policy right). This is the fair test of whether the router copies the oracle.
- Breakdown by category, for example `[only GFN right] -> ASGCN` (a wrong routing) vs `[only GFN right] -> GFN` (a correct one).
- Pooled per-group node-property tables and per-class tables.

**How to read the result:** compare the router's decisive-node summary with the Oracle summary from Step 4. If they match, the router has learned the homophily-based rule. If not, the router is using other signals or failing to learn the rule.

---

## Execution order when you run the script

1. Stage 0: Oracle (trains or loads both policies for every seed, prints oracle stats)
2. Stage 1: 12-config router sweep (prints the sweep table and best config)
3. Stage 2: best-config rerun with the homophily/heterophily routing report
