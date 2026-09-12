"""Command-line options. All defaults live in DEFAULTS so that the same values can be
reused programmatically (e.g. by hyper-parameter sweeps).
"""
import argparse

DEFAULTS = dict(
    # ---------- Data ----------
    dataset='taobao',
    data_dir='./data',
    batch_size=1024,
    eval_batch_size=1024,
    eval_item_chunk_size=2048,
    num_workers=8,
    noise_ratio=0.0,              # RQ5: fraction of random noise edges; 0 = no noise injected
    noise_behaviors='view',       # RQ5: behaviours for noise injection, comma-separated
    noise_seed=42,                # RQ5: random seed for sampling noise edges
    validation_ratio=0.1,         # used by the unified protocol; the HGIB protocol directly uses the pre-built train/test splits
    split_seed=42,
    experiment_protocol='hgib',  # hgib = official HGIB data/model-selection/evaluation protocol; unified = validation-set protocol

    # ---------- Training ----------
    lr=5e-4,
    weight_decay=0.0,
    num_epochs=100,
    patience=10,
    seed=42,
    device='cuda:0',
    topk=10,
    resume=None,
    save_checkpoint=True,
    p0_diagnostics=False,          # record gradients, parameter updates and REGA attention diagnostics on the first training batch

    # ---------- Shared model options ----------
    emb_dim=64,
    gnn_layers=1,
    tda_layers=3,

    # ---------- Model selection ----------
    model='hgib',          # the default baseline and experimental protocol follow the official HGIB

    # ---------- Plugin switches ----------
    use_tda=False,          # whether to enable topological entropy convolution (AEGD)
    use_rega=False,         # whether to enable REGA entropy gating (activate gating while retaining attention aggregation)
    tda_mode='entropy',     # soft / hard / entropy / soft_entropy
    tda_min=0.3,
    tda_max=2.0,
    eg_alpha=0.3,
    fast_entropy=True,
    rega_attn_dim=16,
    rega_blend=0.1,          # proportion of the REGA entropy-gated result blended into the ungated aggregation result
    beta=50.0,               # HGIB HSIC weight
    alpha=1.0,               # HGIB contrastive learning weight
    threshold=0.05,          # HGIB graph learner threshold
    sigma=1.0,               # HGIB HSIC kernel bandwidth
    # aligned with EaMRec: topological entropy layer scheduling policy
    denoise_layers='last1',         # progressive / all / none / last1 / last2 / last_half / first4
    denoise_alpha='auto',
    # ---------- Switches specific to ablation experiments ----------
    use_attn_agg=True,      # False = remove REGA entirely and aggregate behaviours by mean -> corresponds to "w/o REGA"
    entropy_norm=True,      # False = do not divide topological entropy by ln(max(d,2)) -> corresponds to "w/o entropy normalisation"
    keep_min_edges=True,    # False = remove the "keep at least one edge per node" fallback -> corresponds to "w/o minimum retention ratio"
    min_keep_ratio=0.0,     # minimum fraction of incoming edges retained per node after truncation; 0 = use only the keep_min_edges fallback

    # ---------- Output ----------
    checkpoint_dir='./experiments',
    log_file=None,          # if None, log to stdout only
)


def _add_arg(parser, key, default):
    """Register one option; booleans are passed as 0/1 and None means free-form string."""
    if isinstance(default, bool):
        parser.add_argument(f'--{key}', type=lambda x: bool(int(x)), default=default)
    elif default is None:
        parser.add_argument(f'--{key}', type=str, default=None)
    else:
        parser.add_argument(f'--{key}', type=type(default), default=default)


def parse_args():
    p = argparse.ArgumentParser(description='AR-TDA')
    for k, v in DEFAULTS.items():
        _add_arg(p, k, v)
    return p.parse_args()
