# AR-TDA

Code and data for **AR-TDA**, a target-conditioned dual-level denoising framework for
multi-behavior recommendation (MBR).

> **Anonymous release.** This repository accompanies an anonymous submission.
> Author names, affiliations and citation information are intentionally omitted;
> they will be added in the camera-ready version.

## 🔬 Overview

AR-TDA is a **model-agnostic plug-in** for multi-behavior backbones. It denoises the
auxiliary behaviour signals at two levels:

- **AEGD (edge level).** Each node neighbourhood gets a target-conditioned uncertainty
  statistic. The statistic is turned into a threshold multiplier, edges below the
  threshold are filtered, and the retained edge weights are renormalised. This removes
  unreliable auxiliary interactions without touching the target graph.
- **REGA (behavior level).** An entropy-gated attention aggregation over behaviours,
  blended with the ungated aggregate through a residual connection, so that noisy
  auxiliary behaviours contribute less to the fused representation.

Both modules are attached to the backbone through a shared hook, so the same code runs on
top of different backbones. This release contains the two backbones used in the paper:
**MULE** and **HGIB**.

## 🌟 Environment Setup

Tested with Python 3.10 and PyTorch 2.0.1 built for CUDA 11.8. The PyTorch Geometric
wheels must match the torch build, which is why they are pinned to a wheel index in
`requirements.txt`.

```bash
pip install -r requirements.txt

# optional sanity check
python -c "import torch, torch_scatter, torch_geometric; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch_geometric.__version__)"
```

CPU execution works but is considerably slower.

## 🌟 Datasets

The three datasets are provided in `./data`. The files follow the layout used by the
releases of [MULE](https://github.com/geonwooko/MULE) and
[HGIB](https://github.com/zhanghengyu0323/HGIB), which in turn gather Tmall and JData from
[CRGCN](https://github.com/MingshiYan/CRGCN) and Taobao from
[MBCGCN](https://github.com/SS-00-SS/MBCGCN).

| Dataset | Users  | Items  | Views       | Collects | Carts         | Buys   |
|---------|-------:|-------:|------------:|---------:|--------------:|-------:|
| Taobao  | 15,449 | 11,953 | 873,954     | -        | 195,476       | 92,180 |
| Tmall   | 41,738 | 11,953 | 1,813,498   | 221,514  | 1,996         | 255,586|
| JData   | 93,334 | 24,624 | 1,681,430   | 45,613   | 49,891        | 321,883|

```
data/<dataset>/
├── statistics.json      # n_users, n_items, behaviour lists and edge counts
├── ubg.txt              # unified user-item graph, one "user item" pair per line
├── <behaviour>.txt      # per-behaviour edge lists (view / cart / collect / buy …)
├── <behaviour>_<x>.txt  # derived behaviour sets (view_not_buy, view_buy, …)
├── train.json           # pre-split target-behaviour interactions {user: [items]}
└── test.json
```

The files are already in the format the loader expects, so **no preprocessing step is
needed** before training.

## 🚀 Getting Started

Every run goes through the single entry point `src/main.py`; all options are listed in
`src/parser.py` and can be printed with `python ./src/main.py --help`.

### MULE + AR-TDA

The configuration below follows the paper. Embedding dimension 64, batch size 1024, Adam
with a learning rate of 5e-4 and weight decay 0, at most 100 training epochs, early stopping
after 10 epochs without improvement, seed 42.

```bash
python ./src/main.py \
    --dataset taobao --model mule \
    --use_tda 1 --use_rega 1 \
    --tda_mode entropy --tda_layers 3 \
    --tda_min 0.3 --tda_max 2.0 \
    --min_keep_ratio 0.0 --eg_alpha 0.3 \
    --denoise_layers last1 --denoise_alpha auto --rega_blend 0.1 \
    --entropy_norm 1 --keep_min_edges 1 --use_attn_agg 1 \
    --experiment_protocol hgib \
    --emb_dim 64 --gnn_layers 1 \
    --lr 5e-4 --weight_decay 0 \
    --batch_size 1024 --num_epochs 100 --patience 10 \
    --topk 10 --seed 42 --device cuda:0
```

The same configuration is used for all three datasets; only `--dataset` changes. The
plain backbone is obtained by switching both plug-ins off:

```bash
python ./src/main.py \
    --dataset taobao --model mule \
    --use_tda 0 --use_rega 0 \
    --experiment_protocol hgib \
    --emb_dim 64 --gnn_layers 1 \
    --lr 5e-4 --weight_decay 0 \
    --batch_size 1024 --num_epochs 100 --patience 10 \
    --topk 10 --seed 42 --device cuda:0
```

| Option | Value | Meaning |
|---|---|---|
| `--tda_layers` | 3 | number of entropy-aware graph convolution layers $L$ |
| `--tda_mode` | `entropy` | AEGD operating mode |
| `--tda_min`, `--tda_max` | 0.3, 2.0 | dynamic threshold multiplier bounds $[m_{\min}, m_{\max}]$ |
| `--min_keep_ratio` | 0.0 | minimum edge retention ratio $\rho_{\min}$ |
| `--eg_alpha` | 0.3 | REGA attention attenuation coefficient $\alpha_g$ |
| `--denoise_layers` | `last1` | layer scheduling policy of AEGD |
| `--rega_blend` | 0.1 | residual weight of the REGA-gated aggregation |
| `--entropy_norm` | 1 | degree-normalised uncertainty statistic |
| `--keep_min_edges` | 1 | keep at least one edge per node after thresholding |
| `--use_attn_agg` | 1 | attention aggregation inside REGA |

### Note on the uncertainty statistic

Two implementations of the target-conditioned uncertainty statistic are provided:

- `--fast_entropy 1` (default, used for MULE) — computed from the local score dispersion,
  followed by a sigmoid mapping;
- `--fast_entropy 0` — computed as the Shannon entropy of the local softmax distribution,
  normalised by node degree.

The flag matters when reproducing a specific number.

### HGIB + AR-TDA

The HGIB backbone is included in this release and is selected with `--model hgib`. Its
per-backbone configuration is documented separately and is intentionally not listed here.

## 📊 Results

Single-run results with `seed=42` under the protocol described below, as reported in the
paper. AR-TDA (AEGD + REGA) is applied on top of the plain MULE backbone.

| Dataset | MULE HR@10 | MULE NDCG@10 | MULE + AR-TDA HR@10 | MULE + AR-TDA NDCG@10 | Relative gain |
|---|---:|---:|---:|---:|---:|
| Taobao | 0.1506 | 0.0879 | 0.1723 | 0.0962 | +14.41% / +9.44% |
| Tmall  | 0.1417 | 0.0731 | 0.1700 | 0.0912 | +19.97% / +24.76% |
| JData  | 0.5210 | 0.3306 | 0.5745 | 0.3778 | +10.27% / +14.28% |

Because every entry is a single run under one fixed seed, these numbers are controlled
performance observations and are not reported as significance or random-stability evidence.

## 🔁 Reproducing the paper's experiments

All experiments are launched through `src/main.py`; only the flags change.

| Experiment | How |
|---|---|
| Main results | The commands above with `--dataset {taobao,tmall,jdata}`. |
| Component analysis | Vary `--use_tda` / `--use_rega` / `--use_attn_agg` / `--entropy_norm` / `--keep_min_edges`; the combination `--use_tda 0 --use_rega 0 --use_attn_agg 0` is the plain backbone. |
| Hyper-parameter sensitivity | Vary one of `--tda_layers`, `--eg_alpha`, `--min_keep_ratio`, `--tda_min`, `--tda_max` while the others stay at the values in the table above. |
| Noise robustness | Add `--noise_ratio {0.2,0.4,0.6} --noise_behaviors view --noise_seed 42`; noise edges are added to the auxiliary graph only, and the test graph is kept clean. |
| Backbone comparison | Switch `--model mule` / `--model hgib`; the backbone-specific options are documented with each backbone's configuration. |

Each run writes its log to stdout (or to `--log_file`) and prints a `RESULT HR@10=… NDCG@10=…`
line that can be parsed directly.

## ❤️ Repository layout

```
AR-TDA/
├── README.md
├── requirements.txt
├── data/                       # three datasets (see above)
└── src/
    ├── main.py                 # training / evaluation entry point
    ├── parser.py               # all command-line options and their defaults
    ├── data.py                 # dataset loading, graph construction, noise injection
    ├── utils.py                # path/device helpers, seeding, argument validation
    └── model/
        ├── __init__.py         # backbone registry (--model resolves here)
        ├── registry.py         # @register_model decorator and build_model()
        ├── base.py             # shared backbone interface and the plug-in hook
        ├── mule.py             # MULE backbone
        ├── hgib.py             # HGIB backbone
        ├── graph_conv.py       # AEGD: uncertainty statistic, thresholding, propagation
        ├── rega.py             # REGA: entropy-gated attention aggregation
        ├── metrics.py          # HR@K / NDCG@K
        └── trainer.py          # training loop, evaluation, checkpoint selection
```

## ✅ Evaluation protocol

- **Data split.** The provided `train.json` / `test.json` files are used as they are; no new
  random split is created.
- **Ranking.** Full-item ranking over the whole item set.
- **Masking.** Only training interactions are masked at evaluation time.
- **Checkpoint selection.** The best test NDCG@10 checkpoint.
- **Metrics.** HR@10 and NDCG@10 on the target behaviour.

These are the settings behind `--experiment_protocol hgib`. The alternative
`--experiment_protocol unified` derives a validation split from the training file and selects
the model by validation HR; it is kept in the code for reference.

## ✅ Citation

The paper is under anonymous review. Citation information will be added once it is available.
