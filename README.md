# AR-TDA

Code and data for **AR-TDA**, a target-conditioned dual-level denoising framework for
multi-behavior recommendation (MBR).

> This repository accompanies a submission under anonymous review; author and citation
> information will be added later.

## 🔬 Overview

AR-TDA is a **model-agnostic plug-in** for multi-behavior backbones. **AEGD** computes a
target-conditioned uncertainty statistic for every node neighbourhood, filters unreliable edges
with a dynamic threshold and renormalises the retained weights, leaving the target behaviour
graph untouched. **REGA** aggregates behaviours with an entropy-gated attention mechanism that
is blended with the ungated aggregate through a residual connection. Two backbones are
included: **MULE** and **HGIB**.

## 🌟 Environment Setup

Python 3.10 and PyTorch 2.0.1 built for CUDA 11.8; the PyTorch Geometric wheels must match the
torch build, which is why they are pinned to a wheel index in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 🌟 Datasets

Taobao, Tmall and JData are provided in `./data`, in the layout used by the
[MULE](https://github.com/geonwooko/MULE) and
[HGIB](https://github.com/zhanghengyu0323/HGIB) releases, which in turn gather Tmall and JData
from [CRGCN](https://github.com/MingshiYan/CRGCN) and Taobao from
[MBCGCN](https://github.com/SS-00-SS/MBCGCN). No preprocessing step is needed.

| Dataset | Users  | Items  | Views       | Collects | Carts         | Buys   |
|---------|-------:|-------:|------------:|---------:|--------------:|-------:|
| Taobao  | 15,449 | 11,953 | 873,954     | -        | 195,476       | 92,180 |
| Tmall   | 41,738 | 11,953 | 1,813,498   | 221,514  | 1,996         | 255,586|
| JData   | 93,334 | 24,624 | 1,681,430   | 45,613   | 49,891        | 321,883|

## 🚀 Getting Started

#### Train MULE + AR-TDA on the `Taobao` dataset
```bash
python ./src/main.py \
    --dataset taobao --model mule \
    --use_tda 1 --use_rega 1 \
    --tda_mode entropy --tda_layers 3 \
    --tda_min 0.3 --tda_max 2.0 --eg_alpha 0.3 --min_keep_ratio 0.0 \
    --denoise_layers last1 --denoise_alpha auto --rega_blend 0.1 \
    --entropy_norm 1 --keep_min_edges 1 --use_attn_agg 1 \
    --experiment_protocol hgib \
    --emb_dim 64 --gnn_layers 1 \
    --lr 5e-4 --weight_decay 0 \
    --batch_size 1024 --num_epochs 100 --patience 10 \
    --topk 10 --seed 42 --device cuda:0
```

The same configuration is used for all three datasets; only `--dataset` changes. Setting
`--use_tda 0 --use_rega 0` gives the plain backbone, and `--model hgib` switches to the HGIB
backbone. All options are listed in `src/parser.py`.

## ❤️ Acknowledgement

This code is developed on top of the [MULE](https://github.com/geonwooko/MULE) and
[HGIB](https://github.com/zhanghengyu0323/HGIB) implementations.
