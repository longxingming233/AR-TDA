"""Training and evaluation entry point.

Example:
    python ./src/main.py --dataset taobao --model mule --use_tda 1 --use_rega 1

Adding a backbone only requires a new module under src/model/ plus one import in
src/model/__init__.py; this file does not need to change.
"""
from pathlib import Path

import torch
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from data import load_multi_behavior_data
from model import build_model, MODEL_REGISTRY, Trainer
from parser import parse_args
from utils import set_seed, resolve_path, validate_args, validate_device


def main():
    args = parse_args()
    validate_args(args)
    args.device = validate_device(args.device)
    project_dir = Path(__file__).resolve().parent.parent
    args.data_dir = resolve_path(args.data_dir, project_dir)
    args.checkpoint_dir = resolve_path(args.checkpoint_dir, project_dir)
    if args.log_file:
        args.log_file = resolve_path(args.log_file, project_dir)
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if args.log_file and hasattr(logger, 'add'):
        logger.add(args.log_file, rotation='10 MB', encoding='utf-8')
    logger.info(f"Available models: {list(MODEL_REGISTRY.keys())}")
    logger.info(f"Args: {vars(args)}")

    set_seed(args.seed)

    # 1) Data
    data = load_multi_behavior_data(
        data_dir=args.data_dir, dataset=args.dataset,
        device=args.device, batch_size=args.batch_size,
        num_workers=args.num_workers,
        noise_ratio=args.noise_ratio,
        noise_behaviors=args.noise_behaviors,
        noise_seed=args.noise_seed,
        model=args.model,
        eval_batch_size=args.eval_batch_size,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        experiment_protocol=args.experiment_protocol,
    )
    if args.topk > data['n_items']:
        raise ValueError(f'topk={args.topk} exceeds n_items={data["n_items"]}')
    logger.info(f"Dataset {args.dataset}: n_users={data['n_users']}, "
                f"n_items={data['n_items']}, "
                f"bsg={data['bsg_types']}, tcb={data['tcb_types']}, tib={data['tib_types']}")
    if data.get('noise_ratio', 0.0) > 0:
        logger.info(f"Noise injection: ratio={data['noise_ratio']} "
                    f"behaviors={data['noise_behaviors']} added={data['noise_stats']}")

    # 2) Model (constructed by name from the registry)
    logger.info(f"Building model: {args.model}")
    model = build_model(args.model, data, args).to(args.device)
    named_params = list(model.named_parameters())
    n_params = sum(param.numel() for _, param in named_params)
    n_trainable = sum(param.numel() for _, param in named_params if param.requires_grad)
    n_aegd = sum(
        param.numel() for name, param in named_params
        if any(token in name.lower() for token in ('tda', 'entropy_gate'))
    )
    n_rega = sum(
        param.numel() for name, param in named_params
        if 'rega' in name.lower() or 'eg_alpha' in name.lower()
    )
    n_plugin = n_aegd + n_rega
    n_base = max(n_params - n_plugin, 0)
    logger.info(f"Model {args.model} | params={n_params:,} trainable={n_trainable:,}")
    logger.info(
        f"PARAMS total={n_params} trainable={n_trainable} base={n_base} "
        f"aegd={n_aegd} rega={n_rega}"
    )

    # 3) Training
    logger.info("Using trainer: Trainer")
    trainer = Trainer(model, data, args)
    best_hr, best_ndcg = trainer.train()

    logger.info(
        f"DONE | model={args.model} use_tda={args.use_tda} use_rega={args.use_rega} | "
        f"BEST HR@{args.topk}={best_hr:.4f} NDCG@{args.topk}={best_ndcg:.4f}"
    )
    logger.info(
        f"RESULT HR@{args.topk}={best_hr:.12g} NDCG@{args.topk}={best_ndcg:.12g}"
    )


if __name__ == '__main__':
    main()
