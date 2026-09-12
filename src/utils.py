"""Fix random seeds for reproducible experiments."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from pathlib import Path

import torch


def resolve_path(path, base_dir):
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path(base_dir) / value
    return str(value.resolve())


def validate_device(value):
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f'Invalid device: {value}') from exc
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device requested but CUDA is unavailable. Use --device cpu or install a compatible CUDA environment.')
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f'Invalid CUDA device index {index}; available device count is {torch.cuda.device_count()}')
        return torch.device(f'cuda:{index}')
    if device.type != 'cpu':
        raise ValueError(f'Unsupported device type: {device.type}')
    return device


def validate_args(args):
    positive = (
        'batch_size', 'eval_batch_size', 'eval_item_chunk_size',
        'num_epochs', 'patience', 'topk', 'emb_dim',
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.num_workers < 0:
        raise ValueError('num_workers must be non-negative')
    if getattr(args, 'tda_layers', 3) <= 0:
        raise ValueError('tda_layers must be positive')
    if getattr(args, 'gnn_layers', 1) <= 0:
        raise ValueError('gnn_layers must be positive')
    if getattr(args, 'rega_attn_dim', 16) <= 0:
        raise ValueError('rega_attn_dim must be positive')
    if not isinstance(getattr(args, 'fast_entropy', True), bool):
        raise ValueError('fast_entropy must be boolean')
    denoise_layers = getattr(args, 'denoise_layers', 'last1')
    if denoise_layers not in {'progressive', 'all', 'none', 'last1', 'last2', 'last_half', 'first4'}:
        raise ValueError(f'Unsupported denoise_layers: {denoise_layers}')
    if not 0.0 <= args.noise_ratio <= 1.0:
        raise ValueError('noise_ratio must be between 0 and 1')
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError('validation_ratio must be between 0 and 1')
    if getattr(args, 'experiment_protocol', 'hgib') not in {'hgib', 'unified'}:
        raise ValueError('experiment_protocol must be hgib or unified')
    if args.tda_mode not in {'soft', 'soft_entropy', 'hard', 'entropy'}:
        raise ValueError(f'Unsupported tda_mode: {args.tda_mode}')
    denoise_alpha = getattr(args, 'denoise_alpha', 'auto')
    if isinstance(denoise_alpha, str):
        if denoise_alpha.lower() == 'auto':
            args.denoise_alpha = 'auto'
        else:
            try:
                args.denoise_alpha = float(denoise_alpha)
            except ValueError as exc:
                raise ValueError("denoise_alpha must be 'auto' or a value in [0, 1]") from exc
            if not 0.0 <= args.denoise_alpha <= 1.0:
                raise ValueError('denoise_alpha must be in [0, 1]')
    elif not 0.0 <= float(denoise_alpha) <= 1.0:
        raise ValueError('denoise_alpha must be in [0, 1]')
    if not 0.0 < args.tda_min < 1.0:
        raise ValueError('tda_min must be in (0, 1)')
    if args.tda_max <= args.tda_min:
        raise ValueError('tda_max must be greater than tda_min')
    if not 0.0 <= args.min_keep_ratio <= 1.0:
        raise ValueError('min_keep_ratio must be between 0 and 1')
    if args.model not in {'mule', 'hgib'}:
        raise ValueError(f'Unknown model: {args.model}. Available: mule, hgib')
    if args.topk > args.eval_item_chunk_size:
        raise ValueError('topk must not exceed eval_item_chunk_size')
