"""Model package: the backbones, the AR-TDA plug-in modules and the trainer.

Importing a backbone module runs its @register_model decorator, so every backbone
listed below becomes available through --model.
"""
from . import mule       # noqa: F401  -- @register_model('mule')
from . import hgib       # noqa: F401  -- @register_model('hgib')

from .registry import MODEL_REGISTRY, build_model
from .base import BaseRecModel
from .trainer import Trainer

__all__ = ['MODEL_REGISTRY', 'build_model', 'BaseRecModel', 'Trainer']
