"""Model registry: model classes are registered into MODEL_REGISTRY via a decorator and indexed directly by the --model command-line option."""

MODEL_REGISTRY = {}


def register_model(name):
    """Decorator: register a model class under a name in the global MODEL_REGISTRY."""
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered: {MODEL_REGISTRY[name]}")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name, data, args):
    """Construct a model instance by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](data, args)
