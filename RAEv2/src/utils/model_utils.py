"""Model instantiation utilities."""

import importlib

import torch

from configs import ModelConfig


def get_obj_from_str(string: str, reload: bool = False):
    """Import and return a class/function from a dotted string path."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: ModelConfig) -> object:
    """Instantiate a model from ModelConfig.

    Args:
        config: ModelConfig with target, params, and optional ckpt path

    Returns:
        Instantiated model, optionally with loaded checkpoint
    """
    if not config.target:
        raise KeyError("Expected 'target' to instantiate.")

    model = get_obj_from_str(config.target)(**config.params)

    if getattr(config, "ckpt", None) is not None:
        state_dict = torch.load(config.ckpt, map_location="cpu", weights_only=False)
        # prefer EMA weights, but fall back to the raw 'model' weights when EMA is
        # absent or None (e.g. older checkpoints saved before EMA was persisted).
        if state_dict.get("ema") is not None:
            state_dict = state_dict["ema"]
        elif state_dict.get("model") is not None:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded {config.target} from {config.ckpt}")

    return model
