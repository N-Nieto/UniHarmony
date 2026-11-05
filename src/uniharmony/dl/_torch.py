"""Import PyTorch, the optional dependency of the deep learning methods."""

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "The deep learning methods in uniharmony.dl require PyTorch. "
        "Install it with `pip install uniharmony[dl]` (or follow https://pytorch.org/get-started/locally/ "
        "to install a build for your GPU)."
    ) from e


__all__ = ["nn", "torch"]
