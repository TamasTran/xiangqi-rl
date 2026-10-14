"""Minimal checkpointing: weights + model config (+ optional optimizer state).

Checkpoints are plain dicts of tensors and Python primitives, so they are
loaded with ``torch.load(..., weights_only=True)``, which refuses arbitrary
pickled code. Still, only load checkpoints you created or trust.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from xiangqi_rl.nn.config import ARCHITECTURE, ModelConfig
from xiangqi_rl.nn.model import PolicyValueNet

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | os.PathLike[str],
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write ``model`` (and optionally ``optimizer`` state) to ``path``.

    ``extra`` may hold small primitives (step counters, seeds, ...). The file
    is written atomically (temp file + rename). The destination is always the
    caller's choice; nothing is saved implicitly.
    """
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "config": model.config.to_dict(),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Read and validate a checkpoint dict (tensors mapped to ``map_location``)."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"{path} is not a XiangqiRL model checkpoint")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format_version')!r} "
            f"(expected {CHECKPOINT_FORMAT_VERSION})"
        )
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError(
            f"Checkpoint architecture {payload.get('architecture')!r} does not match "
            f"{ARCHITECTURE!r}"
        )
    return payload


def load_model(
    path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
) -> PolicyValueNet:
    """Rebuild the model from the stored config and load its weights."""
    payload = load_checkpoint(path, map_location)
    model = PolicyValueNet(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(map_location)
