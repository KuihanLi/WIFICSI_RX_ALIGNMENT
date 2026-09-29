"""Minimal training helpers matching the frozen B2_MEAN optimization semantics.

The original internal pipeline performs sample-first splitting and supplies one row per
window with a physical-recording sample_id. This public helper keeps the same loss
weighting rule while leaving private dataset routing outside the release.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from b2_mean import B2Mean


@dataclass(frozen=True)
class FrozenTrainingConfig:
    batch_size: int = 128
    max_epochs: int = 18
    patience: int = 3
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    amp: bool = True


def physical_recording_window_weights(sample_ids, labels) -> np.ndarray:
    """Replicate the frozen per-window weighting rule."""
    sample_ids = np.asarray(sample_ids)
    labels = np.asarray(labels)
    if len(sample_ids) != len(labels):
        raise ValueError("sample_ids and labels must have the same length")
    unique_pairs = {(str(s), str(y)) for s, y in zip(sample_ids, labels)}
    class_sample_counts: dict[str, int] = {}
    for s, y in unique_pairs:
        class_sample_counts[y] = class_sample_counts.get(y, 0) + 1
    sample_window_counts: dict[str, int] = {}
    for s in sample_ids:
        sample_window_counts[str(s)] = sample_window_counts.get(str(s), 0) + 1
    weights = np.asarray([
        1.0 / max(1, class_sample_counts[str(y)]) / max(1, sample_window_counts[str(s)])
        for s, y in zip(sample_ids, labels)
    ], dtype=np.float32)
    if weights.size and float(weights.mean()) > 0:
        weights /= float(weights.mean())
    return weights


def weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_item = F.cross_entropy(logits, targets, reduction="none")
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-6)


def fit_epoch(model: B2Mean, loader, optimizer, device: torch.device, *, amp: bool = True) -> float:
    model.train()
    amp_enabled = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total = 0.0
    count = 0
    for x, y, w in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss = weighted_cross_entropy(model(x)["logits"], y, w)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def early_stop_update(model, score: float, state: dict, epoch: int, *, tolerance: float = 1e-8):
    """Expose the frozen 'higher validation score is better' early-stop rule."""
    if score > state.get("best_score", -1e18) + tolerance:
        state["best_score"] = float(score)
        state["best_epoch"] = int(epoch)
        state["best_state"] = copy.deepcopy(model.state_dict())
        state["bad_epochs"] = 0
    else:
        state["bad_epochs"] = int(state.get("bad_epochs", 0)) + 1
    return state
