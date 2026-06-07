#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact/local search wokół najlepszego wyniku z random searcha.

Domyślnie:
- szuka konfiguracji w otoczeniu:
  hidden_layers='256,256,128,64', activation='relu', norm='batch',
  dropout=0.0, lr=0.001, weight_decay=0.001, batch_size=2048,
  imbalance_cap=10.0, loss_variant='focal', focal_gamma=2.0
- ładuje dane na GPU od razu po wczytaniu,
- zapisuje wszystko do katalogu: ./exact_search
- zapisuje CSV, najlepszy model .pt, best_config.json oraz wykresy PNG.

Najprostsze użycie w katalogu projektu:
    python exact_search.py

Gdy dane nie zostaną znalezione automatycznie:
    python exact_search.py --data data/prepared_dataset.npz

Dla CSV:
    python exact_search.py --data data.csv --target-cols left,right,brake,throttle

Oczekiwane klucze w NPZ, jedna z wersji:
    X_train, y_train, X_val, y_val
    albo train_X, train_y, val_X, val_y
    albo X, y  wtedy skrypt sam zrobi split train/val.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


# ============================================================
# Konfiguracja modelu i searcha
# ============================================================

@dataclass(frozen=True)
class AdvancedConfig:
    model_type: str = "binary"
    hidden_layers: str = "256,256,128,64"
    activation: str = "relu"
    norm: str = "batch"
    dropout: float = 0.0
    lr: float = 0.001
    weight_decay: float = 0.001
    batch_size: int = 2048
    imbalance_cap: float = 10.0
    loss_variant: str = "focal"
    focal_gamma: float = 2.0


BASE_CONFIG = AdvancedConfig()

RESULT_FIELDS = [
    "idx",
    "config_hash",
    *[f.name for f in fields(AdvancedConfig)],
    "epoch",
    "train_loss",
    "val_loss",
    "score",
    "macro_f1_runtime",
    "exact_acc_runtime",
    "brake_f1_runtime",
    "macro_f1_tuned",
    "exact_acc_tuned",
    "brake_f1_tuned",
    "thresholds",
    "seconds",
    "error",
]


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch_for_gpu() -> None:
    # Maksymalizuje typową przepustowość na NVIDIA RTX.
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_gpu_tensor(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    arr = np.asarray(x)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def ensure_binary_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    # Dla bezpieczeństwa: wszystko powyżej 0.5 traktujemy jako 1.
    return (y.astype(np.float32) > 0.5).astype(np.float32)


def config_hash(config: AdvancedConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def safe_float(value, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or str(value).strip() == "":
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def format_float(x: float) -> str:
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


# ============================================================
# Ładowanie danych
# ============================================================

def _first_existing_key(data, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {k.lower(): k for k in data.keys()}
    for c in candidates:
        if c in data:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    seed: int,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if n != len(y):
        raise ValueError(f"X i y mają różną liczbę próbek: {len(X)} vs {len(y)}")
    if n < 5:
        raise ValueError("Za mało próbek, żeby zrobić sensowny split train/val.")

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    val_size = max(1, int(round(n * val_fraction)))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def load_npz_dataset(
    path: Path,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)

    x_train_key = _first_existing_key(data, ["X_train", "x_train", "train_X", "train_x", "features_train"])
    y_train_key = _first_existing_key(data, ["y_train", "Y_train", "train_y", "train_Y", "labels_train"])
    x_val_key = _first_existing_key(data, ["X_val", "x_val", "X_valid", "x_valid", "val_X", "valid_X", "features_val"])
    y_val_key = _first_existing_key(data, ["y_val", "Y_val", "y_valid", "Y_valid", "val_y", "valid_y", "labels_val"])

    if x_train_key and y_train_key and x_val_key and y_val_key:
        X_train = np.asarray(data[x_train_key], dtype=np.float32)
        y_train = ensure_binary_y(data[y_train_key])
        X_val = np.asarray(data[x_val_key], dtype=np.float32)
        y_val = ensure_binary_y(data[y_val_key])
        return X_train, y_train, X_val, y_val

    x_key = _first_existing_key(data, ["X", "x", "features", "inputs", "observations"])
    y_key = _first_existing_key(data, ["y", "Y", "labels", "targets", "actions"])

    if x_key and y_key:
        X = np.asarray(data[x_key], dtype=np.float32)
        y = ensure_binary_y(data[y_key])
        return split_train_val(X, y, val_fraction=val_fraction, seed=seed, shuffle=True)

    available = ", ".join(data.keys())
    raise ValueError(
        f"Nie rozpoznaję struktury NPZ: {path}. Dostępne klucze: {available}"
    )


def load_npy_dir_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = {
        "X_train": ["X_train.npy", "x_train.npy", "train_X.npy", "features_train.npy"],
        "y_train": ["y_train.npy", "Y_train.npy", "train_y.npy", "labels_train.npy"],
        "X_val": ["X_val.npy", "x_val.npy", "X_valid.npy", "val_X.npy", "features_val.npy"],
        "y_val": ["y_val.npy", "Y_val.npy", "y_valid.npy", "val_y.npy", "labels_val.npy"],
    }

    resolved = {}
    for key, names in candidates.items():
        for name in names:
            p = path / name
            if p.exists():
                resolved[key] = p
                break

    missing = [k for k in candidates if k not in resolved]
    if missing:
        raise ValueError(f"Brakuje plików NPY w {path}: {missing}")

    X_train = np.asarray(np.load(resolved["X_train"], allow_pickle=True), dtype=np.float32)
    y_train = ensure_binary_y(np.load(resolved["y_train"], allow_pickle=True))
    X_val = np.asarray(np.load(resolved["X_val"], allow_pickle=True), dtype=np.float32)
    y_val = ensure_binary_y(np.load(resolved["y_val"], allow_pickle=True))
    return X_train, y_train, X_val, y_val


def infer_target_cols(columns: Sequence[str]) -> Optional[List[str]]:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}

    candidates = [
        ["left", "right", "brake", "throttle"],
        ["steer_left", "steer_right", "brake", "throttle"],
        ["turn_left", "turn_right", "brake", "throttle"],
        ["a", "d", "s", "w"],
        ["left", "right", "brake", "gas"],
        ["left", "right", "brake", "accelerate"],
        ["steer_left", "steer_right", "brake", "accelerate"],
    ]

    for cand in candidates:
        if all(c.lower() in lower for c in cand):
            return [lower[c.lower()] for c in cand]

    # Fallback: kolumny zawierające typowe nazwy akcji.
    keywords = ["left", "right", "brake", "throttle"]
    found = []
    for kw in keywords:
        matches = [c for c in cols if kw in c.lower()]
        if len(matches) == 1:
            found.append(matches[0])
    if len(found) == 4:
        return found

    return None


def load_csv_dataset(
    path: Path,
    target_cols: Optional[List[str]],
    feature_cols: Optional[List[str]],
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Do ładowania CSV potrzebny jest pandas: pip install pandas") from exc

    df = pd.read_csv(path)

    if target_cols is None:
        target_cols = infer_target_cols(df.columns)
        if target_cols is None:
            raise ValueError(
                "Nie udało się automatycznie wykryć kolumn targetów. "
                "Podaj np. --target-cols left,right,brake,throttle"
            )

    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Brakuje kolumn targetów w CSV: {missing_targets}")

    if feature_cols is None:
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        feature_cols = [c for c in numeric_cols if c not in target_cols]

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Brakuje kolumn feature'ów w CSV: {missing_features}")

    used_cols = feature_cols + target_cols
    df = df[used_cols].dropna(axis=0).reset_index(drop=True)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = ensure_binary_y(df[target_cols].to_numpy(dtype=np.float32))

    return split_train_val(X, y, val_fraction=val_fraction, seed=seed, shuffle=True)


def auto_find_dataset(start_dir: Path) -> Optional[Path]:
    explicit_candidates = [
        "prepared_dataset.npz",
        "dataset_prepared.npz",
        "processed_dataset.npz",
        "training_dataset.npz",
        "train_dataset.npz",
        "dataset.npz",
        "data/prepared_dataset.npz",
        "data/dataset_prepared.npz",
        "data/processed_dataset.npz",
        "data/training_dataset.npz",
        "data/dataset.npz",
        "datasets/prepared_dataset.npz",
        "original_results/prepared_dataset.npz",
        "original_results/dataset.npz",
    ]

    for rel in explicit_candidates:
        p = start_dir / rel
        if p.exists():
            return p

    # Jeżeli nazwa jest inna, próbujemy pierwsze sensowne NPZ w najczęstszych katalogach.
    search_roots = [start_dir, start_dir / "data", start_dir / "datasets", start_dir / "original_results"]
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        npz_files = sorted(root.glob("*.npz"))
        if npz_files:
            return npz_files[0]

    return None


def load_dataset_from_args(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    if args.data:
        path = Path(args.data).expanduser().resolve()
    else:
        found = auto_find_dataset(Path.cwd())
        if found is None:
            raise FileNotFoundError(
                "Nie znalazłem automatycznie danych. Uruchom np.:\n"
                "  python exact_search.py --data data/prepared_dataset.npz\n"
                "albo dla CSV:\n"
                "  python exact_search.py --data data.csv --target-cols left,right,brake,throttle"
            )
        path = found.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Nie istnieje ścieżka danych: {path}")

    if path.is_dir():
        # Najpierw NPZ w katalogu, potem osobne NPY.
        npz_files = sorted(path.glob("*.npz"))
        if npz_files:
            X_train, y_train, X_val, y_val = load_npz_dataset(npz_files[0], args.val_fraction, args.seed)
            return X_train, y_train, X_val, y_val, npz_files[0]
        X_train, y_train, X_val, y_val = load_npy_dir_dataset(path)
        return X_train, y_train, X_val, y_val, path

    suffix = path.suffix.lower()
    if suffix == ".npz":
        X_train, y_train, X_val, y_val = load_npz_dataset(path, args.val_fraction, args.seed)
        return X_train, y_train, X_val, y_val, path

    if suffix == ".csv":
        X_train, y_train, X_val, y_val = load_csv_dataset(
            path=path,
            target_cols=parse_csv_list(args.target_cols),
            feature_cols=parse_csv_list(args.feature_cols),
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        return X_train, y_train, X_val, y_val, path

    if suffix == ".npy":
        raise ValueError(
            "Dla pojedynczego .npy nie wiadomo, gdzie są X/y. "
            "Podaj katalog z X_train.npy, y_train.npy, X_val.npy, y_val.npy albo użyj NPZ."
        )

    raise ValueError(f"Nieobsługiwany typ danych: {path}")


def maybe_standardize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    out_dir: Path,
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return X_train, X_val

    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)

    np.savez(out_dir / "scaler_stats.npz", mean=mean, std=std)
    return X_train, X_val


# ============================================================
# Model
# ============================================================

class MLPBinaryClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: str,
        activation: str = "relu",
        norm: str = "batch",
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()]
        dims = [input_dim] + hidden

        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        elif activation == "silu":
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_features = dims[i]
            out_features = dims[i + 1]

            layers.append(nn.Linear(in_features, out_features))

            if norm == "batch":
                layers.append(nn.BatchNorm1d(out_features))
            elif norm == "layer":
                layers.append(nn.LayerNorm(out_features))
            elif norm in ("none", "", None):
                pass
            else:
                raise ValueError(f"Unsupported norm: {norm}")

            layers.append(act_layer())

            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))

        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Loss
# ============================================================

def compute_pos_weight(y_train: torch.Tensor, imbalance_cap: float) -> torch.Tensor:
    pos = y_train.sum(dim=0)
    neg = y_train.shape[0] - pos
    pos_weight = neg / pos.clamp_min(1.0)
    pos_weight = pos_weight.clamp(min=1.0, max=float(imbalance_cap))
    return pos_weight


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )

    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_factor = (1.0 - pt).clamp_min(1e-6).pow(float(gamma))
    return (focal_factor * bce).mean()


def make_loss_fn(config: AdvancedConfig, y_train: torch.Tensor):
    pos_weight = compute_pos_weight(y_train, config.imbalance_cap)

    if config.loss_variant == "focal":
        def loss_fn(logits, targets):
            return focal_bce_with_logits(
                logits=logits,
                targets=targets,
                pos_weight=pos_weight,
                gamma=config.focal_gamma,
            )
        return loss_fn

    if config.loss_variant == "bce":
        def loss_fn(logits, targets):
            return F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight,
            )
        return loss_fn

    raise ValueError(f"Unsupported loss_variant: {config.loss_variant}")


# ============================================================
# Metrics + threshold tuning
# ============================================================

@torch.no_grad()
def f1_per_class_from_preds(preds: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    preds = preds.bool()
    y_true = y_true.bool()

    tp = (preds & y_true).sum(dim=0).float()
    fp = (preds & ~y_true).sum(dim=0).float()
    fn = (~preds & y_true).sum(dim=0).float()

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return f1


@torch.no_grad()
def metrics_from_thresholds(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    thresholds: torch.Tensor,
    brake_index: int = 2,
) -> Dict[str, float]:
    preds = probs >= thresholds.view(1, -1)
    f1s = f1_per_class_from_preds(preds, y_true)
    macro_f1 = f1s.mean().item()
    exact_acc = (preds == y_true.bool()).all(dim=1).float().mean().item()
    brake_f1 = f1s[brake_index].item() if brake_index < f1s.numel() else float("nan")
    return {
        "macro_f1": macro_f1,
        "exact_acc": exact_acc,
        "brake_f1": brake_f1,
    }


@torch.no_grad()
def score_threshold_metrics(metrics: Dict[str, float], val_loss: float = 0.0) -> float:
    return (
        metrics["macro_f1"]
        + 0.20 * metrics["brake_f1"]
        + 0.05 * metrics["exact_acc"]
        - 0.02 * val_loss
    )


@torch.no_grad()
def tune_thresholds_per_class(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    coarse_min: float = 0.25,
    coarse_max: float = 0.80,
    coarse_step: float = 0.025,
    fine_radius: float = 0.040,
    fine_step: float = 0.005,
) -> torch.Tensor:
    """
    Osobny threshold dla każdej klasy, najpierw pod F1 klasy.
    Potem funkcja tune_thresholds_coordinate_refine może poprawić wynik łączony.
    """
    device = probs.device
    num_classes = probs.shape[1]
    best_thresholds = []

    coarse_grid = torch.arange(coarse_min, coarse_max + 1e-9, coarse_step, device=device)

    for c in range(num_classes):
        p = probs[:, c].view(-1, 1)
        y = y_true[:, c].bool().view(-1, 1)

        preds = p >= coarse_grid.view(1, -1)
        tp = (preds & y).sum(dim=0).float()
        fp = (preds & ~y).sum(dim=0).float()
        fn = (~preds & y).sum(dim=0).float()

        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        best_coarse = float(coarse_grid[torch.argmax(f1)].item())

        fine_min = max(0.05, best_coarse - fine_radius)
        fine_max = min(0.95, best_coarse + fine_radius)
        fine_grid = torch.arange(fine_min, fine_max + 1e-9, fine_step, device=device)

        preds = p >= fine_grid.view(1, -1)
        tp = (preds & y).sum(dim=0).float()
        fp = (preds & ~y).sum(dim=0).float()
        fn = (~preds & y).sum(dim=0).float()

        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        best_fine = float(fine_grid[torch.argmax(f1)].item())
        best_thresholds.append(best_fine)

    return torch.tensor(best_thresholds, device=device)


@torch.no_grad()
def tune_thresholds_coordinate_refine(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    initial_thresholds: torch.Tensor,
    brake_index: int = 2,
    rounds: int = 3,
    radius: float = 0.050,
    step: float = 0.005,
) -> torch.Tensor:
    """
    Drobne poprawianie thresholdów pod łączny score:
    macro_f1 + 0.20*brake_f1 + 0.05*exact_acc.
    Nie robi pełnego 4D brute force, więc nie wybucha pamięciowo.
    """
    device = probs.device
    thresholds = initial_thresholds.clone()
    num_classes = thresholds.numel()

    best_metrics = metrics_from_thresholds(probs, y_true, thresholds, brake_index=brake_index)
    best_score = score_threshold_metrics(best_metrics)

    for _ in range(rounds):
        improved = False
        for c in range(num_classes):
            current = float(thresholds[c].item())
            grid = torch.arange(
                max(0.05, current - radius),
                min(0.95, current + radius) + 1e-9,
                step,
                device=device,
            )
            local_best_t = current
            local_best_score = best_score

            for t in grid:
                candidate = thresholds.clone()
                candidate[c] = t
                m = metrics_from_thresholds(probs, y_true, candidate, brake_index=brake_index)
                s = score_threshold_metrics(m)
                if s > local_best_score:
                    local_best_score = s
                    local_best_t = float(t.item())

            if local_best_score > best_score + 1e-10:
                thresholds[c] = local_best_t
                best_score = local_best_score
                improved = True

        if not improved:
            break

    return thresholds


@torch.no_grad()
def predict_logits_in_batches(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    outs = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size]
        outs.append(model(xb))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    loss_fn,
    eval_batch_size: int,
    brake_index: int = 2,
) -> Dict[str, object]:
    model.eval()

    logits = predict_logits_in_batches(model, X_val, eval_batch_size)
    val_loss = float(loss_fn(logits, y_val).item())
    probs = torch.sigmoid(logits)

    runtime_thresholds = torch.full(
        size=(y_val.shape[1],),
        fill_value=0.5,
        device=y_val.device,
    )

    tuned_thresholds = tune_thresholds_per_class(probs, y_val)
    tuned_thresholds = tune_thresholds_coordinate_refine(
        probs,
        y_val,
        initial_thresholds=tuned_thresholds,
        brake_index=brake_index,
    )

    runtime = metrics_from_thresholds(probs, y_val, runtime_thresholds, brake_index=brake_index)
    tuned = metrics_from_thresholds(probs, y_val, tuned_thresholds, brake_index=brake_index)

    return {
        "val_loss": val_loss,
        "macro_f1_runtime": runtime["macro_f1"],
        "exact_acc_runtime": runtime["exact_acc"],
        "brake_f1_runtime": runtime["brake_f1"],
        "macro_f1_tuned": tuned["macro_f1"],
        "exact_acc_tuned": tuned["exact_acc"],
        "brake_f1_tuned": tuned["brake_f1"],
        "thresholds": [round(float(x), 4) for x in tuned_thresholds.detach().cpu()],
    }


# ============================================================
# Trening jednej konfiguracji
# ============================================================

def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def combined_config_score(metrics: Dict[str, object]) -> float:
    return float(
        metrics["macro_f1_tuned"]
        + 0.20 * metrics["brake_f1_tuned"]
        + 0.05 * metrics["exact_acc_tuned"]
        - 0.02 * metrics["val_loss"]
    )


def train_one_config(
    config: AdvancedConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int,
    patience: int,
    brake_index: int,
    use_amp: bool,
    compile_model: bool,
    grad_clip_norm: Optional[float],
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor], List[Dict[str, float]]]:
    device = X_train.device
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = MLPBinaryClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=config.hidden_layers,
        activation=config.activation,
        norm=config.norm,
        dropout=config.dropout,
    ).to(device)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile pominięty: {exc}")

    loss_fn = make_loss_fn(config, y_train)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=float(config.lr) * 0.05,
    )

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = make_grad_scaler(enabled=amp_enabled)

    n = X_train.shape[0]
    batch_size = int(config.batch_size)
    eval_batch_size = max(batch_size * 2, 4096)

    best_score = -1e18
    best_metrics: Optional[Dict[str, object]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_history: List[Dict[str, float]] = []
    history: List[Dict[str, float]] = []
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)

        epoch_loss_sum = 0.0
        seen = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() < 2 and config.norm == "batch":
                # BatchNorm nie lubi batcha o rozmiarze 1 w trybie train.
                continue

            xb = X_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, amp_enabled):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            bs = int(xb.shape[0])
            epoch_loss_sum += float(loss.detach().item()) * bs
            seen += bs

        train_loss = epoch_loss_sum / max(seen, 1)

        metrics = evaluate_model(
            model=model,
            X_val=X_val,
            y_val=y_val,
            loss_fn=loss_fn,
            eval_batch_size=eval_batch_size,
            brake_index=brake_index,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        metrics["score"] = combined_config_score(metrics)

        scheduler.step(float(metrics["macro_f1_tuned"]))

        history_row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(metrics["val_loss"]),
            "score": float(metrics["score"]),
            "macro_f1_runtime": float(metrics["macro_f1_runtime"]),
            "macro_f1_tuned": float(metrics["macro_f1_tuned"]),
            "exact_acc_runtime": float(metrics["exact_acc_runtime"]),
            "exact_acc_tuned": float(metrics["exact_acc_tuned"]),
            "brake_f1_tuned": float(metrics["brake_f1_tuned"]),
            "lr_current": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(history_row)

        if float(metrics["score"]) > best_score:
            best_score = float(metrics["score"])
            best_metrics = dict(metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_history = list(history)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_metrics is None or best_state is None:
        raise RuntimeError("Nie udało się wytrenować żadnej epoki dla tej konfiguracji.")

    return best_metrics, best_state, best_history


# ============================================================
# Exact/local grid wokół najlepszego punktu
# ============================================================

def generate_exact_configs(
    search_level: str,
    include_layernorm: bool = False,
    include_nearby_architectures: bool = False,
) -> List[AdvancedConfig]:
    """
    Exhaustive grid w lokalnym otoczeniu najlepszego random-search triala.

    fast:      324 konfiguracje
    balanced: 1620 konfiguracji
    large:    9000+ konfiguracji, tylko gdy naprawdę masz czas
    """
    search_level = search_level.lower().strip()

    if search_level == "fast":
        lr_options = [0.00085, 0.00100, 0.00115]
        wd_options = [0.00070, 0.00100, 0.00140]
        dropout_options = [0.00, 0.02]
        batch_size_options = [1536, 2048]
        cap_options = [8.0, 10.0, 12.0]
        gamma_options = [1.75, 2.00, 2.25]
    elif search_level == "balanced":
        lr_options = [0.00070, 0.00085, 0.00100, 0.00115, 0.00130]
        wd_options = [0.00050, 0.00070, 0.00100, 0.00140]
        dropout_options = [0.00, 0.01, 0.02]
        batch_size_options = [1536, 2048, 3072]
        cap_options = [8.0, 10.0, 12.0]
        gamma_options = [1.75, 2.00, 2.25]
    elif search_level == "large":
        lr_options = [0.00055, 0.00070, 0.00085, 0.00100, 0.00115, 0.00130]
        wd_options = [0.00030, 0.00050, 0.00070, 0.00100, 0.00140, 0.00200]
        dropout_options = [0.00, 0.01, 0.02, 0.04]
        batch_size_options = [1024, 1536, 2048, 3072, 4096]
        cap_options = [6.0, 8.0, 10.0, 12.0, 15.0]
        gamma_options = [1.50, 1.75, 2.00, 2.25, 2.50]
    else:
        raise ValueError("--search-level musi być jednym z: fast, balanced, large")

    norm_options = ["batch"]
    if include_layernorm:
        norm_options += ["layer", "none"]

    hidden_options = ["256,256,128,64"]
    if include_nearby_architectures:
        hidden_options += [
            "256,256,128",
            "256,256,128,128",
            "384,256,128,64",
            "512,256,128,64",
            "256,256,256,128,64",
        ]

    configs: List[AdvancedConfig] = []
    seen = set()

    # Bazowy punkt idzie jako pierwszy.
    configs.append(BASE_CONFIG)
    seen.add(config_hash(BASE_CONFIG))

    product_iter = itertools.product(
        hidden_options,
        norm_options,
        dropout_options,
        lr_options,
        wd_options,
        batch_size_options,
        cap_options,
        gamma_options,
    )

    for hidden, norm, dropout, lr, wd, bs, cap, gamma in product_iter:
        cfg = AdvancedConfig(
            model_type="binary",
            hidden_layers=hidden,
            activation="relu",
            norm=norm,
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(wd),
            batch_size=int(bs),
            imbalance_cap=float(cap),
            loss_variant="focal",
            focal_gamma=float(gamma),
        )
        h = config_hash(cfg)
        if h not in seen:
            configs.append(cfg)
            seen.add(h)

    # Sortowanie po odległości od najlepszego punktu, żeby przy ewentualnym --max-configs
    # najpierw szły konfiguracje najbliższe znanemu dobremu wynikowi.
    def distance_from_base(cfg: AdvancedConfig) -> float:
        d = 0.0
        d += abs(math.log(cfg.lr / BASE_CONFIG.lr)) * 1.5
        d += abs(math.log(cfg.weight_decay / BASE_CONFIG.weight_decay)) * 1.2
        d += abs(cfg.dropout - BASE_CONFIG.dropout) * 5.0
        d += abs(math.log(cfg.batch_size / BASE_CONFIG.batch_size)) * 0.7
        d += abs(math.log(cfg.imbalance_cap / BASE_CONFIG.imbalance_cap)) * 0.9
        d += abs(cfg.focal_gamma - BASE_CONFIG.focal_gamma) * 0.7
        if cfg.norm != BASE_CONFIG.norm:
            d += 2.0
        if cfg.hidden_layers != BASE_CONFIG.hidden_layers:
            d += 2.5
        return d

    # Base zostaje pierwszy, reszta lokalnie posortowana.
    base = configs[0]
    rest = sorted(configs[1:], key=distance_from_base)
    return [base] + rest


# ============================================================
# CSV / JSON / wykresy
# ============================================================

def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    normalized = {k: row.get(k, "") for k in RESULT_FIELDS}

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def save_history_csv(path: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def load_completed_hashes(results_csv: Path, retry_errors: bool = False) -> set:
    if not results_csv.exists():
        return set()
    completed = set()
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row.get("config_hash", "")
            if not h:
                continue
            if retry_errors and row.get("error"):
                continue
            completed.add(h)
    return completed


def read_successful_results(results_csv: Path) -> List[Dict[str, object]]:
    if not results_csv.exists():
        return []
    rows: List[Dict[str, object]] = []
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("error"):
                continue
            converted: Dict[str, object] = dict(row)
            for key in [
                "idx",
                "dropout",
                "lr",
                "weight_decay",
                "batch_size",
                "imbalance_cap",
                "focal_gamma",
                "epoch",
                "train_loss",
                "val_loss",
                "score",
                "macro_f1_runtime",
                "exact_acc_runtime",
                "brake_f1_runtime",
                "macro_f1_tuned",
                "exact_acc_tuned",
                "brake_f1_tuned",
                "seconds",
            ]:
                converted[key] = safe_float(row.get(key))
            rows.append(converted)
    return rows


def best_row_from_csv(results_csv: Path) -> Optional[Dict[str, object]]:
    rows = read_successful_results(results_csv)
    if not rows:
        return None
    return max(rows, key=lambda r: safe_float(r.get("score")))


def _plot_line(rows: List[Dict[str, object]], y_key: str, title: str, ylabel: str, path: Path, best_so_far: bool = False) -> None:
    if plt is None or not rows:
        return
    xs = np.array([safe_float(r.get("idx")) for r in rows], dtype=float)
    ys = np.array([safe_float(r.get(y_key)) for r in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() == 0:
        return
    xs, ys = xs[mask], ys[mask]

    fig = plt.figure(figsize=(11, 6))
    plt.plot(xs, ys, marker="o", linewidth=1, markersize=3, label=ylabel)
    if best_so_far:
        order = np.argsort(xs)
        xs_ord = xs[order]
        ys_ord = ys[order]
        plt.plot(xs_ord, np.maximum.accumulate(ys_ord), linewidth=2, label="best so far")
        plt.legend()
    plt.title(title)
    plt.xlabel("trial")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_scatter(rows: List[Dict[str, object]], x_key: str, y_key: str, title: str, xlabel: str, path: Path, log_x: bool = False) -> None:
    if plt is None or not rows:
        return
    xs = np.array([safe_float(r.get(x_key)) for r in rows], dtype=float)
    ys = np.array([safe_float(r.get(y_key)) for r in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() == 0:
        return
    xs, ys = xs[mask], ys[mask]

    fig = plt.figure(figsize=(9, 6))
    plt.scatter(xs, ys, s=24, alpha=0.75)
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(y_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_top_configs(rows: List[Dict[str, object]], out_path: Path, top_k: int = 20) -> None:
    if plt is None or not rows:
        return
    top = sorted(rows, key=lambda r: safe_float(r.get("score")), reverse=True)[:top_k]
    if not top:
        return

    labels = []
    scores = []
    for r in top:
        labels.append(
            f"#{int(safe_float(r.get('idx')))} "
            f"lr={format_float(safe_float(r.get('lr')))} "
            f"wd={format_float(safe_float(r.get('weight_decay')))} "
            f"do={format_float(safe_float(r.get('dropout')))} "
            f"bs={int(safe_float(r.get('batch_size')))}"
        )
        scores.append(safe_float(r.get("score")))

    fig = plt.figure(figsize=(12, max(6, 0.42 * len(top))))
    y_pos = np.arange(len(top))[::-1]
    plt.barh(y_pos, scores[::-1])
    plt.yticks(y_pos, labels[::-1])
    plt.xlabel("score")
    plt.title(f"Top {len(top)} konfiguracji")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_thresholds(best_row: Dict[str, object], out_path: Path) -> None:
    if plt is None or not best_row:
        return
    raw = best_row.get("thresholds", "")
    try:
        thresholds = ast.literal_eval(str(raw))
    except Exception:
        return
    if not isinstance(thresholds, (list, tuple)) or not thresholds:
        return

    fig = plt.figure(figsize=(8, 5))
    xs = np.arange(len(thresholds))
    plt.bar(xs, [float(x) for x in thresholds])
    plt.xticks(xs, [f"out_{i}" for i in xs])
    plt.ylim(0, 1)
    plt.ylabel("threshold")
    plt.title("Thresholdy najlepszego modelu")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _read_history_csv(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: safe_float(v) for k, v in row.items()})
    return rows


def _plot_best_history(history_csv: Path, out_dir: Path) -> None:
    rows = _read_history_csv(history_csv)
    if plt is None or not rows:
        return

    for key, ylabel, filename in [
        ("train_loss", "train_loss", "12_best_train_loss_by_epoch.png"),
        ("val_loss", "val_loss", "13_best_val_loss_by_epoch.png"),
        ("macro_f1_tuned", "macro_f1_tuned", "14_best_macro_f1_tuned_by_epoch.png"),
        ("score", "score", "15_best_score_by_epoch.png"),
    ]:
        xs = np.array([safe_float(r.get("epoch")) for r in rows], dtype=float)
        ys = np.array([safe_float(r.get(key)) for r in rows], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() == 0:
            continue
        fig = plt.figure(figsize=(9, 5))
        plt.plot(xs[mask], ys[mask], marker="o")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(f"Najlepszy model: {ylabel} per epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def make_plots(out_dir: Path) -> None:
    if plt is None:
        print(f"Nie mogę stworzyć wykresów, matplotlib się nie importuje: {MATPLOTLIB_IMPORT_ERROR}")
        return

    results_csv = out_dir / "exact_search_results.csv"
    rows = read_successful_results(results_csv)
    if not rows:
        print("Brak udanych wyników do wykresów.")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_line(rows, "score", "Score per trial", "score", plots_dir / "01_score_by_trial.png", best_so_far=True)
    _plot_line(rows, "macro_f1_tuned", "Macro F1 tuned per trial", "macro_f1_tuned", plots_dir / "02_macro_f1_tuned_by_trial.png", best_so_far=True)
    _plot_line(rows, "macro_f1_runtime", "Macro F1 runtime per trial", "macro_f1_runtime", plots_dir / "03_macro_f1_runtime_by_trial.png", best_so_far=True)
    _plot_line(rows, "val_loss", "Validation loss per trial", "val_loss", plots_dir / "04_val_loss_by_trial.png", best_so_far=False)
    _plot_line(rows, "brake_f1_tuned", "Brake F1 tuned per trial", "brake_f1_tuned", plots_dir / "05_brake_f1_tuned_by_trial.png", best_so_far=True)
    _plot_line(rows, "exact_acc_tuned", "Exact accuracy tuned per trial", "exact_acc_tuned", plots_dir / "06_exact_acc_tuned_by_trial.png", best_so_far=True)

    _plot_scatter(rows, "lr", "score", "LR vs score", "lr", plots_dir / "07_lr_vs_score.png", log_x=True)
    _plot_scatter(rows, "weight_decay", "score", "Weight decay vs score", "weight_decay", plots_dir / "08_weight_decay_vs_score.png", log_x=True)
    _plot_scatter(rows, "dropout", "score", "Dropout vs score", "dropout", plots_dir / "09_dropout_vs_score.png")
    _plot_scatter(rows, "batch_size", "score", "Batch size vs score", "batch_size", plots_dir / "10_batch_size_vs_score.png")
    _plot_scatter(rows, "imbalance_cap", "score", "Imbalance cap vs score", "imbalance_cap", plots_dir / "11_imbalance_cap_vs_score.png")
    _plot_scatter(rows, "focal_gamma", "score", "Focal gamma vs score", "focal_gamma", plots_dir / "12_focal_gamma_vs_score.png")

    _plot_top_configs(rows, plots_dir / "13_top20_configs_by_score.png", top_k=20)

    best = max(rows, key=lambda r: safe_float(r.get("score")))
    _plot_thresholds(best, plots_dir / "14_best_thresholds.png")

    _plot_best_history(out_dir / "best_training_history.csv", plots_dir)


def save_best_summary(out_dir: Path) -> None:
    best = best_row_from_csv(out_dir / "exact_search_results.csv")
    if not best:
        return

    best_json = out_dir / "best_config_from_csv.json"
    with best_json.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    summary = out_dir / "summary.txt"
    lines = [
        "BEST RESULT FROM exact_search_results.csv",
        "========================================",
        f"idx: {best.get('idx')}",
        f"score: {best.get('score')}",
        f"val_loss: {best.get('val_loss')}",
        f"macro_f1_runtime: {best.get('macro_f1_runtime')}",
        f"macro_f1_tuned: {best.get('macro_f1_tuned')}",
        f"brake_f1_tuned: {best.get('brake_f1_tuned')}",
        f"exact_acc_tuned: {best.get('exact_acc_tuned')}",
        f"thresholds: {best.get('thresholds')}",
        "",
        "CONFIG",
        "------",
    ]
    for fdef in fields(AdvancedConfig):
        lines.append(f"{fdef.name}: {best.get(fdef.name)}")

    summary.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Search runner
# ============================================================

def run_exact_search(args) -> Optional[Dict[str, object]]:
    set_seed(args.seed)
    setup_torch_for_gpu()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_np, y_train_np, X_val_np, y_val_np, data_path = load_dataset_from_args(args)

    X_train_np = np.asarray(X_train_np, dtype=np.float32)
    X_val_np = np.asarray(X_val_np, dtype=np.float32)
    y_train_np = ensure_binary_y(y_train_np)
    y_val_np = ensure_binary_y(y_val_np)

    if X_train_np.ndim != 2 or X_val_np.ndim != 2:
        raise ValueError(f"X_train/X_val muszą być 2D. Shapes: {X_train_np.shape}, {X_val_np.shape}")
    if y_train_np.ndim != 2 or y_val_np.ndim != 2:
        raise ValueError(f"y_train/y_val muszą być 2D. Shapes: {y_train_np.shape}, {y_val_np.shape}")
    if X_train_np.shape[1] != X_val_np.shape[1]:
        raise ValueError(f"Różny input_dim train/val: {X_train_np.shape[1]} vs {X_val_np.shape[1]}")
    if y_train_np.shape[1] != y_val_np.shape[1]:
        raise ValueError(f"Różny output_dim train/val: {y_train_np.shape[1]} vs {y_val_np.shape[1]}")

    X_train_np, X_val_np = maybe_standardize(X_train_np, X_val_np, out_dir, enabled=args.standardize)

    device = get_device(force_cpu=args.cpu)
    print(f"Using device: {device}")
    print(f"Data path: {data_path}")
    print(f"Output dir: {out_dir}")
    print(f"X_train={X_train_np.shape}, y_train={y_train_np.shape}")
    print(f"X_val={X_val_np.shape}, y_val={y_val_np.shape}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM: {total_vram:.2f} GB")
    else:
        print("UWAGA: CUDA niedostępna albo wymuszono CPU. Search będzie dużo wolniejszy.")

    # Wszystko ładowane od razu na GPU/wybrane urządzenie.
    X_train = to_gpu_tensor(X_train_np, device, dtype=torch.float32)
    y_train = to_gpu_tensor(y_train_np, device, dtype=torch.float32)
    X_val = to_gpu_tensor(X_val_np, device, dtype=torch.float32)
    y_val = to_gpu_tensor(y_val_np, device, dtype=torch.float32)

    # Pozwalamy zwolnić RAM CPU po przerzuceniu na GPU.
    del X_train_np, y_train_np, X_val_np, y_val_np

    if device.type == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"CUDA memory after data preload: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")

    configs = generate_exact_configs(
        search_level=args.search_level,
        include_layernorm=args.include_layernorm,
        include_nearby_architectures=args.include_nearby_architectures,
    )

    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]

    results_csv = out_dir / "exact_search_results.csv"
    best_model_path = out_dir / "best_exact_model.pt"
    best_json_path = out_dir / "best_config.json"
    best_history_path = out_dir / "best_training_history.csv"

    completed = load_completed_hashes(results_csv, retry_errors=args.retry_errors) if args.resume else set()

    print(f"Search level: {args.search_level}")
    print(f"Configs total this run/list: {len(configs)}")
    if completed:
        print(f"Resume: pomijam już ukończone konfiguracje: {len(completed)}")

    best_global_score = -1e18
    best_global: Optional[Dict[str, object]] = None
    start_all = time.time()

    for i, config in enumerate(configs, start=1):
        h = config_hash(config)
        if h in completed:
            continue

        print()
        print(f"[{i}/{len(configs)}] {config}")
        t0 = time.time()

        try:
            metrics, state, history = train_one_config(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                max_epochs=args.max_epochs,
                patience=args.patience,
                brake_index=args.brake_index,
                use_amp=not args.no_amp,
                compile_model=args.compile,
                grad_clip_norm=args.grad_clip_norm,
            )

            seconds = time.time() - t0
            row = {
                "idx": i,
                "config_hash": h,
                **asdict(config),
                **metrics,
                "thresholds": str(metrics["thresholds"]),
                "seconds": round(seconds, 3),
                "error": "",
            }
            append_csv_row(results_csv, row)

            print(
                f"val_loss={metrics['val_loss']:.4f} | "
                f"macro_f1_runtime={metrics['macro_f1_runtime']:.4f} | "
                f"macro_f1_tuned={metrics['macro_f1_tuned']:.4f} | "
                f"brake_f1_tuned={metrics['brake_f1_tuned']:.4f} | "
                f"exact_acc_tuned={metrics['exact_acc_tuned']:.4f} | "
                f"thresholds={metrics['thresholds']} | "
                f"epoch={metrics['epoch']} | "
                f"time={seconds:.1f}s"
            )

            score = float(metrics["score"])
            if score > best_global_score:
                best_global_score = score
                best_global = {
                    "config_hash": h,
                    "config": asdict(config),
                    "metrics": metrics,
                    "data_path": str(data_path),
                }

                torch.save(
                    {
                        "config_hash": h,
                        "config": asdict(config),
                        "metrics": metrics,
                        "model_state_dict": state,
                        "input_dim": int(X_train.shape[1]),
                        "output_dim": int(y_train.shape[1]),
                        "data_path": str(data_path),
                    },
                    best_model_path,
                )
                with best_json_path.open("w", encoding="utf-8") as f:
                    json.dump(best_global, f, indent=2, ensure_ascii=False)
                save_history_csv(best_history_path, history)

                print(">>> NEW BEST SAVED")

            if args.plot_every and args.plot_every > 0 and i % args.plot_every == 0:
                make_plots(out_dir)
                save_best_summary(out_dir)

        except torch.cuda.OutOfMemoryError:
            seconds = time.time() - t0
            print("CUDA OOM — pomijam tę konfigurację.")
            torch.cuda.empty_cache()
            append_csv_row(
                results_csv,
                {
                    "idx": i,
                    "config_hash": h,
                    **asdict(config),
                    "seconds": round(seconds, 3),
                    "error": "CUDA OutOfMemoryError",
                },
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                seconds = time.time() - t0
                print("RuntimeError OOM — pomijam tę konfigurację.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                append_csv_row(
                    results_csv,
                    {
                        "idx": i,
                        "config_hash": h,
                        **asdict(config),
                        "seconds": round(seconds, 3),
                        "error": "RuntimeError OOM",
                    },
                )
            else:
                raise

    make_plots(out_dir)
    save_best_summary(out_dir)

    print()
    print("========== DONE ==========")
    print(f"Total elapsed: {(time.time() - start_all) / 60:.1f} min")
    print(f"Results CSV: {results_csv}")
    print(f"Best model:  {best_model_path}")
    print(f"Plots dir:   {out_dir / 'plots'}")

    best_from_csv = best_row_from_csv(results_csv)
    if best_from_csv:
        print()
        print("========== BEST FROM CSV ==========")
        print(f"idx={best_from_csv.get('idx')}")
        print(f"score={best_from_csv.get('score')}")
        print(f"val_loss={best_from_csv.get('val_loss')}")
        print(f"macro_f1_runtime={best_from_csv.get('macro_f1_runtime')}")
        print(f"macro_f1_tuned={best_from_csv.get('macro_f1_tuned')}")
        print(f"brake_f1_tuned={best_from_csv.get('brake_f1_tuned')}")
        print(f"exact_acc_tuned={best_from_csv.get('exact_acc_tuned')}")
        print(f"thresholds={best_from_csv.get('thresholds')}")

    return best_global


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exact/local GPU search wokół najlepszego configu z random searcha.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data", type=str, default=None, help="Ścieżka do danych: NPZ, CSV albo katalog z NPY.")
    p.add_argument("--out-dir", type=str, default="exact_search", help="Katalog wyników.")
    p.add_argument("--target-cols", type=str, default=None, help="Dla CSV, np. left,right,brake,throttle")
    p.add_argument("--feature-cols", type=str, default=None, help="Dla CSV, lista feature columns; domyślnie numeric bez targetów.")
    p.add_argument("--val-fraction", type=float, default=0.20, help="Gdy dane mają tylko X,y, skrypt robi split.")
    p.add_argument("--standardize", action="store_true", help="Standaryzuje X train/val na podstawie train.")

    p.add_argument("--search-level", type=str, default="fast", choices=["fast", "balanced", "large"], help="Rozmiar lokalnego exact gridu.")
    p.add_argument("--max-configs", type=int, default=0, help="0 = wszystkie z danego search-level. Inaczej bierze najbliższe bazowemu punktowi.")
    p.add_argument("--max-epochs", type=int, default=70, help="Maksymalna liczba epok na config.")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    p.add_argument("--seed", type=int, default=123, help="Seed.")
    p.add_argument("--brake-index", type=int, default=2, help="Indeks wyjścia odpowiadającego brake.")
    p.add_argument("--grad-clip-norm", type=float, default=5.0, help="Gradient clipping; <=0 wyłącza.")

    p.add_argument("--include-layernorm", action="store_true", help="Dodaje norm=layer i norm=none do searcha.")
    p.add_argument("--include-nearby-architectures", action="store_true", help="Dodaje kilka struktur sieci blisko bazowej.")
    p.add_argument("--compile", action="store_true", help="Używa torch.compile. Zwykle lepsze dla finalnego treningu niż dla wielu krótkich triali.")
    p.add_argument("--no-amp", action="store_true", help="Wyłącza mixed precision AMP.")
    p.add_argument("--cpu", action="store_true", help="Wymusza CPU.")

    p.add_argument("--resume", action="store_true", help="Kontynuuje search, pomijając config_hash obecne w CSV.")
    p.add_argument("--retry-errors", action="store_true", help="Przy --resume ponawia konfiguracje zakończone błędem.")
    p.add_argument("--plot-every", type=int, default=25, help="Co ile triali odświeżać wykresy. 0 = tylko na końcu.")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0:
        args.grad_clip_norm = None

    try:
        run_exact_search(args)
    except KeyboardInterrupt:
        print("\nPrzerwano ręcznie. Dotychczasowy CSV i okresowe wykresy zostają w katalogu wyników.")
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
