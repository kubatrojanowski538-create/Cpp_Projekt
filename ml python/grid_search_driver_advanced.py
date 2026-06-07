"""
Advanced search for the 2D car neural driver.

Place this file next to `grid_search_driver_basic.py`, then run for example:

    python grid_search_driver_advanced.py --csv "ml data/GameStates1.csv" --epochs 50 --max-runs 100

This script searches a much larger space than the basic version:
    - MLP architecture: number of layers and neurons
    - activation: ReLU/GELU/SiLU
    - normalization: none/batch/layer
    - dropout
    - optimizer hyperparameters
    - imbalance weights
    - output formulation:
        1) binary: original 4 independent logits + BCE/Focal BCE
        2) factorized: throttle class {coast, accelerate, brake}
                      steering class {straight, left, right}

The factorized output is useful for testing whether mutually exclusive output heads
reduce left/right oscillation and accelerator/brake ambiguity.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Reuse data loading and feature building from the basic script.
# Keep both files in the same directory.
from grid_search_driver_basic import (  # noqa: E402
    LABEL_COLS,
    RUNTIME_THRESHOLDS,
    action_predictions,
    build_features,
    default_project_csv,
    get_device,
    load_data,
    split_dataframe,
    tune_thresholds_for_f1,
    set_seed,
)


class AdvancedDataset(Dataset):
    def __init__(self, x: np.ndarray, y_bin: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_bin = torch.tensor(y_bin, dtype=torch.float32)
        throttle, steering = make_factorized_targets(y_bin)
        self.y_throttle = torch.tensor(throttle, dtype=torch.long)
        self.y_steering = torch.tensor(steering, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y_bin[index], self.y_throttle[index], self.y_steering[index]


@dataclass(frozen=True)
class AdvancedConfig:
    model_type: str       # binary | factorized
    hidden_layers: str    # e.g. "128,128,64"
    activation: str       # relu | gelu | silu
    norm: str             # none | batch | layer
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    imbalance_cap: float
    loss_variant: str     # bce | focal | ce
    focal_gamma: float


def parse_hidden_layers(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLPBackbone(nn.Module):
    def __init__(self, input_size: int, hidden_layers: Sequence[int], activation: str, dropout: float, norm: str):
        super().__init__()
        layers: List[nn.Module] = []
        current = input_size
        for hidden in hidden_layers:
            layers.append(nn.Linear(current, hidden))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(hidden))
            elif norm == "layer":
                layers.append(nn.LayerNorm(hidden))
            elif norm == "none":
                pass
            else:
                raise ValueError(f"Unknown norm: {norm}")
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden
        self.output_size = current
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BinaryDriverNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: Sequence[int], activation: str, dropout: float, norm: str):
        super().__init__()
        self.backbone = MLPBackbone(input_size, hidden_layers, activation, dropout, norm)
        self.head = nn.Linear(self.backbone.output_size, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class FactorizedDriverNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: Sequence[int], activation: str, dropout: float, norm: str):
        super().__init__()
        self.backbone = MLPBackbone(input_size, hidden_layers, activation, dropout, norm)
        self.throttle_head = nn.Linear(self.backbone.output_size, 3)
        self.steering_head = nn.Linear(self.backbone.output_size, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.throttle_head(h), self.steering_head(h)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        return (focal_factor * bce).mean()


def make_factorized_targets(y_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 4 binary labels into two mutually exclusive classes.

    throttle: 0=coast, 1=accelerate, 2=brake. Brake wins if both appear.
    steering: 0=straight, 1=left, 2=right. If both appear, stronger cleanup is needed;
              here left wins only if right is not active, otherwise right wins.
    """
    throttle = np.zeros(len(y_bin), dtype=np.int64)
    throttle[y_bin[:, 0] > 0.5] = 1
    throttle[y_bin[:, 1] > 0.5] = 2

    steering = np.zeros(len(y_bin), dtype=np.int64)
    steering[y_bin[:, 2] > 0.5] = 1
    steering[y_bin[:, 3] > 0.5] = 2
    return throttle, steering


def factorized_classes_to_binary(throttle: np.ndarray, steering: np.ndarray) -> np.ndarray:
    pred = np.zeros((len(throttle), 4), dtype=np.int64)
    pred[throttle == 1, 0] = 1
    pred[throttle == 2, 1] = 1
    pred[steering == 1, 2] = 1
    pred[steering == 2, 3] = 1
    return pred


def compute_binary_pos_weight(y_train: np.ndarray, cap: float) -> np.ndarray:
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    return np.clip(pos_weight, 0.5, cap).astype(np.float32)


def compute_class_weights(classes: np.ndarray, num_classes: int, cap: float) -> np.ndarray:
    counts = np.bincount(classes, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.25, cap)
    return weights.astype(np.float32)


def safe_average_precision(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
    if len(np.unique(y_true_col)) < 2:
        return float("nan")
    return float(average_precision_score(y_true_col, y_prob_col))


def compute_metrics_from_preds(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    per_label_f1 = []
    per_label_balanced = []
    per_label_ap = []

    result[f"exact_acc_{prefix}"] = float((y_pred == y_true).all(axis=1).mean())
    result[f"both_throttle_pred_rate_{prefix}"] = float(((y_pred[:, 0] == 1) & (y_pred[:, 1] == 1)).mean())
    result[f"both_steer_pred_rate_{prefix}"] = float(((y_pred[:, 2] == 1) & (y_pred[:, 3] == 1)).mean())

    for i, name in enumerate(LABEL_COLS):
        yt = y_true[:, i].astype(np.int64)
        yp = y_pred[:, i].astype(np.int64)
        prob = y_prob[:, i]

        precision = precision_score(yt, yp, zero_division=0)
        recall = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        balanced = balanced_accuracy_score(yt, yp) if len(np.unique(yt)) >= 2 else float("nan")
        ap = safe_average_precision(yt, prob)
        cm = confusion_matrix(yt, yp, labels=[0, 1])

        result[f"{name}_precision_{prefix}"] = float(precision)
        result[f"{name}_recall_{prefix}"] = float(recall)
        result[f"{name}_f1_{prefix}"] = float(f1)
        result[f"{name}_balanced_acc_{prefix}"] = float(balanced)
        result[f"{name}_pr_auc"] = float(ap)
        result[f"{name}_true_pos_rate"] = float(yt.mean())
        result[f"{name}_pred_pos_rate_{prefix}"] = float(yp.mean())
        result[f"{name}_tn_{prefix}"] = int(cm[0, 0])
        result[f"{name}_fp_{prefix}"] = int(cm[0, 1])
        result[f"{name}_fn_{prefix}"] = int(cm[1, 0])
        result[f"{name}_tp_{prefix}"] = int(cm[1, 1])

        per_label_f1.append(f1)
        if not math.isnan(balanced):
            per_label_balanced.append(balanced)
        if not math.isnan(ap):
            per_label_ap.append(ap)

    result[f"macro_f1_{prefix}"] = float(np.mean(per_label_f1))
    result[f"macro_balanced_acc_{prefix}"] = float(np.mean(per_label_balanced)) if per_label_balanced else float("nan")
    result["macro_pr_auc"] = float(np.mean(per_label_ap)) if per_label_ap else float("nan")
    return result


def thresholded_control_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, prefix: str) -> Dict[str, Any]:
    y_pred = action_predictions(y_prob, thresholds)
    return compute_metrics_from_preds(y_true, y_prob, y_pred, prefix=prefix)


def build_model(config: AdvancedConfig, input_size: int) -> nn.Module:
    hidden_layers = parse_hidden_layers(config.hidden_layers)
    if config.model_type == "binary":
        return BinaryDriverNet(input_size, hidden_layers, config.activation, config.dropout, config.norm)
    if config.model_type == "factorized":
        return FactorizedDriverNet(input_size, hidden_layers, config.activation, config.dropout, config.norm)
    raise ValueError(f"Unknown model_type: {config.model_type}")


def build_losses(config: AdvancedConfig, y_train: np.ndarray, device: torch.device) -> Dict[str, nn.Module]:
    if config.model_type == "binary":
        pos_weight = torch.tensor(compute_binary_pos_weight(y_train, config.imbalance_cap), dtype=torch.float32, device=device)
        if config.loss_variant == "bce":
            return {"binary": nn.BCEWithLogitsLoss(pos_weight=pos_weight)}
        if config.loss_variant == "focal":
            return {"binary": FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=config.focal_gamma)}
        raise ValueError("Binary model supports loss_variant=bce or focal")

    if config.model_type == "factorized":
        throttle, steering = make_factorized_targets(y_train)
        throttle_weight = torch.tensor(compute_class_weights(throttle, 3, config.imbalance_cap), dtype=torch.float32, device=device)
        steering_weight = torch.tensor(compute_class_weights(steering, 3, config.imbalance_cap), dtype=torch.float32, device=device)
        return {
            "throttle": nn.CrossEntropyLoss(weight=throttle_weight),
            "steering": nn.CrossEntropyLoss(weight=steering_weight),
        }

    raise ValueError(f"Unknown model_type: {config.model_type}")


def batch_loss(
    model: nn.Module,
    config: AdvancedConfig,
    losses: Dict[str, nn.Module],
    x: torch.Tensor,
    y_bin: torch.Tensor,
    y_throttle: torch.Tensor,
    y_steering: torch.Tensor,
) -> torch.Tensor:
    if config.model_type == "binary":
        logits = model(x)
        return losses["binary"](logits, y_bin)

    throttle_logits, steering_logits = model(x)
    return losses["throttle"](throttle_logits, y_throttle) + losses["steering"](steering_logits, y_steering)


def predict_probs_and_loss(
    model: nn.Module,
    config: AdvancedConfig,
    losses: Dict[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    samples = 0
    all_y: List[np.ndarray] = []
    all_prob4: List[np.ndarray] = []
    all_runtime_pred: List[np.ndarray] = []

    with torch.no_grad():
        for x, y_bin, y_throttle, y_steering in loader:
            x = x.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_throttle = y_throttle.to(device, non_blocking=True)
            y_steering = y_steering.to(device, non_blocking=True)

            loss = batch_loss(model, config, losses, x, y_bin, y_throttle, y_steering)
            loss_sum += loss.item() * len(x)
            samples += len(x)

            if config.model_type == "binary":
                logits = model(x)
                probs4 = torch.sigmoid(logits).cpu().numpy()
                runtime_pred = action_predictions(probs4, RUNTIME_THRESHOLDS)
            else:
                throttle_logits, steering_logits = model(x)
                throttle_probs = torch.softmax(throttle_logits, dim=1).cpu().numpy()
                steering_probs = torch.softmax(steering_logits, dim=1).cpu().numpy()
                probs4 = np.column_stack([
                    throttle_probs[:, 1],
                    throttle_probs[:, 2],
                    steering_probs[:, 1],
                    steering_probs[:, 2],
                ])
                throttle_class = throttle_probs.argmax(axis=1)
                steering_class = steering_probs.argmax(axis=1)
                runtime_pred = factorized_classes_to_binary(throttle_class, steering_class)

            all_y.append(y_bin.cpu().numpy())
            all_prob4.append(probs4)
            all_runtime_pred.append(runtime_pred)

    return (
        loss_sum / max(samples, 1),
        np.vstack(all_y),
        np.vstack(all_prob4),
        np.vstack(all_runtime_pred),
    )


def train_one_config(
    config: AdvancedConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    num_workers: int,
    seed: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    set_seed(seed)

    train_dataset = AdvancedDataset(x_train, y_train)
    val_dataset = AdvancedDataset(x_val, y_val)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > config.batch_size and config.norm == "batch",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(512, config.batch_size * 2),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(config, input_size=x_train.shape[1]).to(device)
    losses = build_losses(config, y_train, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for x, y_bin, y_throttle, y_steering in train_loader:
            x = x.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_throttle = y_throttle.to(device, non_blocking=True)
            y_steering = y_steering.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, config, losses, x, y_bin, y_throttle, y_steering)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(x)
            train_samples += len(x)

        train_loss = train_loss_sum / max(train_samples, 1)
        val_loss, _, _, _ = predict_probs_and_loss(model, config, losses, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
    }


def build_advanced_grid() -> List[AdvancedConfig]:
    hidden_options = [
        "64,64",
        "128,64",
        "128,128,64",
        "256,128,64",
        "256,256,128,64",
    ]
    activation_options = ["relu", "gelu", "silu"]
    norm_options = ["none", "batch", "layer"]
    dropout_options = [0.0, 0.10, 0.20]
    lr_options = [1e-3, 5e-4]
    wd_options = [1e-4, 1e-3]
    batch_options = [2048, 4096]
    cap_options = [5.0, 10.0, 20.0]

    configs: List[AdvancedConfig] = []
    shared_product = itertools.product(
        hidden_options,
        activation_options,
        norm_options,
        dropout_options,
        lr_options,
        wd_options,
        batch_options,
        cap_options,
    )

    shared = list(shared_product)
    for hidden, activation, norm, dropout, lr, wd, batch_size, cap in shared:
        for loss_variant in ["bce", "focal"]:
            configs.append(AdvancedConfig(
                model_type="binary",
                hidden_layers=hidden,
                activation=activation,
                norm=norm,
                dropout=dropout,
                lr=lr,
                weight_decay=wd,
                batch_size=batch_size,
                imbalance_cap=cap,
                loss_variant=loss_variant,
                focal_gamma=2.0,
            ))
        configs.append(AdvancedConfig(
            model_type="factorized",
            hidden_layers=hidden,
            activation=activation,
            norm=norm,
            dropout=dropout,
            lr=lr,
            weight_decay=wd,
            batch_size=batch_size,
            imbalance_cap=cap,
            loss_variant="ce",
            focal_gamma=0.0,
        ))

    return configs


def maybe_limit_configs(configs: List[AdvancedConfig], max_runs: int, seed: int) -> List[AdvancedConfig]:
    if max_runs <= 0 or max_runs >= len(configs):
        return configs
    rng = random.Random(seed)
    configs = configs.copy()
    rng.shuffle(configs)
    return configs[:max_runs]


def format_config_short(row: pd.Series) -> str:
    return (
        f"{row['model_type']}/{row['loss_variant']} layers={row['hidden_layers']} "
        f"act={row['activation']} norm={row['norm']} drop={row['dropout']} "
        f"lr={row['lr']} wd={row['weight_decay']} bs={int(row['batch_size'])} cap={row['imbalance_cap']}"
    )


def save_rankings(results_df: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    metrics = [
        ("val_loss", True),
        ("macro_f1_runtime", False),
        ("macro_f1_tuned", False),
        ("brake_f1_runtime", False),
        ("brake_f1_tuned", False),
        ("brake_recall_tuned", False),
        ("exact_acc_runtime", False),
        ("exact_acc_tuned", False),
        ("macro_pr_auc", False),
    ]

    lines = ["# Grid search ranking — advanced", ""]
    lines.append("Runtime metrics use actual control-like decoding: brake priority over acceleration and left/right mutual exclusion.")
    lines.append("Tuned metrics use thresholds found on the validation set, also decoded with the same control-like logic.")
    lines.append("")

    for metric, ascending in metrics:
        if metric not in results_df.columns:
            continue
        subset = results_df.sort_values(metric, ascending=ascending).head(top_n)
        lines.append(f"## Top {top_n}: `{metric}`")
        lines.append("")
        lines.append("| rank | combo_id | value | config | thresholds_tuned |")
        lines.append("|---:|---:|---:|---|---|")
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            value = row[metric]
            value_str = f"{value:.6f}" if pd.notna(value) else "nan"
            lines.append(
                f"| {rank} | {int(row['combo_id'])} | {value_str} | {format_config_short(row)} | `{row['thresholds_tuned']}` |"
            )
        lines.append("")

    # Aggregate winners by architecture/output type.
    lines.append("## Mean metrics by model_type/loss_variant")
    lines.append("")
    group_cols = ["model_type", "loss_variant"]
    agg_cols = [col for col in ["macro_f1_runtime", "macro_f1_tuned", "brake_f1_tuned", "macro_pr_auc"] if col in results_df.columns]
    if agg_cols:
        grouped = results_df.groupby(group_cols)[agg_cols].mean(numeric_only=True).reset_index()
        lines.append("| model_type | loss_variant | " + " | ".join(agg_cols) + " |")
        lines.append("|---|---|" + "---:|" * len(agg_cols))
        for _, row in grouped.iterrows():
            values = " | ".join(f"{row[col]:.6f}" if pd.notna(row[col]) else "nan" for col in agg_cols)
            lines.append(f"| {row['model_type']} | {row['loss_variant']} | {values} |")
        lines.append("")

    (out_dir / "top_by_metric.md").write_text("\n".join(lines), encoding="utf-8")


def plot_top_metric(results_df: pd.DataFrame, out_dir: Path, metric: str, top_n: int, ascending: bool = False) -> None:
    if metric not in results_df.columns:
        return
    subset = results_df.sort_values(metric, ascending=ascending).head(top_n).copy().iloc[::-1]
    labels = [f"#{int(v)}" for v in subset["combo_id"]]
    plt.figure(figsize=(10, max(4, 0.45 * len(subset))))
    plt.barh(labels, subset[metric])
    plt.xlabel(metric)
    plt.ylabel("combo_id")
    plt.title(f"Top {top_n}: {metric}")
    plt.tight_layout()
    plt.savefig(out_dir / f"top_{metric}.png", dpi=160)
    plt.close()


def plot_scatter(results_df: pd.DataFrame, out_dir: Path) -> None:
    if "macro_f1_tuned" not in results_df.columns or "brake_f1_tuned" not in results_df.columns:
        return
    plt.figure(figsize=(8, 6))
    for model_type in sorted(results_df["model_type"].unique()):
        subset = results_df[results_df["model_type"] == model_type]
        plt.scatter(subset["macro_f1_tuned"], subset["brake_f1_tuned"], label=model_type)
    plt.xlabel("macro_f1_tuned")
    plt.ylabel("brake_f1_tuned")
    plt.title("Trade-off: macro F1 vs brake F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_macro_f1_vs_brake_f1.png", dpi=160)
    plt.close()


def plot_grouped_mean(results_df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    if metric not in results_df.columns:
        return
    grouped = results_df.groupby(["model_type", "loss_variant"])[metric].mean().sort_values()
    labels = ["/".join(idx) for idx in grouped.index]
    plt.figure(figsize=(8, max(4, 0.45 * len(grouped))))
    plt.barh(labels, grouped.values)
    plt.xlabel(f"mean {metric}")
    plt.title(f"Mean {metric} by output/loss type")
    plt.tight_layout()
    plt.savefig(out_dir / f"mean_{metric}_by_model_type.png", dpi=160)
    plt.close()


def save_plots(results_df: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    plot_top_metric(results_df, out_dir, "macro_f1_runtime", top_n)
    plot_top_metric(results_df, out_dir, "macro_f1_tuned", top_n)
    plot_top_metric(results_df, out_dir, "brake_f1_tuned", top_n)
    plot_top_metric(results_df, out_dir, "brake_recall_tuned", top_n)
    plot_top_metric(results_df, out_dir, "exact_acc_runtime", top_n)
    plot_top_metric(results_df, out_dir, "val_loss", top_n, ascending=True)
    plot_scatter(results_df, out_dir)
    plot_grouped_mean(results_df, out_dir, "macro_f1_tuned")
    plot_grouped_mean(results_df, out_dir, "brake_f1_tuned")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=default_project_csv())
    parser.add_argument("--out", type=Path, default=Path("grid_results_advanced"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--group-col", type=str, default="episode_id")
    parser.add_argument("--drop-duplicates", action="store_true")
    parser.add_argument("--drop-leading-idle", action="store_true")
    parser.add_argument("--max-runs", type=int, default=100, help="Default caps the huge grid. Use 0 for full exhaustive grid.")
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(
        csv_path=args.csv,
        drop_duplicates=args.drop_duplicates,
        drop_leading_idle=args.drop_leading_idle,
        group_col=args.group_col,
    )
    train_df, val_df = split_dataframe(df, args.val_size, args.seed, args.group_col)

    x_train, scaler = build_features(train_df, fit_scaler=True)
    x_val, _ = build_features(val_df, scaler=scaler, fit_scaler=False)
    y_train = train_df[LABEL_COLS].astype(np.float32).to_numpy()
    y_val = val_df[LABEL_COLS].astype(np.float32).to_numpy()
    joblib.dump(scaler, out_dir / "scaler_used_for_grid.joblib")

    configs = maybe_limit_configs(build_advanced_grid(), args.max_runs, args.seed)
    print(f"Search runs: {len(configs)}")

    rows: List[Dict[str, Any]] = []
    start_all = time.time()

    for combo_id, config in enumerate(configs, start=1):
        print(f"\n[{combo_id}/{len(configs)}] {config}")
        start = time.time()

        model, train_info = train_one_config(
            config=config,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed + combo_id,
        )

        val_dataset = AdvancedDataset(x_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(512, config.batch_size * 2),
            shuffle=False,
            pin_memory=device.type == "cuda",
            num_workers=args.num_workers,
        )
        losses = build_losses(config, y_train, device)
        val_loss, y_true, y_prob, runtime_pred = predict_probs_and_loss(model, config, losses, val_loader, device)

        tuned_thresholds, threshold_info = tune_thresholds_for_f1(y_true, y_prob, step=args.threshold_step)
        metrics_runtime = compute_metrics_from_preds(y_true, y_prob, runtime_pred, prefix="runtime")
        metrics_tuned = thresholded_control_metrics(y_true, y_prob, tuned_thresholds, prefix="tuned")

        row: Dict[str, Any] = {
            "combo_id": combo_id,
            **asdict(config),
            "val_loss": float(val_loss),
            "best_epoch": int(train_info["best_epoch"]),
            "elapsed_sec": round(time.time() - start, 3),
            "thresholds_runtime": json.dumps(RUNTIME_THRESHOLDS.tolist()),
            "thresholds_tuned": json.dumps([round(float(x), 4) for x in tuned_thresholds.tolist()]),
            "param_count": int(sum(p.numel() for p in model.parameters())),
        }
        row.update(threshold_info)
        row.update(metrics_runtime)
        row.update(metrics_tuned)
        rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df.to_csv(out_dir / "grid_results_advanced.csv", index=False)
        save_rankings(results_df, out_dir, args.top_n)
        save_plots(results_df, out_dir, args.top_n)

        print(
            f"val_loss={val_loss:.4f} | macro_f1_runtime={row['macro_f1_runtime']:.4f} | "
            f"macro_f1_tuned={row['macro_f1_tuned']:.4f} | brake_f1_tuned={row['brake_f1_tuned']:.4f} | "
            f"thresholds={row['thresholds_tuned']}"
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "grid_results_advanced.csv", index=False)
    save_rankings(results_df, out_dir, args.top_n)
    save_plots(results_df, out_dir, args.top_n)

    print(f"\nDone in {(time.time() - start_all) / 60:.2f} min")
    print("Results:", out_dir / "grid_results_advanced.csv")
    print("Ranking:", out_dir / "top_by_metric.md")
    print("Plots:", out_dir)


if __name__ == "__main__":
    main()
