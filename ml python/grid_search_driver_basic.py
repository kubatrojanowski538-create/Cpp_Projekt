"""
Basic grid search for the 2D car neural driver.

Place this file in the project root or in `ml python/` and run, for example:

    python grid_search_driver_basic.py --csv "ml data/GameStates1.csv" --epochs 40 --top-n 10

The script keeps the original 4-output BCE setup:
    accelerate, brake, steer_left, steer_right

It searches the most important hyperparameters first:
    - learning rate
    - weight decay
    - dropout
    - batch size
    - pos_weight cap for imbalanced labels

It also tunes decision thresholds on the validation split, because for this project the
runtime thresholds can matter as much as the model weights.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


RAY_COUNT = 20
LABEL_COLS = ["accelerate", "brake", "steer_left", "steer_right"]
DIST_COLS = [f"ray_dist_{i}" for i in range(RAY_COUNT)]
TYPE_COLS = [f"ray_type_{i}" for i in range(RAY_COUNT)]
NUMERIC_COLS = ["car_speed"] + DIST_COLS
RAY_TYPE_VALUES = [-1, 0, 1]

# Thresholds copied from the current C++ runtime idea:
# accelerate > 0.30, brake > 0.50, steering > 0.35.
RUNTIME_THRESHOLDS = np.array([0.30, 0.50, 0.35, 0.35], dtype=np.float32)


class GameStateDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class DriverNet(nn.Module):
    """Original-style MLP, but dropout is configurable."""

    def __init__(self, input_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class BasicConfig:
    lr: float
    weight_decay: float
    dropout: float
    batch_size: int
    pos_weight_cap: float


def default_project_csv() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "ml data" / "GameStates1.csv",
        script_dir.parent / "ml data" / "GameStates1.csv",
        Path.cwd() / "ml data" / "GameStates1.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: Path, drop_duplicates: bool, drop_leading_idle: bool, group_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    required_cols = NUMERIC_COLS + TYPE_COLS + LABEL_COLS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_dropna = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped_by_numeric_cleanup = before_dropna - len(df)
    if len(df) == 0:
        raise ValueError("No valid rows after numeric conversion/dropna. Check the CSV file.")

    df[LABEL_COLS] = (df[LABEL_COLS] > 0.5).astype(np.float32)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=required_cols).reset_index(drop=True)
        print(f"drop_duplicates: removed {before - len(df)} rows")

    if drop_leading_idle:
        before = len(df)
        df = drop_leading_idle_rows(df, group_col=group_col).reset_index(drop=True)
        print(f"drop_leading_idle: removed {before - len(df)} rows")

    print(f"Rows loaded: {len(df)}  | dropped by numeric cleanup: {dropped_by_numeric_cleanup}")
    print("Positive label rates:")
    print((df[LABEL_COLS].mean() * 100).round(3).astype(str) + "%")
    print("Ray type values found:", sorted(pd.unique(df[TYPE_COLS].values.ravel())))
    return df


def drop_leading_idle_rows(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Removes initial frames before the first input or movement.
    If episode_id/group_col exists, it is applied per episode; otherwise globally.
    """

    def trim_group(group: pd.DataFrame) -> pd.DataFrame:
        active = (
            (group["car_speed"].abs() > 0.2)
            | (group[LABEL_COLS].sum(axis=1) > 0)
        )
        if not active.any():
            return group.iloc[0:0]
        first_pos = np.flatnonzero(active.to_numpy())[0]
        return group.iloc[first_pos:]

    if group_col in df.columns:
        return df.groupby(group_col, group_keys=False).apply(trim_group)
    return trim_group(df)


def split_dataframe(
    df: pd.DataFrame,
    val_size: float,
    seed: int,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_col in df.columns and df[group_col].nunique() >= 3:
        groups = np.array(sorted(df[group_col].dropna().unique()))
        train_groups, val_groups = train_test_split(groups, test_size=val_size, random_state=seed, shuffle=True)
        train_df = df[df[group_col].isin(train_groups)].copy()
        val_df = df[df[group_col].isin(val_groups)].copy()
        print(f"Group split by {group_col}: train groups={len(train_groups)}, val groups={len(val_groups)}")
    else:
        train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed, shuffle=True)
        print("Random row split. For more honest validation, add episode_id/track_id and use grouped split.")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit_scaler: bool = False) -> Tuple[np.ndarray, StandardScaler]:
    numeric = df[NUMERIC_COLS].astype(np.float32).to_numpy()

    if fit_scaler:
        scaler = StandardScaler()
        numeric = scaler.fit_transform(numeric).astype(np.float32)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        numeric = scaler.transform(numeric).astype(np.float32)

    type_features: List[np.ndarray] = []
    for col in TYPE_COLS:
        values = df[col].astype(int).to_numpy()
        for ray_type in RAY_TYPE_VALUES:
            type_features.append((values == ray_type).astype(np.float32).reshape(-1, 1))

    x = np.concatenate([numeric] + type_features, axis=1).astype(np.float32)
    return x, scaler


def compute_pos_weight(y_train: np.ndarray, cap: float) -> np.ndarray:
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    pos_weight = np.clip(pos_weight, 0.5, cap)
    return pos_weight.astype(np.float32)


def train_one_config(
    config: BasicConfig,
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

    train_dataset = GameStateDataset(x_train, y_train)
    val_dataset = GameStateDataset(x_val, y_val)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > config.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(512, config.batch_size * 2),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = DriverNet(input_size=x_train.shape[1], dropout=config.dropout).to(device)
    pos_weight = compute_pos_weight(y_train, cap=config.pos_weight_cap)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
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

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(x)
            train_samples += len(x)

        train_loss = train_loss_sum / max(train_samples, 1)
        val_loss, _, _ = predict_and_loss(model, val_loader, loss_fn, device)
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
        "pos_weight": pos_weight.tolist(),
        "history": history,
    }


def predict_and_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    samples = 0
    all_y: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            probs = torch.sigmoid(logits)

            loss_sum += loss.item() * len(x)
            samples += len(x)
            all_y.append(y.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

    avg_loss = loss_sum / max(samples, 1)
    return avg_loss, np.vstack(all_y), np.vstack(all_prob)


def action_predictions(y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Converts probabilities into actual control-like binary outputs.

    This follows the C++ runtime idea more closely than independent thresholds:
    - brake has priority over accelerate,
    - steering chooses left OR right, not both.
    """
    pred = np.zeros_like(y_prob, dtype=np.int64)
    t_acc, t_brake, t_left, t_right = thresholds.tolist()

    p_acc = y_prob[:, 0]
    p_brake = y_prob[:, 1]
    p_left = y_prob[:, 2]
    p_right = y_prob[:, 3]

    brake_mask = p_brake >= t_brake
    acc_mask = (~brake_mask) & (p_acc >= t_acc)
    pred[brake_mask, 1] = 1
    pred[acc_mask, 0] = 1

    left_active = p_left >= t_left
    right_active = p_right >= t_right
    steer_active = left_active | right_active
    choose_left = steer_active & (p_left >= p_right)
    choose_right = steer_active & (p_right > p_left)
    pred[choose_left, 2] = 1
    pred[choose_right, 3] = 1

    return pred


def tune_thresholds_for_f1(y_true: np.ndarray, y_prob: np.ndarray, step: float) -> Tuple[np.ndarray, Dict[str, float]]:
    candidate_thresholds = np.arange(step, 1.0, step)
    tuned = np.zeros(len(LABEL_COLS), dtype=np.float32)
    per_label_f1: Dict[str, float] = {}

    for i, name in enumerate(LABEL_COLS):
        best_thr = 0.5
        best_f1 = -1.0
        if len(np.unique(y_true[:, i])) < 2:
            tuned[i] = 0.5
            per_label_f1[f"{name}_best_f1_for_threshold"] = float("nan")
            continue
        for thr in candidate_thresholds:
            pred = (y_prob[:, i] >= thr).astype(np.int64)
            score = f1_score(y_true[:, i], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)
        tuned[i] = best_thr
        per_label_f1[f"{name}_best_f1_for_threshold"] = float(best_f1)

    return tuned, per_label_f1


def safe_average_precision(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
    if len(np.unique(y_true_col)) < 2:
        return float("nan")
    return float(average_precision_score(y_true_col, y_prob_col))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, prefix: str) -> Dict[str, Any]:
    y_pred = action_predictions(y_prob, thresholds)
    result: Dict[str, Any] = {}

    per_label_f1 = []
    per_label_balanced = []
    per_label_ap = []

    exact_acc = (y_pred == y_true).all(axis=1).mean()
    result[f"exact_acc_{prefix}"] = float(exact_acc)

    for i, name in enumerate(LABEL_COLS):
        yt = y_true[:, i].astype(np.int64)
        yp = y_pred[:, i].astype(np.int64)
        prob = y_prob[:, i]

        precision = precision_score(yt, yp, zero_division=0)
        recall = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        if len(np.unique(yt)) < 2:
            balanced = float("nan")
        else:
            balanced = balanced_accuracy_score(yt, yp)
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


def build_grid() -> List[BasicConfig]:
    # This is intentionally not enormous. It targets the parameters that matter most first.
    values = {
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-3],
        "dropout": [0.05, 0.10, 0.20],
        "batch_size": [128, 256],
        "pos_weight_cap": [5.0, 10.0, 20.0],
    }
    configs = []
    keys = list(values.keys())
    for combo in itertools.product(*(values[k] for k in keys)):
        configs.append(BasicConfig(**dict(zip(keys, combo))))
    return configs


def maybe_limit_configs(configs: List[BasicConfig], max_runs: int, seed: int) -> List[BasicConfig]:
    if max_runs <= 0 or max_runs >= len(configs):
        return configs
    rng = random.Random(seed)
    configs = configs.copy()
    rng.shuffle(configs)
    return configs[:max_runs]


def format_config_short(row: pd.Series) -> str:
    return (
        f"lr={row['lr']}, wd={row['weight_decay']}, drop={row['dropout']}, "
        f"bs={int(row['batch_size'])}, pwcap={row['pos_weight_cap']}"
    )


def save_rankings(results_df: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    metrics = [
        ("val_loss", True),
        ("macro_f1_tuned", False),
        ("brake_f1_tuned", False),
        ("brake_recall_tuned", False),
        ("brake_precision_tuned", False),
        ("exact_acc_tuned", False),
        ("macro_pr_auc", False),
    ]

    lines = ["# Grid search ranking — basic", ""]
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

    (out_dir / "top_by_metric.md").write_text("\n".join(lines), encoding="utf-8")


def plot_top_metric(results_df: pd.DataFrame, out_dir: Path, metric: str, top_n: int, ascending: bool = False) -> None:
    if metric not in results_df.columns:
        return
    subset = results_df.sort_values(metric, ascending=ascending).head(top_n).copy()
    subset = subset.iloc[::-1]
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
    x = "macro_f1_tuned"
    y = "brake_f1_tuned"
    if x not in results_df.columns or y not in results_df.columns:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(results_df[x], results_df[y])
    for _, row in results_df.nlargest(min(10, len(results_df)), x).iterrows():
        plt.annotate(str(int(row["combo_id"])), (row[x], row[y]), fontsize=8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("Trade-off: macro F1 vs brake F1")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_macro_f1_vs_brake_f1.png", dpi=160)
    plt.close()


def save_plots(results_df: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    plot_top_metric(results_df, out_dir, "macro_f1_tuned", top_n)
    plot_top_metric(results_df, out_dir, "brake_f1_tuned", top_n)
    plot_top_metric(results_df, out_dir, "brake_recall_tuned", top_n)
    plot_top_metric(results_df, out_dir, "exact_acc_tuned", top_n)
    plot_top_metric(results_df, out_dir, "val_loss", top_n, ascending=True)
    plot_scatter(results_df, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=default_project_csv())
    parser.add_argument("--out", type=Path, default=Path("grid_results_basic"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--group-col", type=str, default="episode_id")
    parser.add_argument("--drop-duplicates", action="store_true")
    parser.add_argument("--drop-leading-idle", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 = run full grid")
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

    configs = maybe_limit_configs(build_grid(), args.max_runs, args.seed)
    print(f"Grid runs: {len(configs)}")

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

        val_dataset = GameStateDataset(x_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(512, config.batch_size * 2),
            shuffle=False,
            pin_memory=device.type == "cuda",
            num_workers=args.num_workers,
        )
        pos_weight = torch.tensor(train_info["pos_weight"], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        val_loss, y_true, y_prob = predict_and_loss(model, val_loader, loss_fn, device)

        tuned_thresholds, threshold_info = tune_thresholds_for_f1(y_true, y_prob, step=args.threshold_step)
        metrics_runtime = compute_metrics(y_true, y_prob, RUNTIME_THRESHOLDS, prefix="runtime")
        metrics_tuned = compute_metrics(y_true, y_prob, tuned_thresholds, prefix="tuned")

        row: Dict[str, Any] = {
            "combo_id": combo_id,
            **asdict(config),
            "val_loss": float(val_loss),
            "best_epoch": int(train_info["best_epoch"]),
            "elapsed_sec": round(time.time() - start, 3),
            "pos_weight": json.dumps(train_info["pos_weight"]),
            "thresholds_runtime": json.dumps(RUNTIME_THRESHOLDS.tolist()),
            "thresholds_tuned": json.dumps([round(float(x), 4) for x in tuned_thresholds.tolist()]),
        }
        row.update(threshold_info)
        row.update(metrics_runtime)
        row.update(metrics_tuned)
        rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df.to_csv(out_dir / "grid_results_basic.csv", index=False)
        save_rankings(results_df, out_dir, args.top_n)
        save_plots(results_df, out_dir, args.top_n)

        print(
            f"val_loss={val_loss:.4f} | macro_f1_tuned={row['macro_f1_tuned']:.4f} | "
            f"brake_f1_tuned={row['brake_f1_tuned']:.4f} | thresholds={row['thresholds_tuned']}"
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "grid_results_basic.csv", index=False)
    save_rankings(results_df, out_dir, args.top_n)
    save_plots(results_df, out_dir, args.top_n)

    print(f"\nDone in {(time.time() - start_all) / 60:.2f} min")
    print("Results:", out_dir / "grid_results_basic.csv")
    print("Ranking:", out_dir / "top_by_metric.md")
    print("Plots:", out_dir)


if __name__ == "__main__":
    main()
