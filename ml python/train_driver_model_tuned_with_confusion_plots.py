import copy
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # zapis wykresów PNG bez otwierania okienek
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV_PATH = r".\ml data\GameStates1.csv"

MODEL_PATH = r"..\models\driver_model.pt"
SCALER_PATH = r"..\models\driver_scaler.joblib"

RESULTS_DIR = "tuned_results"

RAY_COUNT = 20

LABEL_COLS = [
    "accelerate",
    "brake",
    "steer_left",
    "steer_right",
]

DIST_COLS = [f"ray_dist_{i}" for i in range(RAY_COUNT)]
TYPE_COLS = [f"ray_type_{i}" for i in range(RAY_COUNT)]

NUMERIC_COLS = ["car_speed"] + DIST_COLS

# Z Twojego kodu wynika:
# -1 = nic nie trafiono
#  0 = ściana / przeszkoda
#  1 = trigger mety
RAY_TYPE_VALUES = [-1, 0, 1]

# Najlepsza konfiguracja z exact searcha.
# Pola loss/score/F1 są zapisane informacyjnie i nie sterują treningiem.
BEST_SEARCH_RESULT = {
    "model_type": "binary",
    "hidden_layers": "256,256,128,64",
    "activation": "relu",
    "norm": "batch",
    "dropout": 0.0,
    "lr": 0.00115,
    "weight_decay": 0.0014,
    "batch_size": 2048.0,
    "imbalance_cap": 12.0,
    "loss_variant": "focal",
    "focal_gamma": 2.0,
    "epoch": 70.0,
    "train_loss": 0.03514011519452993,
    "val_loss": 0.09024114906787872,
    "score": 1.0424567666649818,
    "macro_f1_runtime": 0.8095278739929199,
    "exact_acc_runtime": 0.7576778531074524,
    "brake_f1_runtime": 0.7745664715766907,
    "macro_f1_tuned": 0.8366215229034424,
    "exact_acc_tuned": 0.7979627251625061,
    "brake_f1_tuned": 0.8387096524238586,
    "thresholds": [0.57, 0.59, 0.575, 0.465],
}

MODEL_TYPE = BEST_SEARCH_RESULT["model_type"]
HIDDEN_LAYERS = [int(size) for size in BEST_SEARCH_RESULT["hidden_layers"].split(",")]
ACTIVATION = BEST_SEARCH_RESULT["activation"]
NORM = BEST_SEARCH_RESULT["norm"]
DROPOUT = float(BEST_SEARCH_RESULT["dropout"])
LR = float(BEST_SEARCH_RESULT["lr"])
WEIGHT_DECAY = float(BEST_SEARCH_RESULT["weight_decay"])
BATCH_SIZE = int(BEST_SEARCH_RESULT["batch_size"])
IMBALANCE_CAP = float(BEST_SEARCH_RESULT["imbalance_cap"])
LOSS_VARIANT = BEST_SEARCH_RESULT["loss_variant"]
FOCAL_GAMMA = float(BEST_SEARCH_RESULT["focal_gamma"])
EPOCHS = int(BEST_SEARCH_RESULT["epoch"])
TUNED_THRESHOLDS = np.array(BEST_SEARCH_RESULT["thresholds"], dtype=np.float32)


class GameStateDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        loss = focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_activation(name):
    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    raise ValueError(f"Nieobsługiwana aktywacja: {name}")


class DriverNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers=None,
        activation=ACTIVATION,
        norm=NORM,
        dropout=DROPOUT,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = HIDDEN_LAYERS

        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))

            if norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_size))
            elif norm not in (None, "none"):
                raise ValueError(f"Nieobsługiwana normalizacja: {norm}")

            layers.append(make_activation(activation))

            # Dropout=0.0 oznacza brak losowego wyłączania neuronów.
            # Warunek zostawia architekturę zgodną z hiperparametrem bez dodawania pustej warstwy.
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            previous_size = hidden_size

        # model_type="binary": cztery niezależne logity dla czterech klawiszy.
        layers.append(nn.Linear(previous_size, len(LABEL_COLS)))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = [col.strip() for col in df.columns]

    required_cols = NUMERIC_COLS + TYPE_COLS + LABEL_COLS

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn w CSV: {missing}")

    # Gdyby przez przypadek nagłówek pojawił się w środku pliku,
    # to pd.to_numeric zamieni go na NaN i potem wiersz zostanie wyrzucony.
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Wyjścia jako 0/1.
    df[LABEL_COLS] = (df[LABEL_COLS] > 0.5).astype(np.float32)

    print("Liczba rekordów:", len(df))
    print()
    print("Procent wciśnięć klawiszy:")
    print(df[LABEL_COLS].mean() * 100)
    print()
    print("Wartości ray_type znalezione w danych:")
    print(sorted(pd.unique(df[TYPE_COLS].values.ravel())))

    return df


def build_features(df, scaler=None, fit_scaler=False):
    numeric = df[NUMERIC_COLS].astype(np.float32).to_numpy()

    if fit_scaler:
        scaler = StandardScaler()
        numeric = scaler.fit_transform(numeric).astype(np.float32)
    else:
        numeric = scaler.transform(numeric).astype(np.float32)

    type_features = []

    for col in TYPE_COLS:
        values = df[col].astype(int).to_numpy()

        for ray_type in RAY_TYPE_VALUES:
            one_hot = (values == ray_type).astype(np.float32).reshape(-1, 1)
            type_features.append(one_hot)

    x = np.concatenate([numeric] + type_features, axis=1).astype(np.float32)

    return x, scaler


def evaluate(model, loader, loss_fn, device, thresholds):
    model.eval()

    threshold_tensor = torch.tensor(
        thresholds,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    total_loss = 0.0
    total_samples = 0
    exact_correct = 0
    per_button_correct = torch.zeros(len(LABEL_COLS), device=device)
    true_positive = torch.zeros(len(LABEL_COLS), device=device)
    false_positive = torch.zeros(len(LABEL_COLS), device=device)
    true_negative = torch.zeros(len(LABEL_COLS), device=device)
    false_negative = torch.zeros(len(LABEL_COLS), device=device)

    non_blocking = device == "cuda"

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)

            logits = model(x)
            loss = loss_fn(logits, y)

            probs = torch.sigmoid(logits)
            preds = probs > threshold_tensor
            targets = y > 0.5

            total_loss += loss.item() * len(x)
            total_samples += len(x)

            exact_correct += (preds == targets).all(dim=1).sum().item()
            per_button_correct += (preds == targets).float().sum(dim=0)

            true_positive += (preds & targets).sum(dim=0).float()
            false_positive += (preds & ~targets).sum(dim=0).float()
            true_negative += (~preds & ~targets).sum(dim=0).float()
            false_negative += (~preds & targets).sum(dim=0).float()

    avg_loss = total_loss / total_samples
    exact_acc = exact_correct / total_samples
    per_button_acc = per_button_correct / total_samples

    precision_per_button = true_positive / torch.clamp(
        true_positive + false_positive,
        min=1e-8,
    )
    f1_per_button = (2.0 * true_positive) / torch.clamp(
        2.0 * true_positive + false_positive + false_negative,
        min=1e-8,
    )
    macro_f1 = f1_per_button.mean().item()

    brake_index = LABEL_COLS.index("brake")
    brake_f1 = f1_per_button[brake_index].item()
    brake_precision = precision_per_button[brake_index].item()

    confusion_counts = {
        "tp": true_positive.cpu().numpy().astype(np.int64),
        "fp": false_positive.cpu().numpy().astype(np.int64),
        "tn": true_negative.cpu().numpy().astype(np.int64),
        "fn": false_negative.cpu().numpy().astype(np.int64),
    }

    return (
        avg_loss,
        exact_acc,
        per_button_acc.cpu().numpy(),
        macro_f1,
        brake_f1,
        brake_precision,
        f1_per_button.cpu().numpy(),
        precision_per_button.cpu().numpy(),
        confusion_counts,
    )


def make_loss(pos_weight, device):
    pos_weight_tensor = torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device,
    )

    if LOSS_VARIANT == "focal":
        return BinaryFocalWithLogitsLoss(
            gamma=FOCAL_GAMMA,
            pos_weight=pos_weight_tensor,
            reduction="mean",
        )

    if LOSS_VARIANT == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    raise ValueError(f"Nieobsługiwany wariant funkcji straty: {LOSS_VARIANT}")


def copy_state_to_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def print_config():
    print()
    print("Konfiguracja treningu:")
    print(f"  model_type: {MODEL_TYPE}")
    print(f"  hidden_layers: {HIDDEN_LAYERS}")
    print(f"  activation: {ACTIVATION}")
    print(f"  norm: {NORM}")
    print(f"  dropout: {DROPOUT}")
    print(f"  lr: {LR}")
    print(f"  weight_decay: {WEIGHT_DECAY}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  imbalance_cap: {IMBALANCE_CAP}")
    print(f"  loss_variant: {LOSS_VARIANT}")
    print(f"  focal_gamma: {FOCAL_GAMMA}")
    print(f"  epochs: {EPOCHS}")
    print(f"  thresholds: {TUNED_THRESHOLDS.tolist()}")


def confusion_matrix_from_row(row, label_name):
    """Zwraca macierz [[TN, FP], [FN, TP]] dla jednej akcji."""
    tn = int(round(float(row[f"{label_name}_tn"])))
    fp = int(round(float(row[f"{label_name}_fp"])))
    fn = int(round(float(row[f"{label_name}_fn"])))
    tp = int(round(float(row[f"{label_name}_tp"])))

    return np.array(
        [
            [tn, fp],
            [fn, tp],
        ],
        dtype=np.int64,
    )


def draw_confusion_matrix_table(ax, matrix, title):
    """Rysuje czytelną macierz pomyłek bez zależności od seaborn/sklearn."""
    ax.axis("off")

    cell_text = [
        [f"TN\n{matrix[0, 0]}", f"FP\n{matrix[0, 1]}"],
        [f"FN\n{matrix[1, 0]}", f"TP\n{matrix[1, 1]}"],
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=["actual 0", "actual 1"],
        colLabels=["predicted 0", "predicted 1"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.25, 2.2)

    ax.set_title(title)


def save_tuned_results(history, best_epoch):
    """Zapisuje historię, wykresy i macierze pomyłek do tuned_results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(RESULTS_DIR, "tuned_train_history.csv")
    history_df.to_csv(history_csv_path, index=False)

    generated_paths = [history_csv_path]

    plt.figure(figsize=(10, 6))
    plt.plot(
        history_df["epoch"],
        history_df["brake_f1_tuned"],
        label="brake F1",
    )
    plt.plot(
        history_df["epoch"],
        history_df["brake_precision_tuned"],
        label="brake precision",
    )

    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", label=f"best epoch = {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.ylim(0.0, 1.05)
    plt.title("Brake F1 and precision on validation set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    brake_plot_path = os.path.join(RESULTS_DIR, "brake_f1_precision_curve.png")
    plt.savefig(brake_plot_path, dpi=160)
    plt.close()
    generated_paths.append(brake_plot_path)

    plt.figure(figsize=(10, 6))
    for name in LABEL_COLS:
        plt.plot(
            history_df["epoch"],
            history_df[f"{name}_f1_tuned"],
            label=f"{name} F1",
        )

    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", label=f"best epoch = {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.05)
    plt.title("F1 for each action on validation set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    all_actions_f1_plot_path = os.path.join(RESULTS_DIR, "all_actions_f1_curve.png")
    plt.savefig(all_actions_f1_plot_path, dpi=160)
    plt.close()
    generated_paths.append(all_actions_f1_plot_path)

    plt.figure(figsize=(10, 6))
    plt.plot(
        history_df["epoch"],
        history_df["exact_acc_tuned"],
        label="exact accuracy",
    )

    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", label=f"best epoch = {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Exact accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Exact accuracy on validation set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    exact_accuracy_plot_path = os.path.join(RESULTS_DIR, "exact_accuracy_curve.png")
    plt.savefig(exact_accuracy_plot_path, dpi=160)
    plt.close()
    generated_paths.append(exact_accuracy_plot_path)

    if best_epoch is not None:
        best_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]
    else:
        best_row = history_df.iloc[-1]

    confusion_rows = []
    for name in LABEL_COLS:
        matrix = confusion_matrix_from_row(best_row, name)
        confusion_rows.append(
            {
                "action": name,
                "TN": int(matrix[0, 0]),
                "FP": int(matrix[0, 1]),
                "FN": int(matrix[1, 0]),
                "TP": int(matrix[1, 1]),
            }
        )

    confusion_df = pd.DataFrame(confusion_rows)
    confusion_csv_path = os.path.join(RESULTS_DIR, "confusion_matrices_best_epoch.csv")
    confusion_df.to_csv(confusion_csv_path, index=False)
    generated_paths.append(confusion_csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes.ravel(), LABEL_COLS):
        matrix = confusion_matrix_from_row(best_row, name)
        draw_confusion_matrix_table(ax, matrix, f"{name}")

    fig.suptitle(
        f"Confusion matrices on validation set (best epoch = {int(best_row['epoch'])})",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    combined_confusion_plot_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrices_best_epoch.png",
    )
    fig.savefig(combined_confusion_plot_path, dpi=160)
    plt.close(fig)
    generated_paths.append(combined_confusion_plot_path)

    for name in LABEL_COLS:
        matrix = confusion_matrix_from_row(best_row, name)
        fig, ax = plt.subplots(figsize=(6, 4))
        draw_confusion_matrix_table(
            ax,
            matrix,
            f"{name} confusion matrix\nbest epoch = {int(best_row['epoch'])}",
        )
        fig.tight_layout()

        single_confusion_plot_path = os.path.join(
            RESULTS_DIR,
            f"confusion_matrix_{name}_best_epoch.png",
        )
        fig.savefig(single_confusion_plot_path, dpi=160)
        plt.close(fig)
        generated_paths.append(single_confusion_plot_path)

    report_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# Tuned training results\n\n")
        file.write(f"Best epoch: **{int(best_row['epoch'])}**\n\n")
        file.write("## Main metrics at best epoch\n\n")
        file.write(f"- `exact_acc_tuned`: `{best_row['exact_acc_tuned']:.6f}`\n")
        file.write(f"- `macro_f1_tuned`: `{best_row['macro_f1_tuned']:.6f}`\n")
        file.write(f"- `brake_f1_tuned`: `{best_row['brake_f1_tuned']:.6f}`\n")
        file.write(f"- `brake_precision_tuned`: `{best_row['brake_precision_tuned']:.6f}`\n")
        file.write("\n## F1 per action at best epoch\n\n")
        for name in LABEL_COLS:
            file.write(f"- `{name}_f1_tuned`: `{best_row[f'{name}_f1_tuned']:.6f}`\n")
        file.write("\n## Confusion matrices at best epoch\n\n")
        file.write("| action | TN | FP | FN | TP |\n")
        file.write("|---|---:|---:|---:|---:|\n")
        for row in confusion_rows:
            file.write(
                f"| {row['action']} | {row['TN']} | {row['FP']} | "
                f"{row['FN']} | {row['TP']} |\n"
            )
        file.write("\n## Generated files\n\n")
        for path in generated_paths:
            file.write(f"- `{os.path.basename(path)}`\n")

    print()
    print("Zapisano wyniki tuned do folderu:", RESULTS_DIR)
    print("Zapisane pliki:")
    for path in generated_paths:
        print(" ", path)
    print("Raport:", report_path)


def train():
    torch.manual_seed(67)
    np.random.seed(67)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"

    if device == "cuda":
        torch.cuda.manual_seed_all(67)
        torch.backends.cudnn.benchmark = True

    print_config()

    df = load_data()

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=67,
        shuffle=True,
    )

    x_train, scaler = build_features(train_df, fit_scaler=True)
    x_val, _ = build_features(val_df, scaler=scaler, fit_scaler=False)

    y_train = train_df[LABEL_COLS].astype(np.float32).to_numpy()
    y_val = val_df[LABEL_COLS].astype(np.float32).to_numpy()

    train_dataset = GameStateDataset(x_train, y_train)
    val_dataset = GameStateDataset(x_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )

    print()
    print("Device:", device)

    model = DriverNet(input_size=x_train.shape[1]).to(device)

    # Ważne przy nierównych klasach.
    # Np. brake może występować rzadko, więc bez tego sieć może go ignorować.
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives

    pos_weight = negatives / np.maximum(positives, 1.0)
    pos_weight = np.clip(pos_weight, 0.5, IMBALANCE_CAP)

    print()
    print("pos_weight:")
    for name, weight in zip(LABEL_COLS, pos_weight):
        print(f"{name}: {weight:.3f}")

    loss_fn = make_loss(pos_weight, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro_f1 = -1.0
    best_val_loss_for_best_f1 = float("inf")
    best_epoch = None
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_samples = 0
        non_blocking = device == "cuda"

        for x, y in train_loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(x)
            train_samples += len(x)

        train_loss = train_loss_sum / train_samples

        (
            val_loss,
            exact_acc,
            per_button_acc,
            macro_f1,
            brake_f1,
            brake_precision,
            f1_per_button,
            precision_per_button,
            confusion_counts,
        ) = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            thresholds=TUNED_THRESHOLDS,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "exact_acc_tuned": exact_acc,
            "macro_f1_tuned": macro_f1,
            "brake_f1_tuned": brake_f1,
            "brake_precision_tuned": brake_precision,
        }
        for index, (name, acc, f1, precision) in enumerate(
            zip(
                LABEL_COLS,
                per_button_acc,
                f1_per_button,
                precision_per_button,
            )
        ):
            row[f"{name}_acc"] = acc
            row[f"{name}_f1_tuned"] = f1
            row[f"{name}_precision_tuned"] = precision
            row[f"{name}_tp"] = int(confusion_counts["tp"][index])
            row[f"{name}_fp"] = int(confusion_counts["fp"][index])
            row[f"{name}_tn"] = int(confusion_counts["tn"][index])
            row[f"{name}_fn"] = int(confusion_counts["fn"][index])
        history.append(row)

        # Konfiguracja była wybierana pod metryki F1, dlatego zapisujemy checkpoint
        # z najlepszym macro_f1 liczonym na progach z exact searcha.
        if (
            macro_f1 > best_macro_f1
            or (
                np.isclose(macro_f1, best_macro_f1)
                and val_loss < best_val_loss_for_best_f1
            )
        ):
            best_macro_f1 = macro_f1
            best_val_loss_for_best_f1 = val_loss
            best_epoch = epoch
            best_state = copy_state_to_cpu(model)

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"exact_acc_tuned={exact_acc * 100:.2f}% | "
                f"macro_f1_tuned={macro_f1:.4f} | "
                f"brake_f1_tuned={brake_f1:.4f} | "
                f"brake_precision_tuned={brake_precision:.4f}"
            )

            for name, acc, f1, precision in zip(
                LABEL_COLS,
                per_button_acc,
                f1_per_button,
                precision_per_button,
            ):
                print(
                    f"  {name}: acc={acc * 100:.2f}% | "
                    f"f1={f1:.4f} | precision={precision:.4f}"
                )

    if best_state is None:
        raise RuntimeError("Nie udało się wybrać najlepszego stanu modelu.")

    model.load_state_dict(best_state)

    ensure_parent_dir(MODEL_PATH)
    ensure_parent_dir(SCALER_PATH)

    torch.save(
        {
            "model_state": copy_state_to_cpu(model),
            "input_size": x_train.shape[1],
            "label_cols": LABEL_COLS,
            "numeric_cols": NUMERIC_COLS,
            "type_cols": TYPE_COLS,
            "ray_type_values": RAY_TYPE_VALUES,
            "model_type": MODEL_TYPE,
            "hidden_layers": HIDDEN_LAYERS,
            "activation": ACTIVATION,
            "norm": NORM,
            "dropout": DROPOUT,
            "thresholds": TUNED_THRESHOLDS.tolist(),
            "hyperparameters": {
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "imbalance_cap": IMBALANCE_CAP,
                "loss_variant": LOSS_VARIANT,
                "focal_gamma": FOCAL_GAMMA,
                "epochs": EPOCHS,
            },
            "best_epoch": best_epoch,
            "best_macro_f1_tuned": best_macro_f1,
            "best_val_loss_for_best_f1": best_val_loss_for_best_f1,
            "source_search_result": BEST_SEARCH_RESULT,
        },
        MODEL_PATH,
    )

    joblib.dump(scaler, SCALER_PATH)

    save_tuned_results(history, best_epoch)

    print()
    print(f"Najlepsza epoka według macro_f1_tuned: {best_epoch}")
    print(f"best_macro_f1_tuned: {best_macro_f1:.4f}")
    print(f"val_loss dla tego checkpointu: {best_val_loss_for_best_f1:.4f}")
    print()
    print("Zapisano model:", MODEL_PATH)
    print("Zapisano scaler:", SCALER_PATH)


if __name__ == "__main__":
    train()
