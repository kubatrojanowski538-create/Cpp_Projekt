import copy
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # zapis do plikow PNG bez otwierania okienek
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV_PATH = Path("ml data") / "GameStates1.csv"

MODEL_PATH = Path("models") / "driver_model.pt"
SCALER_PATH = Path("models") / "driver_scaler.joblib"
RESULTS_DIR = Path("original_results")

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
#  0 = sciana / przeszkoda
#  1 = trigger mety
RAY_TYPE_VALUES = [-1, 0, 1]


class GameStateDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class DriverNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = [col.strip() for col in df.columns]

    required_cols = NUMERIC_COLS + TYPE_COLS + LABEL_COLS

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn w CSV: {missing}")

    # Gdyby przez przypadek naglowek pojawil sie w srodku pliku,
    # to pd.to_numeric zamieni go na NaN i potem wiersz zostanie wyrzucony.
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Wyjscia jako 0/1.
    df[LABEL_COLS] = (df[LABEL_COLS] > 0.5).astype(np.float32)

    print("Liczba rekordow:", len(df))
    print()
    print("Procent wcisniec klawiszy:")
    print(df[LABEL_COLS].mean() * 100)
    print()
    print("Wartosci ray_type znalezione w danych:")
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


def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    exact_correct = 0
    per_button_correct = torch.zeros(4, device=device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_loss += loss.item() * len(x)
            total_samples += len(x)

            exact_correct += (preds == y).all(dim=1).sum().item()
            per_button_correct += (preds == y).float().sum(dim=0)

    avg_loss = total_loss / total_samples
    exact_acc = exact_correct / total_samples
    per_button_acc = per_button_correct / total_samples

    return avg_loss, exact_acc, per_button_acc.cpu().numpy()


def save_training_outputs(history, pos_weight, best_epoch, best_val_loss):
    """Zapisuje CSV, raport tekstowy i wykresy PNG do original_results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_csv_path = RESULTS_DIR / "train_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    # 1. Loss train/validation
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.axvline(best_epoch, linestyle="--", label=f"best epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=160)
    plt.close()

    # 2. Exact accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["exact_acc"] * 100.0, label="exact_acc")
    plt.axvline(best_epoch, linestyle="--", label=f"best epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Exact accuracy [%]")
    plt.title("Validation exact accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exact_accuracy_curve.png", dpi=160)
    plt.close()

    # 3. Per-button accuracy over epochs
    plt.figure(figsize=(10, 6))
    for name in LABEL_COLS:
        plt.plot(
            history_df["epoch"],
            history_df[f"{name}_acc"] * 100.0,
            label=name,
        )
    plt.axvline(best_epoch, linestyle="--", label=f"best epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Per-button accuracy [%]")
    plt.title("Validation per-button accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_button_accuracy_curve.png", dpi=160)
    plt.close()

    # 4. Final/best-epoch per-button accuracy as bar plot
    best_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]
    final_acc_values = [best_row[f"{name}_acc"] * 100.0 for name in LABEL_COLS]

    plt.figure(figsize=(9, 6))
    plt.bar(LABEL_COLS, final_acc_values)
    plt.ylabel("Accuracy [%]")
    plt.ylim(0, 100)
    plt.title("Per-button accuracy at best validation loss")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "best_epoch_per_button_accuracy.png", dpi=160)
    plt.close()

    # 5. pos_weight bar plot
    plt.figure(figsize=(9, 6))
    plt.bar(LABEL_COLS, pos_weight)
    plt.ylabel("pos_weight")
    plt.title("BCEWithLogitsLoss pos_weight per output")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pos_weight.png", dpi=160)
    plt.close()

    # 6. Combined summary figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0, 0].axvline(best_epoch, linestyle="--")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_df["epoch"], history_df["exact_acc"] * 100.0)
    axes[0, 1].axvline(best_epoch, linestyle="--")
    axes[0, 1].set_title("Exact accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy [%]")
    axes[0, 1].grid(True, alpha=0.3)

    for name in LABEL_COLS:
        axes[1, 0].plot(
            history_df["epoch"],
            history_df[f"{name}_acc"] * 100.0,
            label=name,
        )
    axes[1, 0].axvline(best_epoch, linestyle="--")
    axes[1, 0].set_title("Per-button accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy [%]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(LABEL_COLS, final_acc_values)
    axes[1, 1].set_title("Best epoch per-button accuracy")
    axes[1, 1].set_ylabel("Accuracy [%]")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Original training results", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_summary.png", dpi=160)
    plt.close(fig)

    report_path = RESULTS_DIR / "summary.md"
    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Original training results\n\n")
        file.write(f"Best epoch: **{best_epoch}**\n\n")
        file.write(f"Best validation loss: **{best_val_loss:.6f}**\n\n")
        file.write("## pos_weight\n\n")
        for name, weight in zip(LABEL_COLS, pos_weight):
            file.write(f"- `{name}`: `{weight:.6f}`\n")
        file.write("\n## Best epoch metrics\n\n")
        file.write(f"- `exact_acc`: `{best_row['exact_acc'] * 100.0:.2f}%`\n")
        for name in LABEL_COLS:
            file.write(f"- `{name}_acc`: `{best_row[f'{name}_acc'] * 100.0:.2f}%`\n")
        file.write("\n## Generated files\n\n")
        file.write("- `train_history.csv`\n")
        file.write("- `loss_curve.png`\n")
        file.write("- `exact_accuracy_curve.png`\n")
        file.write("- `per_button_accuracy_curve.png`\n")
        file.write("- `best_epoch_per_button_accuracy.png`\n")
        file.write("- `pos_weight.png`\n")
        file.write("- `training_summary.png`\n")

    print()
    print("Zapisano wyniki treningu do folderu:", RESULTS_DIR)
    print("Historia treningu:", history_csv_path)
    print("Raport:", report_path)


def train():
    torch.manual_seed(123)
    np.random.seed(123)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=123,
        shuffle=True
    )

    x_train, scaler = build_features(train_df, fit_scaler=True)
    x_val, _ = build_features(val_df, scaler=scaler, fit_scaler=False)

    y_train = train_df[LABEL_COLS].astype(np.float32).to_numpy()
    y_val = val_df[LABEL_COLS].astype(np.float32).to_numpy()

    train_dataset = GameStateDataset(x_train, y_train)
    val_dataset = GameStateDataset(x_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print()
    print("Device:", device)

    model = DriverNet(input_size=x_train.shape[1]).to(device)

    # Wazne przy nierownych klasach.
    # Np. brake moze wystepowac bardzo rzadko, wiec bez tego siec moze go ignorowac.
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives

    pos_weight = negatives / np.maximum(positives, 1.0)
    pos_weight = np.clip(pos_weight, 0.5, 20.0)

    print()
    print("pos_weight:")
    for name, weight in zip(LABEL_COLS, pos_weight):
        print(f"{name}: {weight:.3f}")

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    epochs = 80
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(x)
            train_samples += len(x)

        train_loss = train_loss_sum / train_samples

        val_loss, exact_acc, per_button_acc = evaluate(
            model,
            val_loader,
            loss_fn,
            device
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "exact_acc": exact_acc,
        }
        for name, acc in zip(LABEL_COLS, per_button_acc):
            row[f"{name}_acc"] = acc
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"exact_acc={exact_acc * 100:.2f}%"
            )

            for name, acc in zip(LABEL_COLS, per_button_acc):
                print(f"  {name}: {acc * 100:.2f}%")

    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": x_train.shape[1],
            "label_cols": LABEL_COLS,
            "numeric_cols": NUMERIC_COLS,
            "type_cols": TYPE_COLS,
            "ray_type_values": RAY_TYPE_VALUES,
        },
        MODEL_PATH
    )

    joblib.dump(scaler, SCALER_PATH)

    save_training_outputs(
        history=history,
        pos_weight=pos_weight,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )

    print()
    print("Zapisano model:", MODEL_PATH)
    print("Zapisano scaler:", SCALER_PATH)


if __name__ == "__main__":
    train()
