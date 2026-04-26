import copy
import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV_PATH = ".\ml data\GameStates1.csv"

MODEL_PATH = ".\models\driver_model.pt"
SCALER_PATH = ".\models\driver_scaler.joblib"

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


def train():
    torch.manual_seed(123)
    np.random.seed(123)

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

    # Ważne przy nierównych klasach.
    # Np. brake może występować bardzo rzadko, więc bez tego sieć może go ignorować.
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

    epochs = 80

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

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

    print()
    print("Zapisano model:", MODEL_PATH)
    print("Zapisano scaler:", SCALER_PATH)


if __name__ == "__main__":
    train()