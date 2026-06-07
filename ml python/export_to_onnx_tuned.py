import json
from pathlib import Path

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "../" / "models" / "driver_model.pt"
ONNX_PATH = SCRIPT_DIR / "driver_model_tuned.onnx"
THRESHOLDS_PATH = SCRIPT_DIR / "driver_model_tuned_thresholds.json"

# Domyślne wartości zgodne z najlepszą konfiguracją z searcha.
DEFAULT_HIDDEN_LAYERS = [256, 256, 128, 64]
DEFAULT_ACTIVATION = "relu"
DEFAULT_NORM = "batch"
DEFAULT_DROPOUT = 0.0
DEFAULT_LABEL_COLS = [
    "accelerate",
    "brake",
    "steer_left",
    "steer_right",
]
DEFAULT_THRESHOLDS = [0.57, 0.59, 0.575, 0.465]


def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono checkpointu: {path}\n"
            "Najpierw uruchom trenowanie tuned modelu albo popraw MODEL_PATH."
        )

    # weights_only istnieje w nowszych wersjach PyTorch. Starsze wersje go nie znają.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]

    # Awaryjnie obsłuży także plik będący bezpośrednio state_dictiem.
    if isinstance(checkpoint, dict) and checkpoint:
        first_value = next(iter(checkpoint.values()))
        if torch.is_tensor(first_value):
            return checkpoint

    raise ValueError("Checkpoint nie zawiera pola 'model_state' ani bezpośredniego state_dict.")


def strip_module_prefix(state_dict):
    # Przydatne, gdyby model był kiedyś zapisany przez DataParallel.
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict

    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def infer_input_size(state_dict):
    first_linear_weight = state_dict.get("net.0.weight")
    if first_linear_weight is None:
        raise ValueError("Nie mogę odczytać input_size z checkpointu: brakuje 'net.0.weight'.")
    return int(first_linear_weight.shape[1])


def parse_hidden_layers(value):
    if value is None:
        return DEFAULT_HIDDEN_LAYERS

    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]

    return [int(size) for size in value]


def make_activation(name):
    name = str(name).lower()

    if name == "relu":
        return nn.ReLU()

    raise ValueError(f"Nieobsługiwana aktywacja w eksporcie ONNX: {name}")


class DriverNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers=None,
        activation: str = DEFAULT_ACTIVATION,
        norm: str = DEFAULT_NORM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        hidden_layers = parse_hidden_layers(hidden_layers)
        norm = None if norm is None else str(norm).lower()
        dropout = float(dropout)

        layers = []
        previous_size = int(input_size)

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))

            if norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_size))
            elif norm in (None, "none", ""):
                pass
            else:
                raise ValueError(f"Nieobsługiwana normalizacja w eksporcie ONNX: {norm}")

            layers.append(make_activation(activation))

            # Dla tuned modelu dropout=0.0, więc warstwa Dropout NIE jest dodawana.
            # To musi się zgadzać z kluczami w checkpointcie.
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, int(output_size)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    checkpoint = load_checkpoint(MODEL_PATH)
    state_dict = strip_module_prefix(get_model_state(checkpoint))

    input_size = int(checkpoint.get("input_size", infer_input_size(state_dict)))
    label_cols = checkpoint.get("label_cols", DEFAULT_LABEL_COLS)
    output_size = len(label_cols)

    hidden_layers = parse_hidden_layers(checkpoint.get("hidden_layers", DEFAULT_HIDDEN_LAYERS))
    activation = checkpoint.get("activation", DEFAULT_ACTIVATION)
    norm = checkpoint.get("norm", DEFAULT_NORM)
    dropout = float(checkpoint.get("dropout", DEFAULT_DROPOUT))
    thresholds = checkpoint.get("thresholds", DEFAULT_THRESHOLDS)

    model = DriverNet(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation=activation,
        norm=norm,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, input_size, dtype=torch.float32)

    # Eksportuje surowe logity, tak jak poprzedni export_to_onnx.py.
    # W C++ zastosuj sigmoid(logit), a potem porównaj z thresholdami.
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    export_info = {
        "onnx_output": "logits",
        "postprocessing": "probabilities = sigmoid(logits); actions = probabilities > thresholds",
        "input_size": input_size,
        "label_cols": label_cols,
        "thresholds": [float(value) for value in thresholds],
        "hidden_layers": hidden_layers,
        "activation": activation,
        "norm": norm,
        "dropout": dropout,
    }

    with THRESHOLDS_PATH.open("w", encoding="utf-8") as file:
        json.dump(export_info, file, ensure_ascii=False, indent=2)

    print("Zapisano ONNX:", ONNX_PATH)
    print("Zapisano progi/metadane:", THRESHOLDS_PATH)
    print("Input size:", input_size)
    print("Wyjścia:", label_cols)
    print("Thresholds:", export_info["thresholds"])
    print("Uwaga: ONNX zwraca logity; przed progami użyj sigmoid().")


if __name__ == "__main__":
    main()
