import json
import joblib
from pathlib import Path

from train_driver_model import SCALER_PATH, NUMERIC_COLS


SCRIPT_DIR = Path(__file__).resolve().parent
JSON_PATH = ".\ml python\driver_scaler.json"

scaler = joblib.load(SCALER_PATH)

data = {
    "numeric_cols": NUMERIC_COLS,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}

with open(JSON_PATH, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)

print("Zapisano:", JSON_PATH)