import joblib
import torch
import pandas as pd

from train_driver_model import (
    DriverNet,
    build_features,
    LABEL_COLS,
    MODEL_PATH,
    SCALER_PATH,
    CSV_PATH
)


def probs_to_controls(probs):
    p_accelerate = probs[LABEL_COLS.index("accelerate")]
    p_brake = probs[LABEL_COLS.index("brake")]
    p_left = probs[LABEL_COLS.index("steer_left")]
    p_right = probs[LABEL_COLS.index("steer_right")]

    controls = {
        "accelerate": False,
        "brake": False,
        "steer_left": False,
        "steer_right": False,
    }

    if p_brake > 0.50:
        controls["brake"] = True
    elif p_accelerate > 0.30:
        controls["accelerate"] = True

    if p_left > 0.35 or p_right > 0.35:
        if p_left > p_right:
            controls["steer_left"] = True
        else:
            controls["steer_right"] = True

    return controls


checkpoint = torch.load(MODEL_PATH, map_location="cpu")
scaler = joblib.load(SCALER_PATH)

model = DriverNet(input_size=checkpoint["input_size"])
model.load_state_dict(checkpoint["model_state"])
model.eval()

df = pd.read_csv(CSV_PATH)
df.columns = [col.strip() for col in df.columns]

# Losowy prawdziwy stan z danych
sample = df.sample(1).copy()

x, _ = build_features(sample, scaler=scaler, fit_scaler=False)
x = torch.tensor(x, dtype=torch.float32)

with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits).numpy()[0]

print("Prawdziwe inputy gracza z CSV:")
for name in LABEL_COLS:
    print(f"{name}: {int(sample[name].iloc[0])}")

print()
print("Prawdopodobieństwa modelu:")
for name, prob in zip(LABEL_COLS, probs):
    print(f"{name}: {prob:.3f}")

print()
print("Decyzja modelu:")
print(probs_to_controls(probs))