import joblib
import torch
import pandas as pd

from train_driver_model import (
    DriverNet,
    build_features,
    LABEL_COLS,
    MODEL_PATH,
    SCALER_PATH
)


checkpoint = torch.load(MODEL_PATH, map_location="cpu")
scaler = joblib.load(SCALER_PATH)

model = DriverNet(input_size=checkpoint["input_size"])
model.load_state_dict(checkpoint["model_state"])
model.eval()


state = {
    "car_speed": 5.0,
}

for i in range(20):
    state[f"ray_dist_{i}"] = 500.0
    state[f"ray_type_{i}"] = -1

# Przykładowa ściana blisko z przodu/lewej strony.
state["ray_dist_7"] = 90.0
state["ray_type_7"] = 0

state["ray_dist_8"] = 70.0
state["ray_type_8"] = 0

state["ray_dist_9"] = 110.0
state["ray_type_9"] = 0

df = pd.DataFrame([state])

x, _ = build_features(df, scaler=scaler, fit_scaler=False)
x = torch.tensor(x, dtype=torch.float32)

with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits).numpy()[0]

print("Prawdopodobieństwa:")
for name, prob in zip(LABEL_COLS, probs):
    print(f"{name}: {prob:.3f}")

controls = {
    name: bool(prob > 0.5)
    for name, prob in zip(LABEL_COLS, probs)
}

print()
print("Decyzja:")
print(controls)