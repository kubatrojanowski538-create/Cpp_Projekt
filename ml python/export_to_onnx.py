import torch
from pathlib import Path

from train_driver_model import DriverNet, MODEL_PATH


SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_PATH = SCRIPT_DIR / "driver_model.onnx"


checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model = DriverNet(input_size=checkpoint["input_size"])
model.load_state_dict(checkpoint["model_state"])
model.eval()

dummy_input = torch.randn(1, checkpoint["input_size"], dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=17
)

print("Zapisano:", ONNX_PATH)
print("Input size:", checkpoint["input_size"])