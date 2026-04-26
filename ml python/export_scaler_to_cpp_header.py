from pathlib import Path
import joblib

from train_driver_model import SCALER_PATH, NUMERIC_COLS


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "AIScalerData.h"

scaler = joblib.load(SCALER_PATH)

means = scaler.mean_.tolist()
scales = scaler.scale_.tolist()

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write("#pragma once\n\n")
    file.write("// Ten plik jest wygenerowany automatycznie przez export_scaler_to_cpp_header.py\n")
    file.write("// Kolejnosc: car_speed, ray_dist_0, ..., ray_dist_19\n\n")

    file.write("constexpr int AI_NUMERIC_INPUT_COUNT = 21;\n\n")

    file.write("static const float AI_INPUT_MEAN[AI_NUMERIC_INPUT_COUNT] = {\n")
    for value in means:
        file.write(f"    {value:.9f}f,\n")
    file.write("};\n\n")

    file.write("static const float AI_INPUT_SCALE[AI_NUMERIC_INPUT_COUNT] = {\n")
    for value in scales:
        file.write(f"    {value:.9f}f,\n")
    file.write("};\n\n")

    file.write("// Kolumny numeryczne:\n")
    for i, name in enumerate(NUMERIC_COLS):
        file.write(f"// {i}: {name}\n")

print("Zapisano:", OUTPUT_PATH)