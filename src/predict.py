from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf


project_folder = Path(__file__).resolve().parent.parent
model_folder = project_folder / "model"
model_path = model_folder / "plant_classifier.keras"
names_path = model_folder / "class_names.json"

image_path = Path(input("Paste the path to a plant image: ").strip().strip('"'))

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

if not model_path.is_file():
    raise FileNotFoundError("Run src/train.py first to create the trained model.")

model = tf.keras.models.load_model(model_path)
with open(names_path, "r", encoding="utf-8") as file:
    class_names = json.load(file)

image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"Could not open image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 128))
image = np.array(image, dtype="float32") / 255.0
image = np.expand_dims(image, axis=0)

scores = model.predict(image, verbose=0)[0]
best_number = int(np.argmax(scores))
category, subcategory = class_names[best_number].split("/", 1)

print("\nCategory:", category)
print("Subcategory:", subcategory)
print("Confidence:", round(float(scores[best_number]) * 100, 2), "%")

