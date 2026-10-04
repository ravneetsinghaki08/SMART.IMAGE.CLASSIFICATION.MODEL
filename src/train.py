from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Project folders
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "plants"
model_folder = project_folder / "model"
results_folder = project_folder / "results"

image_size = 128
epochs = 10
image_types = {".jpg", ".jpeg", ".png"}

if not data_folder.is_dir():
    raise FileNotFoundError(f"Plant images folder not found: {data_folder}")

# Read images from data/plants/category/subcategory/
images = []
labels = []
image_paths = []

for category_folder in sorted(data_folder.iterdir()):
    if not category_folder.is_dir():
        continue

    for subcategory_folder in sorted(category_folder.iterdir()):
        if not subcategory_folder.is_dir():
            continue

        for image_path in sorted(subcategory_folder.iterdir()):
            if image_path.suffix.lower() not in image_types:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print("Skipping unreadable image:", image_path)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_size, image_size))

            images.append(image)
            labels.append(category_folder.name + "/" + subcategory_folder.name)
            image_paths.append(image_path)

if not images:
    raise ValueError(f"No JPG, JPEG, or PNG images found under {data_folder}")

# Prepare pixel arrays and convert folder labels to numbers
images = np.array(images, dtype="float32") / 255.0
class_names = sorted(set(labels))
name_to_number = {name: number for number, name in enumerate(class_names)}
label_numbers = np.array([name_to_number[name] for name in labels])

if len(class_names) < 2:
    raise ValueError("Images are needed in at least two subcategory folders.")

class_counts = np.bincount(label_numbers)
if np.any(class_counts < 2):
    raise ValueError("Each subcategory needs at least two images for the train/test split.")

# Keep some images separate so they are not used to teach the model
test_size = max(3, len(class_names), int(np.ceil(len(images) * 0.20)))
if len(images) - test_size < len(class_names):
    raise ValueError("Add more images so each subcategory can be in both sets.")

train_images, test_images, train_labels, test_labels, train_paths, test_paths = train_test_split(
    images,
    label_numbers,
    image_paths,
    test_size=test_size,
    random_state=42,
    stratify=label_numbers
)

# Build and train one small image classifier
model = tf.keras.Sequential([
    tf.keras.Input(shape=(image_size, image_size, 3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=epochs)

# Evaluate using only images held back from training
scores = model.predict(test_images, verbose=0)
predicted_labels = np.argmax(scores, axis=1)
accuracy = accuracy_score(test_labels, predicted_labels)
matrix = confusion_matrix(
    test_labels,
    predicted_labels,
    labels=np.arange(len(class_names))
)

print("\nTest accuracy:", round(accuracy * 100, 2), "%")
print("Confusion matrix label order:", class_names)
print(matrix)

print("\nThree held-out image predictions:")
for path, actual, predicted in zip(test_paths[:3], test_labels[:3], predicted_labels[:3]):
    print(path.name, "| actual:", class_names[actual], "| predicted:", class_names[predicted])

print("\nMisclassified held-out images:")
for path, actual, predicted in zip(test_paths, test_labels, predicted_labels):
    if actual != predicted:
        print(path.name, "| actual:", class_names[actual], "| predicted:", class_names[predicted])

# Save the model for new-image predictions and save evaluation results
model_folder.mkdir(exist_ok=True)
results_folder.mkdir(exist_ok=True)

model.save(model_folder / "plant_classifier.keras")
with open(model_folder / "class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

np.savetxt(results_folder / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
(results_folder / "accuracy.txt").write_text(
    f"Test accuracy: {accuracy:.2%}\nClass order: {class_names}\n",
    encoding="utf-8"
)

print("\nSaved trained model in:", model_folder)
print("Saved evaluation results in:", results_folder)

