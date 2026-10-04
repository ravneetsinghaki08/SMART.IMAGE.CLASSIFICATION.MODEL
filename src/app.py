from pathlib import Path
import json
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
import tensorflow as tf
from tkinterdnd2 import DND_FILES, TkinterDnD


project_folder = Path(__file__).resolve().parent.parent
model_folder = project_folder / "model"

try:
    model = tf.keras.models.load_model(model_folder / "plant_classifier.keras")
    class_names = json.loads(
        (model_folder / "class_names.json").read_text(encoding="utf-8")
    )
    model_message = "Ready. Drop a plant image below."
except Exception:
    model = None
    class_names = []
    model_message = "Model files not found. Run src/train.py first."

root = TkinterDnD.Tk()
root.title("Smart Plant Classifier")
root.geometry("1280x720")
root.minsize(800, 450)
root.configure(bg="#f4f6fb")
root.resizable(True, True)

# A scrollable canvas lets the window content move in both directions.
outer_frame = tk.Frame(root, bg="#f4f6fb")
outer_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(outer_frame, bg="#f4f6fb", highlightthickness=0)
vertical_bar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
horizontal_bar = ttk.Scrollbar(outer_frame, orient="horizontal", command=canvas.xview)
canvas.configure(yscrollcommand=vertical_bar.set, xscrollcommand=horizontal_bar.set)

outer_frame.grid_rowconfigure(0, weight=1)
outer_frame.grid_columnconfigure(0, weight=1)
canvas.grid(row=0, column=0, sticky="nsew")
vertical_bar.grid(row=0, column=1, sticky="ns")
horizontal_bar.grid(row=1, column=0, sticky="ew")

content = tk.Frame(canvas, bg="#f4f6fb")
content_window = canvas.create_window((0, 0), window=content, anchor="nw")


def update_scroll_region(event=None):
    canvas.configure(scrollregion=canvas.bbox("all"))


def fit_content_width(event):
    # Keep the content at least 1000 pixels wide so horizontal scrolling works
    # when the window is made narrower than the content.
    width = max(event.width, 1000)
    canvas.itemconfigure(content_window, width=width)
    update_scroll_region()


content.bind("<Configure>", update_scroll_region)
canvas.bind("<Configure>", fit_content_width)


def scroll_vertical(event):
    canvas.yview_scroll(-int(event.delta / 120), "units")


def scroll_horizontal(event):
    canvas.xview_scroll(-int(event.delta / 120), "units")


root.bind_all("<MouseWheel>", scroll_vertical)
root.bind_all("<Shift-MouseWheel>", scroll_horizontal)

title = tk.Label(
    content,
    text="Smart Plant Classifier",
    font=("Segoe UI", 26, "bold"),
    bg="#f4f6fb",
    fg="#1f2937"
)
title.pack(pady=(40, 6))

subtitle = tk.Label(
    content,
    text="Drop one plant image to see its category and type",
    font=("Segoe UI", 13),
    bg="#f4f6fb",
    fg="#5b6472"
)
subtitle.pack(pady=(0, 24))

drop_area = tk.Label(
    content,
    text="Drag an image here\n\nPNG, JPG, or JPEG",
    font=("Segoe UI", 16),
    bg="white",
    fg="#52627a",
    relief="groove",
    bd=2,
    width=70,
    height=9
)
drop_area.pack(padx=80, fill="x")
drop_area.drop_target_register(DND_FILES)

file_text = tk.StringVar(value="No image selected")
file_label = tk.Label(
    content,
    textvariable=file_text,
    font=("Segoe UI", 10),
    bg="#f4f6fb",
    fg="#5b6472"
)
file_label.pack(pady=(16, 4))

result_text = tk.StringVar(value=model_message)
result_label = tk.Label(
    content,
    textvariable=result_text,
    font=("Segoe UI", 16, "bold"),
    bg="#f4f6fb",
    fg="#1f2937",
    justify="center",
    wraplength=850
)
result_label.pack(pady=18)


def classify_image(image_path):
    if model is None:
        result_text.set(model_message)
        return

    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        result_text.set("Please choose a PNG, JPG, or JPEG image.")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        result_text.set("I couldn't open that image. Please choose another one.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.array(image, dtype="float32") / 255.0
    image = np.expand_dims(image, axis=0)

    scores = model.predict(image, verbose=0)[0]
    best_number = int(np.argmax(scores))
    category, subcategory = class_names[best_number].split("/", 1)

    file_text.set(image_path.name)
    result_text.set(
        f"Category: {category.title()}\n"
        f"Subcategory: {subcategory.title()}\n"
        f"Confidence: {scores[best_number] * 100:.1f}%"
    )


def choose_image():
    filename = filedialog.askopenfilename(
        title="Choose a plant image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )
    if filename:
        classify_image(Path(filename))


def image_dropped(event):
    dropped_paths = root.tk.splitlist(event.data)
    if dropped_paths:
        classify_image(Path(dropped_paths[0]))


drop_area.dnd_bind("<<Drop>>", image_dropped)

browse_button = tk.Button(
    content,
    text="Choose image",
    command=choose_image,
    font=("Segoe UI", 12, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    relief="flat",
    padx=28,
    pady=12,
    cursor="hand2"
)
browse_button.pack(pady=(4, 40))

root.mainloop()

