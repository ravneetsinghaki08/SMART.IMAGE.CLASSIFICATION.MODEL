Smart Plant Image Classifier

A Python desktop application that classifies a plant photo into one of the labeled plant types in the dataset. It reports a broad category (such as `FLOWER`, `TREE`, `FRUITS`, or `VEGETABLES`) and a subcategory (such as `ROSE` or `mango tree`).

The interface runs locally on the desktop. Select an image with the button or drag and drop a PNG, JPG, or JPEG file into the app.

## Dataset layout

Each image is labeled by its parent and subcategory folder:

```text
data/plants/
├── FLOWER/
│   ├── MARIGOLD/
│   ├── ROSE/
│   └── SUNFLOWER/
├── FRUITS/
│   ├── apple/
│   ├── mango/
│   └── peach/
├── TREE/
│   ├── apple tree/
│   ├── mango tree/
│   └── oak tree/
└── VEGETABLES/
    ├── CARROT/
    ├── ONION/
    └── TOMATO/
```

Put each photo in the folder that matches its label. The model learns a combined label such as `TREE/mango tree` from the folder path.

## Run the project

Open PowerShell in the project folder. Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy scikit-learn tkinterdnd2
```

Train the model and evaluate it on held-out images:

```powershell
python src\train.py
```

This saves the trained model and its class names under `model/`. Accuracy and the confusion matrix are saved under `results/`.

Open the desktop interface:

```powershell
python src\app.py
```

Drag a plant photo into the window or click **Choose image**. The app displays the predicted category, subcategory, and confidence score. To classify an image from the terminal instead, run `python src\predict.py` and enter its file path.

## How it works

- **OpenCV** reads images, converts them to RGB, and resizes them to 128 × 128 pixels.
- **NumPy** stores the pixels as arrays and scales values to the 0–1 range.
- **Scikit-learn** creates a stratified training/test split and calculates accuracy and a confusion matrix.
- **TensorFlow** trains a small convolutional neural network (CNN) with a horizontal-flip augmentation step.
- **Tkinter** provides the local desktop window; `tkinterdnd2` adds drag-and-drop support.

## Evaluation and limitations

The saved evaluation currently reports **25% test accuracy**. The confusion matrix includes a held-out `FLOWER/SUNFLOWER` image predicted as `FLOWER/ROSE`; this failure and a likely explanation are recorded in [DECISIONS.md](DECISIONS.md). The current baseline needs more varied labeled examples and further improvement before its predictions can be relied on.

This is a single-label image classifier: it predicts one known category/subcategory for the whole image. It does not locate multiple plants or identify two different plants separately when both appear in one photo. It can only predict labels represented in the training folders.

## Project files

- `data/plants/` — labeled training images.
- `src/train.py` — image preparation, data split, model training, and evaluation.
- `src/predict.py` — command-line prediction for a chosen image.
- `src/app.py` — local drag-and-drop desktop interface.
- `model/` — saved TensorFlow model and class-name mapping.
- `results/` — accuracy and confusion-matrix outputs.
- `DECISIONS.md` — project decisions, observed model failure, usage, and limitations.

Do not upload `.venv/`; it contains the local Python environment and is excluded with `.gitignore`.
