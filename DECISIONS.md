# SMART.IMAGE.CLASSIFICATION.MODEL
Classifies plant photos by category—flower, fruit, tree, or vegetable—and subtype, such as rose or mango. Built with Python, TensorFlow, OpenCV, NumPy, and scikit-learn, with a desktop app for selecting or dragging and dropping an image to see its prediction.

## How does this model classify

The labels come from the folder path. For example, an image in
`data/plants/TREE/mango tree` has the category `TREE` and subcategory
`mango tree`. The model predicts one category/subcategory pair for the
whole image.

## How does it prepare image and model

OpenCV loads each PNG/JPG, converts it to RGB, and resizes it to 128 by 128
pixels. NumPy scales pixel values to the 0–1 range. A small TensorFlow CNN is
used as the baseline model, with a random horizontal flip as the augmentation
step. Scikit-learn makes a stratified train/test split and calculates accuracy
and the confusion matrix.

## Recorded model failure

The saved evaluation reports **25% test accuracy**. In the confusion matrix,
a held-out image labeled `FLOWER/SUNFLOWER` was predicted as
`FLOWER/ROSE`. The saved matrix records labels but not the image filename.

**Likely explanation:** the model may not have learned enough visual details
to distinguish the flower types. Similar colors, shapes, backgrounds, or
lighting could have influenced the prediction. This is a likely explanation;
the image itself should be inspected to confirm.

## Limitation

This baseline predicts one known category/subcategory for the whole image. It
does not locate multiple plants in one photo, and it can only choose among
the labels represented in the training folders. The 25% test accuracy shows
that the current model needs more varied labeled examples and further
improvement before its predictions can be relied on.
