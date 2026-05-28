# Brain MRI Tumour Detection — Launch

One-line launch:

```powershell
python run.py
```

If `brain_tumor_dataset.zip` is placed in the project root the script will
automatically extract it so no manual dataset setup is required.

Dataset layout:

```
data/raw/mri/tumour/*
data/raw/mri/no_tumour/*
```

Optional flags: `--sample-dir <path>`, `--use-glcm`, `--use-orb`, `--clahe-clip`, `--clahe-grid`

Outputs: `models/`, `outputs/figures/`, `outputs/predictions/`, `outputs/reports/`, `outputs/gallery/`, `outputs/features/`, `outputs/misclassified/`

What this does:

- Preprocesses MRI scans (grayscale → CLAHE → denoise).
- Segments candidate tumour regions using Otsu thresholding, morphological cleaning, and small-component removal to stabilise ROI estimation.
- Extracts HOG features (optionally GLCM texture / ORB) and trains/predicts with SVM and Random Forest; a simple raw-pixel baseline (Logistic Regression) is also provided for justification.
- Per-image confidence scores (SVM decision function / predict_proba) are exposed and included in saved galleries and annotations.
- False-negative examples (tumour → predicted no_tumour) are saved to `outputs/misclassified/false_negatives/` for manual review.
- HOG visualisations for sample images are saved under `outputs/features/` and a gallery grouped by outcome is saved under `outputs/gallery/`.
- Per-stage timing (preprocessing, segmentation, feature extraction, classification) is measured and written to the reports for efficiency analysis.

How to interact :

- Place images in `data/raw/mri/tumour/` and `data/raw/mri/no_tumour/`.
- Run `python run.py` to train and evaluate using defaults.
- To run predictions on a folder without retraining, use `--sample-dir <path>`.
- Toggle extra features: `--use-glcm` (texture), `--use-orb` (keypoint summary).
- Adjust CLAHE and segmentation via `--clahe-clip`, `--clahe-grid`, and `--segmentation-method`.

Typical outputs are saved under `models/` and `outputs/`.
