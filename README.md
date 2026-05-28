
# Brain MRI Tumour Detection — Launch

One-line launch:

```powershell
python run.py
```

If `brain_tumor_dataset.zip` is placed in the project root the script will
automatically extract it so no manual dataset setup is required.

Dataset layout (required):

```
data/raw/mri/tumour/*
data/raw/mri/no_tumour/*
```

Optional flags: `--sample-dir <path>`, `--use-glcm`, `--use-orb`, `--clahe-clip`, `--clahe-grid`

Outputs: `models/`, `outputs/figures/`, `outputs/predictions/`, `outputs/reports/`

If you want a short demo notebook or a one-off example, I can add it.

What this does (short):

- Preprocesses MRI scans (grayscale → CLAHE → denoise).
- Segments candidate tumour regions using thresholding and morphological cleanup.
- Extracts HOG features (optionally GLCM texture / ORB) and trains/predicts with SVM or Random Forest.

How to interact (brief):

- Place images in `data/raw/mri/tumour/` and `data/raw/mri/no_tumour/`.
- Run `python run.py` to train and evaluate using defaults.
- To run predictions on a folder without retraining, use `--sample-dir <path>`.
- Toggle extra features: `--use-glcm` (texture), `--use-orb` (keypoint summary).
- Adjust CLAHE and segmentation via `--clahe-clip`, `--clahe-grid`, and `--segmentation-method`.

Typical outputs are saved under `models/` and `outputs/`.


