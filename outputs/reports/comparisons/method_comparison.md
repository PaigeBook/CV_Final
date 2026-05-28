# Method Comparison
## Quantitative Comparison
| Method | Accuracy | Precision | Recall | F1-score | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| HOG + SVM | 0.9839 | 0.9806 | 0.9855 | 0.9830 | |
| HOG + Random Forest | 0.9772 | 0.9782 | 0.9734 | 0.9757 | |

## Interpretation

- Results summarize scalar metrics for each method on the same train/test split.
- The raw-pixel baseline provides the minimum expected performance for justification in the report.
- Use confusion matrices and per-class reports to inspect class-specific behaviour.

## Efficiency Metrics

| Metric | Value |
| --- | ---: |
| Preprocessing time | 0.0017 s |
| Segmentation time | 0.0011 s |
| Feature extraction time | 0.0069 s |
| Classification time | 0.4143 s |
| Total inference time per image | 0.423985 s |
| SVM training time | 36.1147 s |
| Random Forest training time | 16.6011 s |
| Baseline training time | 0.0000 s |
| SVM inference time per image | 0.414338 s |
| Random Forest inference time per image | 0.000151 s |
| Baseline inference time per image | 0.000000 s |
