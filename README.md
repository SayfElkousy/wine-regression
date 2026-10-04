# UCI Wine Quality Regression Project

## Problem

Predict the numerical `quality` score in the UCI Wine Quality dataset from physicochemical measurements. The main task is regression; a secondary binary evaluation labels wines as high quality when `quality >= 7`.

## Dataset and Cleaning

Data source: UCI Wine Quality Dataset, red and white variants. The project combines both wine types and adds `wine_type` before concatenation so the model can learn type-specific differences.

- Raw rows: 6497
- Raw duplicate rows: 1177
- Exact duplicate rows removed before splitting: 1177
- Missing values: 0
- Quality distribution: {3: 30, 4: 216, 5: 2138, 6: 2836, 7: 1079, 8: 193, 9: 5}

Exact duplicates were removed before the train/test split to reduce the chance that identical records appear in both training and testing. The split is stratified by the integer quality score and uses random seed `42`.

## Methodology

The final test set was held out once and not used for model selection. Candidate models and feature choices were compared with 4-fold stratified cross-validation on the training set. Learned transformations, including winsorization when tested, are inside scikit-learn pipelines so they are fitted on training folds only.

## Feature Engineering

Engineered features included sulfur dioxide ratios, bound sulfur dioxide, acidity ratios, alcohol and density ratios, sulphates/chlorides ratio, alcohol interactions, acidity-pH interaction, total acidity proxy, and log transforms for skewed variables. These were compared against raw-feature models instead of assumed beneficial.

## Model Selection

Top validation models:

| model                    |   CV_RMSE_mean |   CV_RMSE_std |   CV_MAE_mean |   CV_R2_mean |   Train_RMSE_mean |   Train_CV_gap_RMSE |   training_time_sec |
|:-------------------------|---------------:|--------------:|--------------:|-------------:|------------------:|--------------------:|--------------------:|
| Extra Trees raw          |         0.6836 |        0.0051 |        0.5235 |       0.3959 |            0      |              0.6836 |              2.3702 |
| Extra Trees engineered   |         0.6849 |        0.0029 |        0.5232 |       0.3936 |            0      |              0.6849 |              3.1297 |
| Random Forest engineered |         0.686  |        0.0036 |        0.528  |       0.3916 |            0.2535 |              0.4326 |              2.469  |
| Random Forest raw        |         0.6862 |        0.0034 |        0.5286 |       0.3914 |            0.2545 |              0.4317 |              2.2876 |
| XGBoost engineered       |         0.6957 |        0.0042 |        0.5406 |       0.3744 |            0.5589 |              0.1368 |              1.7317 |
| XGBoost raw              |         0.6964 |        0.0031 |        0.54   |       0.3732 |            0.5801 |              0.1163 |              1.2349 |
| GradientBoosting raw     |         0.6996 |        0.0038 |        0.542  |       0.3674 |            0.5807 |              0.1189 |              6.8435 |
| LightGBM raw             |         0.7024 |        0.0059 |        0.5401 |       0.3622 |            0.269  |              0.4334 |              3.127  |

The selected model was `Tuned final extra_trees (raw features)`. Important tuned hyperparameters:

```json
{
  "model__max_depth": 28,
  "model__max_features": 0.7469126002159203,
  "model__min_samples_leaf": 1,
  "model__min_samples_split": 5,
  "model__n_estimators": 874
}
```

## Final Regression Results

- Final CV RMSE: 0.6830
- Train RMSE: 0.1820
- Train MAE: 0.1327
- Train R2: 0.9572
- Test RMSE: 0.6803
- Test MAE: 0.5271
- Test R2: 0.4024
- Within +/-0.5 quality points: 56.8%
- Within +/-1.0 quality point: 87.7%
- Rounded exact accuracy: 56.8%

## Secondary Classification Evaluation

This is not the main task. It converts the target to `high_quality = 1 if quality >= 7` and trains a separate probabilistic classifier on the training split. Class labels use probability threshold `0.5`; ROC-AUC uses predicted probabilities.

- Classifier: LightGBMClassifier with balanced class weights
- Accuracy: 0.8139
- Precision: 0.5080
- Recall: 0.6287
- F1: 0.5619
- ROC-AUC: 0.8622
- Confusion matrix [[TN, FP], [FN, TP]]: [[739, 123], [75, 127]]

## Outlier Conclusion

Outlier methods were tested by cross-validation:

| model                                                |   CV_RMSE_mean |   CV_RMSE_std |   CV_MAE_mean |   CV_R2_mean |   Train_RMSE_mean |   Train_CV_gap_RMSE |   mean_removed_train_rows |
|:-----------------------------------------------------|---------------:|--------------:|--------------:|-------------:|------------------:|--------------------:|--------------------------:|
| raw contender + no outlier treatment                 |         0.6836 |        0.0051 |        0.5235 |       0.3959 |                 0 |              0.6836 |                       nan |
| raw contender + 1/99% winsorization                  |         0.6867 |        0.0041 |        0.5251 |       0.3904 |                 0 |              0.6867 |                       nan |
| raw contender + IsolationForest train-fold filtering |         0.6871 |        0.002  |        0.5253 |       0.3898 |               nan |            nan      |                        96 |

The best outlier treatment by CV RMSE was `raw contender + no outlier treatment`. Apparent extremes were treated as valid but unusual wines unless validation evidence showed benefit from clipping/filtering.

## Interpretation

The final model is a boosted/tree ensemble or randomized tree ensemble selected by validation performance. These models learn nonlinear relationships and interactions by recursively splitting the chemistry feature space into regions with different average quality. Regularization comes from limited tree depth/leaf constraints, shrinkage for boosting, subsampling, and cross-validated hyperparameter selection.

Top permutation-importance features:

alcohol, volatile acidity, free sulfur dioxide, wine_type, sulphates, total sulfur dioxide, residual sugar, density

These are predictive associations, not causal claims.

## How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python src\run_project.py
```

Outputs are written to `results/`, figures to `results/figures/`, and the trained pipeline to `models/final_wine_quality_pipeline.joblib`.

## Limitations

Quality scores are human sensory ratings and are ordinal despite being modeled as numeric. Exact duplicate removal is conservative, but it may remove real repeated batches rather than data-entry duplicates. The classification metrics are secondary and depend on the chosen high-quality threshold; they should not be presented as the main regression result.
