# Experiment Log

All scores below were computed on training-set cross-validation unless explicitly labeled test.

## Dataset Setup

- Tested design: combine red and white wines with `wine_type`.
- Reason: use all available data without losing wine-type information.
- Conclusion: chosen for final project because it is stronger and more interesting than modeling a single subset.

## Candidate Model Search

| model                           | engineered   |   CV_RMSE_mean |   CV_RMSE_std |   CV_MAE_mean |   CV_R2_mean |   Train_RMSE_mean |   Train_CV_gap_RMSE |
|:--------------------------------|:-------------|---------------:|--------------:|--------------:|-------------:|------------------:|--------------------:|
| Extra Trees raw                 | False        |         0.6836 |        0.0051 |        0.5235 |       0.3959 |            0      |              0.6836 |
| Extra Trees engineered          | True         |         0.6849 |        0.0029 |        0.5232 |       0.3936 |            0      |              0.6849 |
| Random Forest engineered        | True         |         0.686  |        0.0036 |        0.528  |       0.3916 |            0.2535 |              0.4326 |
| Random Forest raw               | False        |         0.6862 |        0.0034 |        0.5286 |       0.3914 |            0.2545 |              0.4317 |
| XGBoost engineered              | True         |         0.6957 |        0.0042 |        0.5406 |       0.3744 |            0.5589 |              0.1368 |
| XGBoost raw                     | False        |         0.6964 |        0.0031 |        0.54   |       0.3732 |            0.5801 |              0.1163 |
| GradientBoosting raw            | False        |         0.6996 |        0.0038 |        0.542  |       0.3674 |            0.5807 |              0.1189 |
| LightGBM raw                    | False        |         0.7024 |        0.0059 |        0.5401 |       0.3622 |            0.269  |              0.4334 |
| LightGBM engineered             | True         |         0.7042 |        0.0078 |        0.5453 |       0.3591 |            0.2201 |              0.4841 |
| HistGradientBoosting raw        | False        |         0.7097 |        0.0068 |        0.5454 |       0.349  |            0.2836 |              0.4261 |
| HistGradientBoosting engineered | True         |         0.7115 |        0.0077 |        0.5498 |       0.3456 |            0.2346 |              0.4769 |
| Ridge engineered                | True         |         0.7228 |        0.0142 |        0.5595 |       0.3245 |            0.7148 |              0.008  |
| Ridge raw                       | False        |         0.7342 |        0.0144 |        0.5655 |       0.303  |            0.7301 |              0.0041 |
| ElasticNet raw                  | False        |         0.7343 |        0.0144 |        0.5656 |       0.3029 |            0.7301 |              0.0042 |
| Linear Regression raw           | False        |         0.7345 |        0.0148 |        0.5656 |       0.3024 |            0.7301 |              0.0044 |
| Dummy mean baseline             | False        |         0.8796 |        0.0007 |        0.6947 |      -0      |            0.8796 |              0      |

Conclusion: `Extra Trees raw` was the strongest untuned/initial candidate by CV RMSE.

## Outlier Experiments

| model                                                |   CV_RMSE_mean |   CV_RMSE_std |   CV_MAE_mean |   CV_R2_mean |   Train_RMSE_mean |   Train_CV_gap_RMSE |   mean_removed_train_rows |
|:-----------------------------------------------------|---------------:|--------------:|--------------:|-------------:|------------------:|--------------------:|--------------------------:|
| raw contender + no outlier treatment                 |         0.6836 |        0.0051 |        0.5235 |       0.3959 |                 0 |              0.6836 |                       nan |
| raw contender + 1/99% winsorization                  |         0.6867 |        0.0041 |        0.5251 |       0.3904 |                 0 |              0.6867 |                       nan |
| raw contender + IsolationForest train-fold filtering |         0.6871 |        0.002  |        0.5253 |       0.3898 |               nan |            nan      |                        96 |

Conclusion: `raw contender + no outlier treatment` performed best among tested outlier strategies. We did not remove difficult observations merely because they were unusual.

## Hyperparameter Tuning

- Tuned family: `extra_trees`
- Engineered features in final model: `False`
- Search method: RandomizedSearchCV, 4-fold stratified CV, RMSE objective.
- Best CV RMSE from search: 0.6830
- Re-evaluated final pipeline CV RMSE: 0.6830
- Best parameters: `{"model__max_depth": 28, "model__max_features": 0.7469126002159203, "model__min_samples_leaf": 1, "model__min_samples_split": 5, "model__n_estimators": 874}`

## Final Test Evaluation

- Test RMSE: 0.6803
- Test MAE: 0.5271
- Test R2: 0.4024

Conclusion: the final score is reported once from the untouched test set after model selection and tuning.
