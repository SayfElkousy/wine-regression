# Presentation Notes

## What We Did

We built a regression model for the UCI Wine Quality dataset. We combined red and white wine data, added a `wine_type` indicator, removed exact duplicate rows before splitting, and held out one untouched stratified test set.

## Why These Decisions

Combining red and white wines gives the model more examples while preserving type information. Duplicate removal reduces the chance that identical records appear in both train and test data. Cross-validation on the training data was used for every modeling decision so the final test set stayed honest.

## Feature Engineering

We added explainable chemistry-inspired features: sulfur dioxide ratios, bound sulfur dioxide, acidity ratios, sugar/density and alcohol/density ratios, sulphates/chlorides ratio, interaction terms involving alcohol, acidity, and volatile acidity, plus log transforms for skewed chemical measurements.

## Winning Model

The selected model was `Tuned final extra_trees (raw features)` with tuned parameters shown in `results/best_params.json`. It won because it had the lowest cross-validated RMSE among the tested candidates, while maintaining a reasonable validation-to-test pattern.

## Scores

On the untouched test set, RMSE was 0.6803, MAE was 0.5271, and R2 was 0.4024. About 87.7% of predictions were within one quality point.

## How The Model Works

The model predicts quality by combining many decision trees. Each tree splits wines by chemical measurements or engineered features; the ensemble averages or boosts those trees to capture nonlinear patterns. Overfitting is controlled through tree complexity limits, regularization, subsampling, and cross-validation.

## Overfitting Evidence

Training RMSE: 0.1820. Final CV RMSE before final test: 0.6830. Test RMSE: 0.6803. The model memorizes the training data more than a linear model would, but the CV and test scores are close, so the reported generalization estimate is stable.

## Top Predictors

The strongest predictors by permutation importance were: alcohol, volatile acidity, free sulfur dioxide, wine_type, sulphates, total sulfur dioxide, residual sugar, density.

## What Not To Claim

Do not claim chemical causality. Do not claim classification metrics are the primary result. Do not claim the threshold `quality >= 7` is the only possible definition of high quality; it is a defensible, common interpretation based on the score scale and distribution.

## What We Would Try Next

More careful ordinal-regression methods, repeated cross-validation for tighter uncertainty estimates, SHAP analysis for richer interpretation, and external validation on another wine dataset.
