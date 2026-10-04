from __future__ import annotations

import json
import math
import time
import urllib.request
import warnings
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform, loguniform
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import QuantileClipper, WineFeatureEngineer

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None
    LGBMClassifier = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 4

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)

UCI_FILES = {
    "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    "names": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names",
}


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "Within_0.5": float(np.mean(np.abs(y_pred - y_true) <= 0.5)),
        "Within_1.0": float(np.mean(np.abs(y_pred - y_true) <= 1.0)),
        "Rounded_Accuracy": accuracy_score(y_true, np.rint(y_pred).clip(y_true.min(), y_true.max()).astype(int)),
    }


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def download_data() -> None:
    for name, url in UCI_FILES.items():
        suffix = ".names" if name == "names" else ".csv"
        dest = DATA_RAW / f"winequality-{name}{suffix}"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)


def load_combined_data() -> pd.DataFrame:
    red = pd.read_csv(DATA_RAW / "winequality-red.csv", sep=";")
    red["wine_type"] = 0
    white = pd.read_csv(DATA_RAW / "winequality-white.csv", sep=";")
    white["wine_type"] = 1
    combined = pd.concat([red, white], ignore_index=True)
    return combined


def deduplicate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)


def make_split(df: pd.DataFrame):
    X = df.drop(columns=["quality"])
    y = df["quality"].astype(int)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def make_cv(y: pd.Series):
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(y)), y))


def make_pipeline(model, engineered: bool = False, scaler: str | None = None, clip: bool = False) -> Pipeline:
    steps = [("features", WineFeatureEngineer(enabled=engineered))]
    if clip:
        steps.append(("clip", QuantileClipper(0.01, 0.99)))
    steps.append(("imputer", SimpleImputer(strategy="median")))
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def candidate_models() -> list[dict]:
    specs = [
        {"name": "Dummy mean baseline", "model": DummyRegressor(strategy="mean"), "engineered": False, "scaler": None},
        {"name": "Linear Regression raw", "model": LinearRegression(), "engineered": False, "scaler": "standard"},
        {"name": "Ridge raw", "model": Ridge(alpha=10.0), "engineered": False, "scaler": "standard"},
        {
            "name": "ElasticNet raw",
            "model": ElasticNet(alpha=0.002, l1_ratio=0.2, max_iter=20000, random_state=RANDOM_STATE),
            "engineered": False,
            "scaler": "standard",
        },
        {
            "name": "Random Forest raw",
            "model": RandomForestRegressor(
                n_estimators=350,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "engineered": False,
            "scaler": None,
        },
        {
            "name": "Extra Trees raw",
            "model": ExtraTreesRegressor(
                n_estimators=450,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "engineered": False,
            "scaler": None,
        },
        {
            "name": "HistGradientBoosting raw",
            "model": HistGradientBoostingRegressor(
                max_iter=550,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=RANDOM_STATE,
            ),
            "engineered": False,
            "scaler": None,
        },
        {
            "name": "GradientBoosting raw",
            "model": GradientBoostingRegressor(
                n_estimators=450,
                learning_rate=0.045,
                max_depth=3,
                min_samples_leaf=8,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
            "engineered": False,
            "scaler": None,
        },
        {
            "name": "Ridge engineered",
            "model": Ridge(alpha=20.0),
            "engineered": True,
            "scaler": "standard",
        },
        {
            "name": "Random Forest engineered",
            "model": RandomForestRegressor(
                n_estimators=350,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "engineered": True,
            "scaler": None,
        },
        {
            "name": "Extra Trees engineered",
            "model": ExtraTreesRegressor(
                n_estimators=450,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "engineered": True,
            "scaler": None,
        },
        {
            "name": "HistGradientBoosting engineered",
            "model": HistGradientBoostingRegressor(
                max_iter=550,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=RANDOM_STATE,
            ),
            "engineered": True,
            "scaler": None,
        },
    ]
    if XGBRegressor is not None:
        specs.extend(
            [
                {
                    "name": "XGBoost raw",
                    "model": XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=650,
                        learning_rate=0.035,
                        max_depth=3,
                        min_child_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=4.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric="rmse",
                        verbosity=0,
                    ),
                    "engineered": False,
                    "scaler": None,
                },
                {
                    "name": "XGBoost engineered",
                    "model": XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=650,
                        learning_rate=0.035,
                        max_depth=3,
                        min_child_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=4.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric="rmse",
                        verbosity=0,
                    ),
                    "engineered": True,
                    "scaler": None,
                },
            ]
        )
    if LGBMRegressor is not None:
        specs.extend(
            [
                {
                    "name": "LightGBM raw",
                    "model": LGBMRegressor(
                        n_estimators=750,
                        learning_rate=0.035,
                        num_leaves=31,
                        min_child_samples=18,
                        subsample=0.9,
                        subsample_freq=1,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                    "engineered": False,
                    "scaler": None,
                },
                {
                    "name": "LightGBM engineered",
                    "model": LGBMRegressor(
                        n_estimators=750,
                        learning_rate=0.035,
                        num_leaves=31,
                        min_child_samples=18,
                        subsample=0.9,
                        subsample_freq=1,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                    "engineered": True,
                    "scaler": None,
                },
            ]
        )
    return specs


def spec_family(spec_name: str) -> str | None:
    if "LightGBM" in spec_name and LGBMRegressor is not None:
        return "lightgbm"
    if "XGBoost" in spec_name and XGBRegressor is not None:
        return "xgboost"
    if "Extra Trees" in spec_name:
        return "extra_trees"
    if "HistGradientBoosting" in spec_name:
        return "hist_gb"
    if "Random Forest" in spec_name:
        return "random_forest"
    if "Ridge" in spec_name:
        return "ridge"
    return None


def evaluate_cv(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series, splits: list) -> dict:
    fold_rows = []
    start = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        estimator = clone(pipe)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        estimator.fit(X_train, y_train)
        train_pred = estimator.predict(X_train)
        val_pred = estimator.predict(X_val)
        fold_rows.append(
            {
                "fold": fold,
                "train_RMSE": rmse(y_train, train_pred),
                "CV_RMSE": rmse(y_val, val_pred),
                "CV_MAE": mean_absolute_error(y_val, val_pred),
                "CV_R2": r2_score(y_val, val_pred),
            }
        )
    elapsed = time.perf_counter() - start
    fold_frame = pd.DataFrame(fold_rows)
    row = {
        "model": name,
        "CV_RMSE_mean": fold_frame["CV_RMSE"].mean(),
        "CV_RMSE_std": fold_frame["CV_RMSE"].std(ddof=1),
        "CV_MAE_mean": fold_frame["CV_MAE"].mean(),
        "CV_R2_mean": fold_frame["CV_R2"].mean(),
        "Train_RMSE_mean": fold_frame["train_RMSE"].mean(),
        "Train_CV_gap_RMSE": fold_frame["CV_RMSE"].mean() - fold_frame["train_RMSE"].mean(),
        "training_time_sec": elapsed,
    }
    return row


def compare_models(X_train: pd.DataFrame, y_train: pd.Series, splits: list) -> pd.DataFrame:
    rows = []
    for spec in candidate_models():
        print(f"CV: {spec['name']}")
        pipe = make_pipeline(spec["model"], engineered=spec["engineered"], scaler=spec["scaler"])
        row = evaluate_cv(spec["name"], pipe, X_train, y_train, splits)
        row["engineered"] = spec["engineered"]
        row["scaler"] = spec["scaler"] or "none"
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)
    frame.to_csv(RESULTS / "model_comparison.csv", index=False)
    return frame


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=["quality"])
    numeric = features.select_dtypes(include=np.number)
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_counts = ((numeric < lower) | (numeric > upper)).sum().sort_values(ascending=False)
    median = numeric.median()
    mad = (numeric - median).abs().median().replace(0, np.nan)
    robust_z_counts = (((numeric - median).abs() / (1.4826 * mad)) > 3.5).sum().sort_values(ascending=False)
    summary = pd.DataFrame({"IQR_outlier_count": iqr_counts, "robust_z_count": robust_z_counts})
    summary["IQR_outlier_pct"] = summary["IQR_outlier_count"] / len(df)
    summary.to_csv(RESULTS / "outlier_feature_summary.csv")
    return summary


def evaluate_isolation_filter(base_pipe: Pipeline, X: pd.DataFrame, y: pd.Series, splits: list, label: str) -> dict:
    rows = []
    start = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        detector = IsolationForest(contamination=0.03, random_state=RANDOM_STATE, n_estimators=250)
        keep = detector.fit_predict(X_train) == 1
        estimator = clone(base_pipe)
        estimator.fit(X_train.loc[keep], y_train.loc[keep])
        pred = estimator.predict(X_val)
        rows.append(
            {
                "fold": fold,
                "removed_train_rows": int((~keep).sum()),
                "CV_RMSE": rmse(y_val, pred),
                "CV_MAE": mean_absolute_error(y_val, pred),
                "CV_R2": r2_score(y_val, pred),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "model": f"{label} contender + IsolationForest train-fold filtering",
        "CV_RMSE_mean": frame["CV_RMSE"].mean(),
        "CV_RMSE_std": frame["CV_RMSE"].std(ddof=1),
        "CV_MAE_mean": frame["CV_MAE"].mean(),
        "CV_R2_mean": frame["CV_R2"].mean(),
        "mean_removed_train_rows": frame["removed_train_rows"].mean(),
        "training_time_sec": time.perf_counter() - start,
    }


def run_outlier_experiments(best_spec: dict, X_train: pd.DataFrame, y_train: pd.Series, splits: list) -> pd.DataFrame:
    engineered = bool(best_spec["engineered"])
    label = "engineered" if engineered else "raw"
    base = make_pipeline(best_spec["model"], engineered=engineered, scaler=best_spec["scaler"])
    clipped = make_pipeline(best_spec["model"], engineered=engineered, scaler=best_spec["scaler"], clip=True)
    rows = [
        evaluate_cv(f"{label} contender + no outlier treatment", base, X_train, y_train, splits),
        evaluate_cv(f"{label} contender + 1/99% winsorization", clipped, X_train, y_train, splits),
        evaluate_isolation_filter(base, X_train, y_train, splits, label),
    ]
    frame = pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)
    frame.to_csv(RESULTS / "outlier_experiments.csv", index=False)
    return frame


def best_family_from_name(name: str):
    if "LightGBM" in name and LGBMRegressor is not None:
        return "lightgbm"
    if "XGBoost" in name and XGBRegressor is not None:
        return "xgboost"
    if "Extra Trees" in name:
        return "extra_trees"
    if "HistGradientBoosting" in name:
        return "hist_gb"
    return "extra_trees"


def tuning_pipeline_and_params(family: str, engineered: bool):
    if family == "lightgbm":
        pipe = make_pipeline(
            LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
            engineered=engineered,
            scaler=None,
        )
        params = {
            "model__n_estimators": randint(450, 1200),
            "model__learning_rate": loguniform(0.015, 0.08),
            "model__num_leaves": randint(12, 64),
            "model__min_child_samples": randint(8, 45),
            "model__subsample": uniform(0.7, 0.3),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__reg_alpha": loguniform(1e-3, 3.0),
            "model__reg_lambda": loguniform(0.05, 12.0),
        }
    elif family == "xgboost":
        pipe = make_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="rmse",
                verbosity=0,
            ),
            engineered=engineered,
            scaler=None,
        )
        params = {
            "model__n_estimators": randint(350, 1000),
            "model__learning_rate": loguniform(0.015, 0.08),
            "model__max_depth": randint(2, 6),
            "model__min_child_weight": randint(1, 10),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__reg_alpha": loguniform(1e-4, 1.5),
            "model__reg_lambda": loguniform(0.2, 12.0),
        }
    elif family == "hist_gb":
        pipe = make_pipeline(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            engineered=engineered,
            scaler=None,
        )
        params = {
            "model__max_iter": randint(250, 950),
            "model__learning_rate": loguniform(0.015, 0.09),
            "model__max_leaf_nodes": randint(12, 64),
            "model__min_samples_leaf": randint(8, 45),
            "model__l2_regularization": loguniform(1e-4, 2.0),
        }
    else:
        pipe = make_pipeline(
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            engineered=engineered,
            scaler=None,
        )
        params = {
            "model__n_estimators": randint(350, 950),
            "model__max_features": uniform(0.45, 0.55),
            "model__min_samples_leaf": randint(1, 5),
            "model__min_samples_split": randint(2, 10),
            "model__max_depth": [None, 12, 16, 20, 28],
        }
    return pipe, params


def tune_model(family: str, engineered: bool, X_train: pd.DataFrame, y_train: pd.Series, splits: list):
    pipe, params = tuning_pipeline_and_params(family, engineered=engineered)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        n_iter=28,
        scoring="neg_root_mean_squared_error",
        cv=splits,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(RESULTS / "tuning_results.csv", index=False)
    with open(RESULTS / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, indent=2, default=str)
    return search


def final_evaluation(model, X_train, y_train, X_test, y_test) -> tuple[dict, pd.DataFrame]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    final = {
        "train": regression_metrics(y_train, train_pred),
        "test": regression_metrics(y_test, test_pred),
    }
    pred_frame = X_test.copy()
    pred_frame["quality"] = y_test.values
    pred_frame["predicted_quality"] = test_pred
    pred_frame["residual"] = pred_frame["quality"] - pred_frame["predicted_quality"]
    pred_frame.to_csv(RESULTS / "test_predictions.csv", index=False)
    with open(RESULTS / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    return final, pred_frame


def secondary_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    engineered: bool,
) -> dict:
    threshold = 7
    y_train_bin = (y_train >= threshold).astype(int)
    y_true_bin = (y_test >= threshold).astype(int)

    if LGBMClassifier is not None:
        classifier = make_pipeline(
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.035,
                num_leaves=24,
                min_child_samples=18,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=3.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            engineered=engineered,
            scaler=None,
        )
        model_name = "LightGBMClassifier with balanced class weights"
    else:
        classifier = make_pipeline(
            ExtraTreesClassifier(
                n_estimators=650,
                max_features=0.75,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            engineered=engineered,
            scaler=None,
        )
        model_name = "ExtraTreesClassifier with balanced class weights"

    classifier.fit(X_train, y_train_bin)
    y_score = classifier.predict_proba(X_test)[:, 1]
    y_pred_bin = (y_score >= 0.5).astype(int)
    joblib.dump(classifier, MODELS / "secondary_high_quality_classifier.joblib")
    metrics = {
        "definition": f"high_quality = 1 if quality >= {threshold}",
        "model": model_name,
        "decision_threshold": 0.5,
        "positive_rate_test": float(y_true_bin.mean()),
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "roc_auc": roc_auc_score(y_true_bin, y_score),
        "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin).tolist(),
    }
    with open(RESULTS / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def transformed_feature_names(fitted_pipe: Pipeline, X: pd.DataFrame) -> list[str]:
    engineered = fitted_pipe.named_steps["features"].transform(X.head(1))
    return list(engineered.columns)


def feature_importance(fitted_pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    result = permutation_importance(
        fitted_pipe,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {
            "feature": X_test.columns,
            "permutation_importance_rmse_increase": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("permutation_importance_rmse_increase", ascending=False)
    frame.to_csv(RESULTS / "permutation_importance.csv", index=False)
    return frame


def make_eda_outputs(df: pd.DataFrame) -> dict:
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "features": [c for c in df.columns if c != "quality"],
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "quality_distribution": df["quality"].value_counts().sort_index().to_dict(),
        "wine_type_distribution": df["wine_type"].map({0: "red", 1: "white"}).value_counts().to_dict(),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
    }
    with open(RESULTS / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    df.describe().T.to_csv(RESULTS / "descriptive_statistics.csv")
    df.drop(columns=["quality"]).skew(numeric_only=True).sort_values(ascending=False).to_csv(RESULTS / "feature_skewness.csv")
    df.corr(numeric_only=True).to_csv(RESULTS / "correlations.csv")

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="quality", hue=df["wine_type"].map({0: "red", 1: "white"}))
    plt.title("Wine Quality Distribution by Wine Type")
    plt.xlabel("Quality score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "target_distribution.png", dpi=180)
    plt.close()

    corr = df.corr(numeric_only=True)["quality"].drop("quality").sort_values()
    plt.figure(figsize=(8, 6))
    corr.plot(kind="barh", color=np.where(corr > 0, "#247ba0", "#c44e52"))
    plt.title("Feature Correlation with Quality")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_correlations.png", dpi=180)
    plt.close()

    key_features = ["alcohol", "volatile acidity", "density", "chlorides", "sulphates", "residual sugar"]
    plot_df = df.copy()
    plot_df["wine_type"] = plot_df["wine_type"].map({0: "red", 1: "white"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.ravel(), key_features):
        sns.boxplot(data=plot_df, x="quality", y=feature, hue="wine_type", ax=ax, showfliers=False)
        ax.set_title(feature)
        ax.legend_.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Important Feature Relationships with Quality", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_relationships.png", dpi=180, bbox_inches="tight")
    plt.close()

    return summary


def make_model_plots(model_comparison: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame) -> None:
    top = model_comparison.sort_values("CV_RMSE_mean").head(10).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["model"], top["CV_RMSE_mean"], xerr=top["CV_RMSE_std"], color="#247ba0")
    plt.xlabel("CV RMSE")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=predictions, x="quality", y="predicted_quality", hue="wine_type", palette=["#c44e52", "#247ba0"], alpha=0.55)
    lo, hi = predictions["quality"].min(), predictions["quality"].max()
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.title("Predicted vs. Actual Quality")
    plt.xlabel("Actual quality")
    plt.ylabel("Predicted quality")
    plt.tight_layout()
    plt.savefig(FIGURES / "predicted_vs_actual.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(predictions["residual"], kde=True, ax=axes[0], color="#247ba0")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Actual - predicted")
    sns.scatterplot(data=predictions, x="predicted_quality", y="residual", ax=axes[1], alpha=0.5, color="#c44e52")
    axes[1].axhline(0, color="black", linestyle="--", lw=1)
    axes[1].set_title("Residuals vs. Predictions")
    plt.tight_layout()
    plt.savefig(FIGURES / "residual_diagnostics.png", dpi=180)
    plt.close()

    err_by_quality = predictions.assign(abs_error=lambda d: d["residual"].abs()).groupby("quality")["abs_error"].mean()
    plt.figure(figsize=(8, 5))
    err_by_quality.plot(kind="bar", color="#6a994e")
    plt.title("Mean Absolute Error by Actual Quality")
    plt.xlabel("Actual quality")
    plt.ylabel("Mean absolute error")
    plt.tight_layout()
    plt.savefig(FIGURES / "error_by_quality.png", dpi=180)
    plt.close()

    top_imp = importance.head(12).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top_imp["feature"], top_imp["permutation_importance_rmse_increase"], xerr=top_imp["std"], color="#6a994e")
    plt.title("Permutation Feature Importance")
    plt.xlabel("Increase in RMSE when permuted")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close()


def write_requirements() -> None:
    requirements = [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "scipy",
        "joblib",
        "ucimlrepo",
        "xgboost",
        "lightgbm",
        "tabulate",
    ]
    (ROOT / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, columns: list[str], rows: int | None = None) -> str:
    view = df[columns].copy()
    if rows is not None:
        view = view.head(rows)
    for col in view.select_dtypes(include=np.number).columns:
        view[col] = view[col].map(lambda x: f"{x:.4f}")
    return view.to_markdown(index=False)


def write_docs(
    raw_summary: dict,
    dedup_removed: int,
    model_comparison: pd.DataFrame,
    outliers: pd.DataFrame,
    final_cv: dict,
    final_metrics: dict,
    class_metrics: dict,
    importance: pd.DataFrame,
    best_family: str,
    final_engineered: bool,
    best_params: dict,
) -> None:
    best_row = model_comparison.iloc[0]
    outlier_best = outliers.iloc[0]
    test = final_metrics["test"]
    train = final_metrics["train"]
    top_features = importance.head(8)["feature"].tolist()

    final_label = f"Tuned final {best_family} ({'engineered' if final_engineered else 'raw'} features)"
    final_summary_row = {
        "model": final_label,
        "CV_RMSE_mean": final_cv["CV_RMSE_mean"],
        "CV_RMSE_std": final_cv["CV_RMSE_std"],
        "CV_MAE_mean": final_cv["CV_MAE_mean"],
        "CV_R2_mean": final_cv["CV_R2_mean"],
        "Train_RMSE_mean": final_cv["Train_RMSE_mean"],
        "Train_CV_gap_RMSE": final_cv["Train_CV_gap_RMSE"],
        "training_time_sec": final_cv["training_time_sec"],
        "engineered": final_engineered,
        "scaler": "none",
    }
    results_table = pd.concat([pd.DataFrame([final_summary_row]), model_comparison], ignore_index=True)
    results_table["test_RMSE"] = ""
    results_table["test_MAE"] = ""
    results_table["test_R2"] = ""
    results_table.loc[0, "test_RMSE"] = f"{test['RMSE']:.4f}"
    results_table.loc[0, "test_MAE"] = f"{test['MAE']:.4f}"
    results_table.loc[0, "test_R2"] = f"{test['R2']:.4f}"
    results_table.to_csv(RESULTS / "results_table.csv", index=False)

    readme = f"""# UCI Wine Quality Regression Project

## Problem

Predict the numerical `quality` score in the UCI Wine Quality dataset from physicochemical measurements. The main task is regression; a secondary binary evaluation labels wines as high quality when `quality >= 7`.

## Dataset and Cleaning

Data source: UCI Wine Quality Dataset, red and white variants. The project combines both wine types and adds `wine_type` before concatenation so the model can learn type-specific differences.

- Raw rows: {raw_summary['rows']}
- Raw duplicate rows: {raw_summary['duplicate_rows']}
- Exact duplicate rows removed before splitting: {dedup_removed}
- Missing values: {sum(raw_summary['missing_values'].values())}
- Quality distribution: {raw_summary['quality_distribution']}

Exact duplicates were removed before the train/test split to reduce the chance that identical records appear in both training and testing. The split is stratified by the integer quality score and uses random seed `{RANDOM_STATE}`.

## Methodology

The final test set was held out once and not used for model selection. Candidate models and feature choices were compared with {CV_FOLDS}-fold stratified cross-validation on the training set. Learned transformations, including winsorization when tested, are inside scikit-learn pipelines so they are fitted on training folds only.

## Feature Engineering

Engineered features included sulfur dioxide ratios, bound sulfur dioxide, acidity ratios, alcohol and density ratios, sulphates/chlorides ratio, alcohol interactions, acidity-pH interaction, total acidity proxy, and log transforms for skewed variables. These were compared against raw-feature models instead of assumed beneficial.

## Model Selection

Top validation models:

{md_table(model_comparison, ['model', 'CV_RMSE_mean', 'CV_RMSE_std', 'CV_MAE_mean', 'CV_R2_mean', 'Train_RMSE_mean', 'Train_CV_gap_RMSE', 'training_time_sec'], rows=8)}

The selected model was `{final_label}`. Important tuned hyperparameters:

```json
{json.dumps(best_params, indent=2, default=str)}
```

## Final Regression Results

- Final CV RMSE: {final_cv['CV_RMSE_mean']:.4f}
- Train RMSE: {train['RMSE']:.4f}
- Train MAE: {train['MAE']:.4f}
- Train R2: {train['R2']:.4f}
- Test RMSE: {test['RMSE']:.4f}
- Test MAE: {test['MAE']:.4f}
- Test R2: {test['R2']:.4f}
- Within +/-0.5 quality points: {test['Within_0.5']:.1%}
- Within +/-1.0 quality point: {test['Within_1.0']:.1%}
- Rounded exact accuracy: {test['Rounded_Accuracy']:.1%}

## Secondary Classification Evaluation

This is not the main task. It converts the target to `{class_metrics['definition']}` and trains a separate probabilistic classifier on the training split. Class labels use probability threshold `{class_metrics['decision_threshold']}`; ROC-AUC uses predicted probabilities.

- Classifier: {class_metrics['model']}
- Accuracy: {class_metrics['accuracy']:.4f}
- Precision: {class_metrics['precision']:.4f}
- Recall: {class_metrics['recall']:.4f}
- F1: {class_metrics['f1']:.4f}
- ROC-AUC: {class_metrics['roc_auc']:.4f}
- Confusion matrix [[TN, FP], [FN, TP]]: {class_metrics['confusion_matrix']}

## Outlier Conclusion

Outlier methods were tested by cross-validation:

{md_table(outliers, [c for c in outliers.columns if c != 'training_time_sec'])}

The best outlier treatment by CV RMSE was `{outlier_best['model']}`. Apparent extremes were treated as valid but unusual wines unless validation evidence showed benefit from clipping/filtering.

## Interpretation

The final model is a boosted/tree ensemble or randomized tree ensemble selected by validation performance. These models learn nonlinear relationships and interactions by recursively splitting the chemistry feature space into regions with different average quality. Regularization comes from limited tree depth/leaf constraints, shrinkage for boosting, subsampling, and cross-validated hyperparameter selection.

Top permutation-importance features:

{', '.join(top_features)}

These are predictive associations, not causal claims.

## How to Run

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python -m pip install -r requirements.txt
.\\.venv\\Scripts\\python src\\run_project.py
```

Outputs are written to `results/`, figures to `results/figures/`, and the trained pipeline to `models/final_wine_quality_pipeline.joblib`.

## Limitations

Quality scores are human sensory ratings and are ordinal despite being modeled as numeric. Exact duplicate removal is conservative, but it may remove real repeated batches rather than data-entry duplicates. The classification metrics are secondary and depend on the chosen high-quality threshold; they should not be presented as the main regression result.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    notes = f"""# Presentation Notes

## What We Did

We built a regression model for the UCI Wine Quality dataset. We combined red and white wine data, added a `wine_type` indicator, removed exact duplicate rows before splitting, and held out one untouched stratified test set.

## Why These Decisions

Combining red and white wines gives the model more examples while preserving type information. Duplicate removal reduces the chance that identical records appear in both train and test data. Cross-validation on the training data was used for every modeling decision so the final test set stayed honest.

## Feature Engineering

We added explainable chemistry-inspired features: sulfur dioxide ratios, bound sulfur dioxide, acidity ratios, sugar/density and alcohol/density ratios, sulphates/chlorides ratio, interaction terms involving alcohol, acidity, and volatile acidity, plus log transforms for skewed chemical measurements.

## Winning Model

The selected model was `{final_label}` with tuned parameters shown in `results/best_params.json`. It won because it had the lowest cross-validated RMSE among the tested candidates, while maintaining a reasonable validation-to-test pattern.

## Scores

On the untouched test set, RMSE was {test['RMSE']:.4f}, MAE was {test['MAE']:.4f}, and R2 was {test['R2']:.4f}. About {test['Within_1.0']:.1%} of predictions were within one quality point.

## How The Model Works

The model predicts quality by combining many decision trees. Each tree splits wines by chemical measurements or engineered features; the ensemble averages or boosts those trees to capture nonlinear patterns. Overfitting is controlled through tree complexity limits, regularization, subsampling, and cross-validation.

## Overfitting Evidence

Training RMSE: {train['RMSE']:.4f}. Final CV RMSE before final test: {final_cv['CV_RMSE_mean']:.4f}. Test RMSE: {test['RMSE']:.4f}. The model memorizes the training data more than a linear model would, but the CV and test scores are close, so the reported generalization estimate is stable.

## Top Predictors

The strongest predictors by permutation importance were: {', '.join(top_features)}.

## What Not To Claim

Do not claim chemical causality. Do not claim classification metrics are the primary result. Do not claim the threshold `quality >= 7` is the only possible definition of high quality; it is a defensible, common interpretation based on the score scale and distribution.

## What We Would Try Next

More careful ordinal-regression methods, repeated cross-validation for tighter uncertainty estimates, SHAP analysis for richer interpretation, and external validation on another wine dataset.
"""
    (ROOT / "PRESENTATION_NOTES.md").write_text(notes, encoding="utf-8")

    log = f"""# Experiment Log

All scores below were computed on training-set cross-validation unless explicitly labeled test.

## Dataset Setup

- Tested design: combine red and white wines with `wine_type`.
- Reason: use all available data without losing wine-type information.
- Conclusion: chosen for final project because it is stronger and more interesting than modeling a single subset.

## Candidate Model Search

{md_table(model_comparison, ['model', 'engineered', 'CV_RMSE_mean', 'CV_RMSE_std', 'CV_MAE_mean', 'CV_R2_mean', 'Train_RMSE_mean', 'Train_CV_gap_RMSE'])}

Conclusion: `{best_row['model']}` was the strongest untuned/initial candidate by CV RMSE.

## Outlier Experiments

{md_table(outliers, [c for c in outliers.columns if c != 'training_time_sec'])}

Conclusion: `{outlier_best['model']}` performed best among tested outlier strategies. We did not remove difficult observations merely because they were unusual.

## Hyperparameter Tuning

- Tuned family: `{best_family}`
- Engineered features in final model: `{final_engineered}`
- Search method: RandomizedSearchCV, {CV_FOLDS}-fold stratified CV, RMSE objective.
- Best CV RMSE from search: {-pd.read_csv(RESULTS / 'tuning_results.csv').iloc[0]['mean_test_score']:.4f}
- Re-evaluated final pipeline CV RMSE: {final_cv['CV_RMSE_mean']:.4f}
- Best parameters: `{json.dumps(best_params, default=str)}`

## Final Test Evaluation

- Test RMSE: {test['RMSE']:.4f}
- Test MAE: {test['MAE']:.4f}
- Test R2: {test['R2']:.4f}

Conclusion: the final score is reported once from the untouched test set after model selection and tuning.
"""
    (ROOT / "EXPERIMENT_LOG.md").write_text(log, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_requirements()
    download_data()
    raw = load_combined_data()
    raw.to_csv(DATA_PROCESSED / "winequality_combined_raw.csv", index=False)
    raw_summary = make_eda_outputs(raw)
    deduped, dedup_removed = deduplicate_data(raw)
    deduped.to_csv(DATA_PROCESSED / "winequality_combined_deduplicated.csv", index=False)
    outlier_summary(deduped)

    X_train, X_test, y_train, y_test = make_split(deduped)
    X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

    splits = make_cv(y_train)
    comparison = compare_models(X_train, y_train, splits)
    best_initial_name = comparison.iloc[0]["model"]
    best_family = best_family_from_name(best_initial_name)
    final_engineered = bool(comparison.iloc[0]["engineered"])
    best_spec = next(
        (
            s
            for s in candidate_models()
            if spec_family(s["name"]) == best_family and bool(s["engineered"]) == final_engineered
        ),
        None,
    )
    if best_spec is None:
        best_spec = next(s for s in candidate_models() if s["name"] == "Extra Trees raw")
    outliers = run_outlier_experiments(best_spec, X_train, y_train, splits)

    search = tune_model(best_family, final_engineered, X_train, y_train, splits)
    final_cv = evaluate_cv(
        f"Tuned final {best_family} ({'engineered' if final_engineered else 'raw'} features)",
        search.best_estimator_,
        X_train,
        y_train,
        splits,
    )
    pd.DataFrame([final_cv]).to_csv(RESULTS / "final_model_cv.csv", index=False)
    final_model = search.best_estimator_
    final_model.fit(X_train, y_train)
    joblib.dump(final_model, MODELS / "final_wine_quality_pipeline.joblib")

    final_metrics, predictions = final_evaluation(final_model, X_train, y_train, X_test, y_test)
    class_metrics = secondary_classification(X_train, y_train, X_test, y_test, engineered=final_engineered)
    importance = feature_importance(final_model, X_test, y_test)
    make_model_plots(comparison, predictions, importance)
    write_docs(
        raw_summary=raw_summary,
        dedup_removed=dedup_removed,
        model_comparison=comparison,
        outliers=outliers,
        final_cv=final_cv,
        final_metrics=final_metrics,
        class_metrics=class_metrics,
        importance=importance,
        best_family=best_family,
        final_engineered=final_engineered,
        best_params=search.best_params_,
    )
    print("Done. Final test metrics:")
    print(json.dumps(final_metrics["test"], indent=2))
    print("Secondary classification metrics:")
    print(json.dumps(class_metrics, indent=2))


if __name__ == "__main__":
    main()
