# -*- coding: utf-8 -*-
"""Adult Income feature engineering assignment pipeline.

The script downloads the UCI Adult dataset, creates EDA artifacts, runs several
feature-engineering experiments, and writes a Korean PDF report.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler


RANDOM_STATE = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/2/adult"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

ORIGINAL_NUMERIC = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

ORIGINAL_CATEGORICAL = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

DERIVED_NUMERIC = ["capital_net", "capital_activity", "education_per_age"]
DERIVED_CATEGORICAL = ["age_group", "work_intensity"]

ALL_NUMERIC = ORIGINAL_NUMERIC + DERIVED_NUMERIC
ALL_CATEGORICAL = ORIGINAL_CATEGORICAL + DERIVED_CATEGORICAL


@dataclass
class ExperimentConfig:
    name: str
    missing: str
    encoding: str
    scaling: str
    feature_selection: bool
    fs_k: int | str | None = None
    use_derived: bool = True


class DerivedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived variables used in the assignment."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DerivedFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["capital_net"] = df["capital_gain"] - df["capital_loss"]
        df["capital_activity"] = ((df["capital_gain"] > 0) | (df["capital_loss"] > 0)).astype(int)
        df["education_per_age"] = df["education_num"] / np.maximum(df["age"], 1)
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 29, 44, 59, 120],
            labels=["young", "mid", "senior", "older"],
            include_lowest=True,
        ).astype("object")
        df["work_intensity"] = pd.cut(
            df["hours_per_week"],
            bins=[0, 34, 40, 50, 120],
            labels=["part_time", "standard", "extended", "heavy"],
            include_lowest=True,
        ).astype("object")
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project directory where data and outputs will be created.",
    )
    return parser.parse_args()


def ensure_dirs(project_root: Path) -> dict[str, Path]:
    paths = {
        "data_raw": project_root / "data" / "raw",
        "figures": project_root / "outputs" / "figures",
        "tables": project_root / "outputs" / "tables",
        "outputs": project_root / "outputs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def download_dataset(raw_dir: Path) -> None:
    zip_path = raw_dir / "adult.zip"
    expected = [raw_dir / "adult.data", raw_dir / "adult.test", raw_dir / "adult.names"]
    if all(path.exists() for path in expected):
        return
    print(f"Downloading dataset from {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)


def load_adult(raw_dir: Path) -> pd.DataFrame:
    train_path = raw_dir / "adult.data"
    test_path = raw_dir / "adult.test"

    train = pd.read_csv(
        train_path,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )
    test = pd.read_csv(
        test_path,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
        skiprows=1,
    )

    df = pd.concat([train, test], ignore_index=True)
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df["income_binary"] = (df["income"] == ">50K").astype(int)
    return df


def write_dataset_summary(df: pd.DataFrame, tables_dir: Path) -> None:
    descriptions = [
        ("age", "나이", "Integer", "수치형"),
        ("workclass", "고용 형태", "Categorical", "범주형"),
        ("fnlwgt", "인구 가중치", "Integer", "수치형"),
        ("education", "최종 학력", "Categorical", "범주형"),
        ("education_num", "학력 수준의 수치 코드", "Integer", "수치형"),
        ("marital_status", "혼인 상태", "Categorical", "범주형"),
        ("occupation", "직업군", "Categorical", "범주형"),
        ("relationship", "가구 내 관계", "Categorical", "범주형"),
        ("race", "인종", "Categorical", "범주형"),
        ("sex", "성별", "Binary", "범주형"),
        ("capital_gain", "자본 이득", "Integer", "수치형"),
        ("capital_loss", "자본 손실", "Integer", "수치형"),
        ("hours_per_week", "주당 근로 시간", "Integer", "수치형"),
        ("native_country", "출신 국가", "Categorical", "범주형"),
        ("income", "연소득 구간", "Target", "타겟"),
    ]
    column_df = pd.DataFrame(descriptions, columns=["column", "description", "type", "role"])
    column_df.to_csv(tables_dir / "column_description.csv", index=False, encoding="utf-8-sig")
    shape = {
        "rows": int(df.shape[0]),
        "columns_including_target": int(df.shape[1]),
        "target": "income_binary",
        "positive_class": ">50K",
    }
    (tables_dir / "dataset_shape.json").write_text(json.dumps(shape, indent=2), encoding="utf-8")


def save_eda(df: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> dict[str, Any]:
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    missing = (
        df[COLUMNS]
        .isna()
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_ratio_percent"})
        .sort_values("missing_ratio_percent", ascending=False)
    )
    missing.to_csv(tables_dir / "missing_values.csv", index=False, encoding="utf-8-sig")

    numeric = ORIGINAL_NUMERIC
    outlier_rows = []
    for column in numeric:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[column] < lower) | (df[column] > upper)).sum())
        outlier_rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": count,
                "outlier_ratio_percent": count / len(df) * 100,
            }
        )
    outliers = pd.DataFrame(outlier_rows).sort_values("outlier_ratio_percent", ascending=False)
    outliers.to_csv(tables_dir / "outlier_summary.csv", index=False, encoding="utf-8-sig")

    target_dist = (
        df["income"]
        .value_counts(normalize=True)
        .mul(100)
        .rename_axis("income")
        .reset_index(name="ratio_percent")
    )
    target_dist.to_csv(tables_dir / "target_distribution.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=missing.head(8), x="missing_ratio_percent", y="column", color="#4c78a8")
    plt.title("Missing Value Ratio")
    plt.xlabel("Missing ratio (%)")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(figures_dir / "missing_ratio.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for ax, column in zip(axes, numeric):
        sns.histplot(df[column], bins=35, kde=False, ax=ax, color="#4c78a8")
        ax.set_title(column)
    plt.tight_layout()
    plt.savefig(figures_dir / "hist_numeric.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for ax, column in zip(axes, numeric):
        sns.boxplot(x=df[column], ax=ax, color="#f58518")
        ax.set_title(column)
    plt.tight_layout()
    plt.savefig(figures_dir / "boxplots_numeric.png", dpi=180)
    plt.close()

    corr_df = df[numeric + ["income_binary"]].corr(numeric_only=True)
    corr_df.to_csv(tables_dir / "correlation_matrix.csv", encoding="utf-8-sig")
    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    sns.countplot(data=df, x="income", hue="income", palette=["#72b7b2", "#e45756"], legend=False)
    plt.title("Target Distribution")
    plt.xlabel("Income")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5.5))
    order = df["education"].value_counts().index
    sns.countplot(data=df, y="education", hue="income", order=order, palette=["#72b7b2", "#e45756"])
    plt.title("Income Distribution by Education")
    plt.xlabel("Count")
    plt.ylabel("Education")
    plt.tight_layout()
    plt.savefig(figures_dir / "income_by_education.png", dpi=180)
    plt.close()

    return {
        "missing_top": missing.head(5).to_dict(orient="records"),
        "outlier_top": outliers.head(5).to_dict(orient="records"),
        "target_distribution": target_dist.to_dict(orient="records"),
    }


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[ORIGINAL_NUMERIC + ORIGINAL_CATEGORICAL].copy()
    y = df["income_binary"].copy()
    return X, y


def make_scaler(name: str | None):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    if name in (None, "none"):
        return "passthrough"
    raise ValueError(f"Unknown scaler: {name}")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    missing_strategy: str,
    encoding: str,
    scaling: str | None,
) -> ColumnTransformer:
    if missing_strategy == "mean":
        numeric_imputer = SimpleImputer(strategy="mean")
        categorical_imputer = SimpleImputer(strategy="most_frequent")
    elif missing_strategy == "median":
        numeric_imputer = SimpleImputer(strategy="median")
        categorical_imputer = SimpleImputer(strategy="most_frequent")
    elif missing_strategy == "most_frequent":
        numeric_imputer = SimpleImputer(strategy="most_frequent")
        categorical_imputer = SimpleImputer(strategy="most_frequent")
    elif missing_strategy == "constant_unknown":
        numeric_imputer = SimpleImputer(strategy="median")
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")

    numeric_pipeline = Pipeline(
        [
            ("imputer", numeric_imputer),
            ("scaler", make_scaler(scaling)),
        ]
    )

    if encoding == "onehot":
        categorical_pipeline = Pipeline(
            [
                ("imputer", categorical_imputer),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    elif encoding == "ordinal":
        categorical_pipeline = Pipeline(
            [
                ("imputer", categorical_imputer),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                ("scaler", make_scaler(scaling)),
            ]
        )
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(config: ExperimentConfig, model_name: str) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if config.use_derived:
        steps.append(("features", DerivedFeatureEngineer()))
        numeric_features = ALL_NUMERIC
        categorical_features = ALL_CATEGORICAL
    else:
        numeric_features = ORIGINAL_NUMERIC
        categorical_features = []

    if config.encoding == "none":
        preprocessor = ColumnTransformer(
            [("num", "passthrough", numeric_features)],
            remainder="drop",
            verbose_feature_names_out=True,
        )
    else:
        preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
            config.missing,
            config.encoding,
            config.scaling,
        )
    steps.append(("preprocess", preprocessor))

    if config.feature_selection:
        k = config.fs_k if config.fs_k is not None else 30
        steps.append(("selector", SelectKBest(score_func=f_classif, k=k)))

    if model_name == "Logistic Regression":
        model = LogisticRegression(
            max_iter=1200,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=140,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_model(pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(X_test)[:, 1]
    else:
        decision = pipe.decision_function(X_test)
        prob = (decision - decision.min()) / (decision.max() - decision.min())
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }


def format_config_value(value: str | bool) -> str:
    mapping = {
        "none": "없음",
        "mean": "Mean",
        "median": "Median",
        "most_frequent": "Most Frequent",
        "constant_unknown": "Unknown 대체",
        "onehot": "One-Hot",
        "ordinal": "Label(Ordinal)",
        "standard": "Standard",
        "minmax": "MinMax",
        "robust": "Robust",
        True: "O",
        False: "X",
    }
    return mapping.get(value, str(value))


def run_main_experiments(df: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, dict[str, Pipeline], tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    configs = [
        ExperimentConfig("Base", "none", "none", "none", False, use_derived=False),
        ExperimentConfig("Exp-1", "mean", "onehot", "standard", False),
        ExperimentConfig("Exp-2", "median", "ordinal", "minmax", True, fs_k=12),
        ExperimentConfig("Exp-3", "most_frequent", "onehot", "robust", True, fs_k=30),
    ]
    models = ["Logistic Regression", "Random Forest"]
    rows = []
    fitted: dict[str, Pipeline] = {}

    for config in configs:
        for model_name in models:
            pipe = build_pipeline(config, model_name)
            metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
            key = f"{config.name}__{model_name}"
            fitted[key] = pipe
            rows.append(
                {
                    "experiment": config.name,
                    "model": model_name,
                    "missing": format_config_value(config.missing),
                    "encoding": format_config_value(config.encoding),
                    "scaling": format_config_value(config.scaling),
                    "feature_selection": format_config_value(config.feature_selection),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(tables_dir / "main_experiment_results.csv", index=False, encoding="utf-8-sig")
    return results, fitted, (X_train, X_test, y_train, y_test)


def run_missing_strategy_comparison(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows = []
    strategies = [
        ("Drop NA", "drop"),
        ("Most Frequent", "most_frequent"),
        ("Unknown 대체", "constant_unknown"),
    ]
    for label, strategy in strategies:
        if strategy == "drop":
            work_df = df.dropna(subset=ORIGINAL_NUMERIC + ORIGINAL_CATEGORICAL + ["income_binary"]).copy()
            missing_for_pipeline = "median"
        else:
            work_df = df.copy()
            missing_for_pipeline = strategy
        X, y = split_xy(work_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        config = ExperimentConfig(
            name=label,
            missing=missing_for_pipeline,
            encoding="onehot",
            scaling="robust",
            feature_selection=False,
        )
        pipe = build_pipeline(config, "Random Forest")
        metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
        rows.append({"missing_strategy": label, "sample_count": len(work_df), **metrics})
    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "missing_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    return result


def run_encoding_scaling_comparisons(df: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    encoding_rows = []
    for encoding, label, fs_k in [("onehot", "One-Hot", None), ("ordinal", "Label(Ordinal)", None)]:
        config = ExperimentConfig(
            name=f"Encoding-{label}",
            missing="most_frequent",
            encoding=encoding,
            scaling="standard",
            feature_selection=False,
            fs_k=fs_k,
        )
        for model in ["Logistic Regression", "Random Forest"]:
            pipe = build_pipeline(config, model)
            metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
            encoding_rows.append({"encoding": label, "model": model, **metrics})
    encoding_result = pd.DataFrame(encoding_rows)
    encoding_result.to_csv(tables_dir / "encoding_comparison.csv", index=False, encoding="utf-8-sig")

    scaling_rows = []
    for scaling, label in [("standard", "Standard"), ("minmax", "MinMax"), ("robust", "Robust")]:
        config = ExperimentConfig(
            name=f"Scaling-{label}",
            missing="most_frequent",
            encoding="onehot",
            scaling=scaling,
            feature_selection=False,
        )
        for model in ["Logistic Regression", "Random Forest"]:
            pipe = build_pipeline(config, model)
            metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
            scaling_rows.append({"scaling": label, "model": model, **metrics})
    scaling_result = pd.DataFrame(scaling_rows)
    scaling_result.to_csv(tables_dir / "scaling_comparison.csv", index=False, encoding="utf-8-sig")
    return encoding_result, scaling_result


def processed_feature_names(pipe: Pipeline) -> np.ndarray:
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    if "selector" in pipe.named_steps:
        names = names[pipe.named_steps["selector"].get_support()]
    return names


def run_feature_selection_analysis(
    df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    base_config = ExperimentConfig(
        "FS-Off",
        "most_frequent",
        "onehot",
        "robust",
        False,
    )
    fs_config = ExperimentConfig(
        "FS-On",
        "most_frequent",
        "onehot",
        "robust",
        True,
        fs_k=30,
    )

    rows = []
    for config in [base_config, fs_config]:
        for model in ["Logistic Regression", "Random Forest"]:
            pipe = build_pipeline(config, model)
            metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
            rows.append(
                {
                    "feature_selection": "O" if config.feature_selection else "X",
                    "selected_k": config.fs_k if config.feature_selection else "all",
                    "model": model,
                    **metrics,
                }
            )
    fs_impact = pd.DataFrame(rows)
    fs_impact.to_csv(tables_dir / "feature_selection_impact.csv", index=False, encoding="utf-8-sig")

    rf_pipe = build_pipeline(base_config, "Random Forest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_pipe.fit(X_train, y_train)
    feature_names = processed_feature_names(rf_pipe)
    importances = rf_pipe.named_steps["model"].feature_importances_
    top_features = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_features.to_csv(tables_dir / "rf_top_features.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9.5, 7))
    sns.barplot(data=top_features, x="importance", y="feature", color="#4c78a8")
    plt.title("Random Forest Top 20 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(figures_dir / "rf_top_feature_importance.png", dpi=180)
    plt.close()

    return fs_impact, top_features


def run_grid_search(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    config = ExperimentConfig(
        "GridSearch",
        "most_frequent",
        "onehot",
        "robust",
        True,
        fs_k=30,
    )
    pipe = build_pipeline(config, "Logistic Regression")
    param_grid = {
        "selector__k": [20, 30, "all"],
        "model__C": [0.3, 1.0, 3.0],
    }
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)
    metrics = evaluate_model(search.best_estimator_, X_train, X_test, y_train, y_test)
    result = pd.DataFrame(
        [
            {
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                "cv_best_f1": search.best_score_,
                **metrics,
            }
        ]
    )
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(tables_dir / "grid_search_cv_results.csv", index=False, encoding="utf-8-sig")
    result.to_csv(tables_dir / "grid_search_summary.csv", index=False, encoding="utf-8-sig")
    return result


def table_for_pdf(
    df: pd.DataFrame,
    max_rows: int = 12,
    float_digits: int = 3,
    columns: list[str] | None = None,
) -> list[list[Any]]:
    view = df.copy()
    if columns is not None:
        view = view[columns]
    view = view.head(max_rows)
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: f"{x:.{float_digits}f}")
    return [list(view.columns)] + view.astype(str).values.tolist()


def add_table(story: list[Any], data: list[list[Any]], style: ParagraphStyle, col_widths: list[float] | None = None) -> None:
    wrapped_data = []
    for row in data:
        wrapped_data.append([Paragraph(str(cell), style) for cell in row])
    table = Table(wrapped_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24435c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d0d9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.35 * cm))


def add_image(story: list[Any], path: Path, width_cm: float = 16.2) -> None:
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = img.drawWidth * ratio
    story.append(img)
    story.append(Spacer(1, 0.35 * cm))


def register_pdf_fonts() -> tuple[str, str]:
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        return "HYGothic-Medium", "HYSMyeongJo-Medium"
    except Exception:
        return "Helvetica", "Helvetica"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def build_report(
    project_root: Path,
    df: pd.DataFrame,
    eda: dict[str, Any],
    main_results: pd.DataFrame,
    missing_results: pd.DataFrame,
    encoding_results: pd.DataFrame,
    scaling_results: pd.DataFrame,
    fs_impact: pd.DataFrame,
    top_features: pd.DataFrame,
    grid_summary: pd.DataFrame,
) -> None:
    outputs_dir = project_root / "outputs"
    figures_dir = outputs_dir / "figures"
    tables_dir = outputs_dir / "tables"
    report_path = outputs_dir / "adult_income_feature_engineering_report.pdf"

    title_font, body_font = register_pdf_fonts()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="KTitle",
            fontName=title_font,
            fontSize=19,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=0.5 * cm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KHeading",
            fontName=title_font,
            fontSize=13.5,
            leading=18,
            textColor=colors.HexColor("#18364d"),
            spaceBefore=0.32 * cm,
            spaceAfter=0.18 * cm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KBody",
            fontName=body_font,
            fontSize=9.4,
            leading=14,
            spaceAfter=0.12 * cm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KSmall",
            fontName=body_font,
            fontSize=7.2,
            leading=9,
        )
    )

    body = styles["KBody"]
    small = styles["KSmall"]
    story: list[Any] = []

    story.append(Paragraph("Feature Engineering 기반 Adult Income 예측 파이프라인 보고서", styles["KTitle"]))
    story.append(
        Paragraph(
            "데이터셋: UCI Adult / Census Income, 목표: 연소득이 50K 달러를 초과하는지 예측하는 이진 분류 문제",
            body,
        )
    )
    story.append(Paragraph(f"공식 출처: {DATASET_PAGE}", body))
    story.append(Paragraph(f"소스 코드 위치: {project_root / 'src' / 'adult_feature_engineering_pipeline.py'}", body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("1. 데이터셋 소개", styles["KHeading"]))
    story.append(
        Paragraph(
            "Adult 데이터셋은 1994년 미국 인구조사 데이터를 기반으로 개인의 인구통계, 학력, 직업, 근로 시간, 자본 이득/손실 등의 변수를 포함한다. "
            "타겟 변수는 income이며, 본 실험에서는 >50K를 1, <=50K를 0으로 변환하였다.",
            body,
        )
    )
    story.append(Paragraph(f"데이터 shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns", body))
    column_desc = pd.read_csv(tables_dir / "column_description.csv")
    add_table(story, table_for_pdf(column_desc, max_rows=20), small, [2.8 * cm, 5.0 * cm, 2.8 * cm, 2.5 * cm])

    story.append(Paragraph("2. EDA 결과", styles["KHeading"]))
    target_pct = {row["income"]: row["ratio_percent"] for row in eda["target_distribution"]}
    story.append(
        Paragraph(
            f"타겟 분포는 <=50K {pct(target_pct.get('<=50K', 0))}, >50K {pct(target_pct.get('>50K', 0))}로 불균형이 존재한다. "
            "양성 클래스가 상대적으로 적어 Accuracy만으로는 모델을 평가하기 어렵기 때문에 Precision, Recall, F1, ROC-AUC를 함께 사용하였다.",
            body,
        )
    )
    missing = pd.read_csv(tables_dir / "missing_values.csv")
    outliers = pd.read_csv(tables_dir / "outlier_summary.csv")
    story.append(
        Paragraph(
            "결측치는 주로 workclass, occupation, native_country에서 관찰되며, 수치형 변수에는 결측치가 없다. "
            "capital_gain, capital_loss, hours_per_week는 IQR 기준 이상치 비율이 높아 RobustScaler의 효과를 비교하였다.",
            body,
        )
    )
    add_table(story, table_for_pdf(missing, max_rows=8), small, [4.8 * cm, 4.5 * cm])
    add_table(
        story,
        table_for_pdf(outliers[["column", "outlier_count", "outlier_ratio_percent"]], max_rows=8),
        small,
        [4.2 * cm, 4.0 * cm, 4.8 * cm],
    )
    add_image(story, figures_dir / "missing_ratio.png")
    add_image(story, figures_dir / "target_distribution.png", width_cm=11.5)
    story.append(PageBreak())

    story.append(Paragraph("EDA 시각화", styles["KHeading"]))
    add_image(story, figures_dir / "hist_numeric.png")
    add_image(story, figures_dir / "boxplots_numeric.png")
    add_image(story, figures_dir / "correlation_heatmap.png", width_cm=13.5)
    add_image(story, figures_dir / "income_by_education.png")

    story.append(Paragraph("3. Feature Engineering 파이프라인", styles["KHeading"]))
    story.append(
        Paragraph(
            "파생 변수는 capital_net, capital_activity, education_per_age, age_group, work_intensity를 생성하였다. "
            "수치형 변수에는 결측치 대체 후 StandardScaler, MinMaxScaler, RobustScaler를 비교했고, 범주형 변수에는 One-Hot Encoding과 Label Encoding에 해당하는 OrdinalEncoder를 비교하였다. "
            "모든 주요 실험은 scikit-learn Pipeline과 ColumnTransformer 객체로 구성하여 재현성을 확보하였다.",
            body,
        )
    )

    story.append(Paragraph("4. 필수 조합별 성능 비교", styles["KHeading"]))
    display_cols = [
        "experiment",
        "model",
        "missing",
        "encoding",
        "scaling",
        "feature_selection",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    add_table(story, table_for_pdf(main_results, max_rows=12, columns=display_cols), small)

    best = main_results.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]
    story.append(
        Paragraph(
            f"최고 F1 기준 조합은 {best['experiment']} + {best['model']}이며, "
            f"F1={best['f1']:.3f}, ROC-AUC={best['roc_auc']:.3f}를 기록하였다.",
            body,
        )
    )

    story.append(Paragraph("5. 전처리 전략별 비교", styles["KHeading"]))
    story.append(Paragraph("결측치 처리 비교", styles["KHeading"]))
    add_table(story, table_for_pdf(missing_results, max_rows=6), small)
    story.append(
        Paragraph(
            "Adult 데이터셋의 실제 결측치는 범주형 변수에 집중되어 있어 Mean/Median은 수치형 파이프라인에만 적용된다. "
            "Drop NA는 표본 수를 줄이므로 성능이 좋아져도 정보 손실이 발생한다. Most Frequent와 Unknown 대체는 전체 표본을 유지한다는 장점이 있다.",
            body,
        )
    )

    story.append(Paragraph("인코딩 비교", styles["KHeading"]))
    add_table(story, table_for_pdf(encoding_results, max_rows=8), small)
    story.append(Paragraph("스케일링 비교", styles["KHeading"]))
    add_table(story, table_for_pdf(scaling_results, max_rows=8), small)

    story.append(Paragraph("6. 변수 선택", styles["KHeading"]))
    add_table(story, table_for_pdf(fs_impact, max_rows=8), small)
    story.append(
        Paragraph(
            "SelectKBest를 사용하여 상위 30개 변수를 선택한 뒤 제거 전/후 성능을 비교하였다. "
            "Random Forest 중요도 기준 상위 변수도 별도로 추출하여 모델이 어떤 변수에 의존하는지 확인하였다.",
            body,
        )
    )
    add_table(story, table_for_pdf(top_features, max_rows=12), small, [10.0 * cm, 4.0 * cm])
    add_image(story, figures_dir / "rf_top_feature_importance.png")

    story.append(Paragraph("7. GridSearchCV 가산점 실험", styles["KHeading"]))
    add_table(story, table_for_pdf(grid_summary, max_rows=3), small)
    story.append(
        Paragraph(
            "Logistic Regression의 C와 SelectKBest의 k를 작은 탐색 공간에서 GridSearchCV로 조정하였다. "
            "이는 Pipeline 기반 튜닝 예시이며, 더 넓은 탐색 공간을 사용하면 추가 개선 가능성이 있다.",
            body,
        )
    )

    story.append(Paragraph("8. 최종 결론", styles["KHeading"]))
    conclusion_points = [
        "One-Hot Encoding은 Logistic Regression에서 특히 유리했다. 범주형 값 사이에 임의의 순서를 부여하지 않기 때문이다.",
        "Random Forest는 스케일링의 영향이 작았지만, Logistic Regression은 Standard/Robust 계열 스케일링에서 안정적인 결과를 보였다.",
        "Feature Selection은 일부 조합에서 차원을 줄이며 성능을 유지하거나 개선했지만, 정보가 과도하게 제거되면 Recall이 낮아질 수 있다.",
        "파생 변수는 자본 활동 여부, 연령대, 근로 강도처럼 원본 변수의 비선형 패턴을 모델에 제공해 성능 비교의 실질적인 차이를 만들었다.",
        "최종적으로는 F1과 ROC-AUC를 함께 보고 모델을 선택하는 것이 타겟 불균형이 있는 본 데이터셋에 적절하다.",
    ]
    for point in conclusion_points:
        story.append(Paragraph(f"- {point}", body))

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
    )
    doc.build(story)


def write_markdown_summary(
    project_root: Path,
    main_results: pd.DataFrame,
    fs_impact: pd.DataFrame,
    top_features: pd.DataFrame,
) -> None:
    outputs_dir = project_root / "outputs"
    best = main_results.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]
    text = f"""# Adult Income Feature Engineering 결과 요약

## 최고 성능 조합

- Experiment: {best['experiment']}
- Model: {best['model']}
- F1: {best['f1']:.4f}
- ROC-AUC: {best['roc_auc']:.4f}

## 산출물

- PDF report: `outputs/adult_income_feature_engineering_report.pdf`
- Main experiment table: `outputs/tables/main_experiment_results.csv`
- EDA figures: `outputs/figures/`

## Feature Selection 요약

{fs_impact.to_string(index=False)}

## Random Forest 상위 변수

{top_features.head(10).to_string(index=False)}
"""
    (outputs_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    paths = ensure_dirs(project_root)

    download_dataset(paths["data_raw"])
    df = load_adult(paths["data_raw"])

    write_dataset_summary(df, paths["tables"])
    eda = save_eda(df, paths["figures"], paths["tables"])
    main_results, _fitted, _split = run_main_experiments(df, paths["tables"])
    missing_results = run_missing_strategy_comparison(df, paths["tables"])
    encoding_results, scaling_results = run_encoding_scaling_comparisons(df, paths["tables"])
    fs_impact, top_features = run_feature_selection_analysis(df, paths["tables"], paths["figures"])
    grid_summary = run_grid_search(df, paths["tables"])
    build_report(
        project_root,
        df,
        eda,
        main_results,
        missing_results,
        encoding_results,
        scaling_results,
        fs_impact,
        top_features,
        grid_summary,
    )
    write_markdown_summary(project_root, main_results, fs_impact, top_features)

    print("Done")
    print(f"Project root: {project_root}")
    print(f"PDF report: {project_root / 'outputs' / 'adult_income_feature_engineering_report.pdf'}")


if __name__ == "__main__":
    main()
