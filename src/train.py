"""Compare full-trace and early-trace compromise detection."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def _fit_experiment(
    features: pd.DataFrame,
    train_ids: set[str],
    test_ids: set[str],
    seed: int,
) -> tuple[RandomForestClassifier, dict[str, float], pd.Series, pd.Series]:
    train = features[features["session_id"].isin(train_ids)].copy()
    test = features[features["session_id"].isin(test_ids)].copy()
    feature_columns = [column for column in features.columns if column not in {"session_id", "label"}]
    X_train = train[feature_columns]
    X_test = test[feature_columns]
    y_train = (train["label"] != "benign").astype(int)
    y_test = (test["label"] != "benign").astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "test_sessions": int(len(y_test)),
        "test_compromise_rate": float(y_test.mean()),
    }
    return model, metrics, y_test, pd.Series(prediction, index=y_test.index)


def run_comparison(
    full_features: pd.DataFrame,
    early_features: pd.DataFrame,
    results_dir: Path,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Train separate full- and early-evidence models on the same session split."""
    sessions = full_features[["session_id", "label"]]
    train_ids, test_ids = train_test_split(
        sessions["session_id"],
        test_size=0.25,
        random_state=seed,
        stratify=(sessions["label"] != "benign").astype(int),
    )
    train_id_set, test_id_set = set(train_ids), set(test_ids)

    full_model, full_metrics, y_test, full_prediction = _fit_experiment(
        full_features, train_id_set, test_id_set, seed
    )
    _, early_metrics, _, _ = _fit_experiment(early_features, train_id_set, test_id_set, seed)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "full_metrics.json").write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    (results_dir / "early_metrics.json").write_text(json.dumps(early_metrics, indent=2), encoding="utf-8")
    comparison = pd.DataFrame(
        [
            {"evidence": "first 40%", **early_metrics},
            {"evidence": "full trace", **full_metrics},
        ]
    )
    comparison.to_csv(results_dir / "metrics_comparison.csv", index=False)

    feature_columns = [column for column in full_features.columns if column not in {"session_id", "label"}]
    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": full_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(results_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        full_prediction,
        display_labels=["benign", "compromised"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Full-trace compromise detection")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    top = importance.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance"], color="#00897B")
    ax.set(xlabel="Random Forest importance", title="Top observability features")
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    chart = comparison.set_index("evidence")[["roc_auc", "f1", "recall"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    chart.plot(kind="bar", ylim=(0, 1.05), color=["#355070", "#6D597A", "#B56576"], ax=ax)
    ax.set(xlabel="Available trace evidence", ylabel="Score", title="Early versus full-trace detection")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(results_dir / "early_vs_full.png", dpi=160)
    plt.close(fig)
    return {"full": full_metrics, "early": early_metrics}
