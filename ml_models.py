#!/usr/bin/env python3
"""
ml_models.py
Clash Royale — Machine Learning Models
DSA210 Term Project
Usage:
    python ml_models.py                     # uses data/features.csv
    python ml_models.py data/features.csv   # explicit path

"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_SPLITS     = 5     

# Resolve paths relative to this script's location, not the current working
# directory — this lets the script run from anywhere (Spyder F5, terminal, etc.)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)            # one level up from scripts/
DATA_PATH   = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_DIR, "data", "features.csv")
PLOTS_DIR   = os.path.join(PROJECT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save(fig, name):
    p = os.path.join(PLOTS_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


def sep(title):
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")


def load_and_prepare(path):
    """Load features.csv, handle missing values, encode categoricals."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Run collect_data.py first.")

    df = pd.read_csv(path)
    print(f"  Loaded: {len(df)} records, {df['outcome'].mean():.1%} win rate")

    # Target
    y = df["outcome"].astype(int)

    # Drop the target and the categorical column (numeric trophy_diff already used)
    feature_cols = [c for c in df.columns
                    if c not in ("outcome", "trophy_diff_category")]
    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            n_missing = X[col].isna().sum()
            X[col] = X[col].fillna(med)
            print(f"    imputed {n_missing} missing values in '{col}' with median={med:.2f}")

    print(f"  Final feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y, list(X.columns)


def evaluate_model(name, model, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, needs_scaling):
    """Train a model and return metric dict + predictions."""
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled  if needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba, model


def cross_validate(name, model, X, y, scaled_X=None, needs_scaling=False):
    """5-fold stratified cross-validation"""
    Xuse = scaled_X if needs_scaling else X
    cv   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, Xuse, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {name:<25}  CV accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
    return scores



def tune_decision_tree(X_train, y_train):
    """
    'vary max_depth in the range 5 to 30 and min_samples_split between 20 and 100'.
    """
    sep("HYPERPARAMETER TUNING — Decision Tree (GridSearchCV)")
    param_grid = {
        "max_depth":         [5, 10, 15, 20, 25, 30, None],
        "min_samples_split": [2, 20, 50, 100],
        "min_samples_leaf":  [1, 5, 10],
    }
    base = DecisionTreeClassifier(random_state=RANDOM_STATE)
    cv   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    gs   = GridSearchCV(base, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)

    print(f"  Best CV accuracy : {gs.best_score_:.4f}")
    print(f"  Best parameters  : {gs.best_params_}")
    return gs.best_estimator_

def plot_confusion_matrices(results, y_test):
    """Confusion matrix per model"""
    n = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (name, info) in enumerate(results.items()):
        cm = confusion_matrix(y_test, info["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"],
                    cbar=False, square=True, annot_kws={"size": 13})
        axes[i].set_title(f"{name}\nAcc = {info['metrics']['Accuracy']:.3f}",
                          fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices on Test Set (n={})".format(len(y_test)),
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "08_confusion_matrices.png")


def plot_roc_curves(results, y_test):
    """ROC curves overlaid"""
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = sns.color_palette("tab10", len(results))

    for (name, info), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, info["y_proba"])
        auc = info["metrics"]["ROC-AUC"]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    save(fig, "09_roc_curves.png")


def plot_pr_curves(results, y_test):
    """Precision-Recall curves"""
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = sns.color_palette("tab10", len(results))
    baseline = y_test.mean()

    for (name, info), color in zip(results.items(), palette):
        prec, rec, _ = precision_recall_curve(y_test, info["y_proba"])
        ap = average_precision_score(y_test, info["y_proba"])
        ax.plot(rec, prec, color=color, lw=2, label=f"{name}  (AP = {ap:.3f})")

    ax.axhline(baseline, ls="--", color="gray", lw=1.2,
               label=f"Baseline (positive rate = {baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    save(fig, "10_pr_curves.png")


def plot_feature_importance(rf_model, feature_names):
    """Variable importance from Random Forest"""
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]
    feats = [feature_names[i] for i in order]
    vals  = importances[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(vals)), vals[::-1],
                   color=sns.color_palette("viridis", len(vals)))
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(feats[::-1])
    ax.set_xlabel("Importance (Gini decrease)")
    ax.set_title("Random Forest — Feature Importance", fontweight="bold")
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    save(fig, "11_feature_importance.png")


def plot_model_comparison(results):
    """Bar chart comparing models"""
    rows = []
    for name, info in results.items():
        m = info["metrics"]
        for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            rows.append({"Model": name, "Metric": metric, "Value": m[metric]})
    long_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long_df, x="Metric", y="Value", hue="Model",
                palette="tab10", ax=ax, edgecolor="white")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="0.5 baseline")
    ax.set_title("Model Comparison Across All Metrics (Test Set)",
                 fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout()
    save(fig, "12_model_comparison.png")


def main():
    print("=" * 60)
    print("  Clash Royale — Machine Learning Models  (DSA210)")
    print("=" * 60)

    # ── Load & prepare ────────────────────────────────────────────────────
    sep("DATA PREPARATION")
    X, y, feature_names = load_and_prepare(DATA_PATH)

    # ── Train/test split─────────────────────────────────────────
    sep("TRAIN / TEST SPLIT  (80 / 20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} samples ({y_train.mean():.1%} win)")
    print(f"  Test : {len(X_test)} samples ({y_test.mean():.1%} win)")

    # ── Standardize (required for KNN) ───────────────────────────
    sep("FEATURE STANDARDIZATION  (StandardScaler — Week 8)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_full_scaled  = scaler.fit_transform(X)  # for full-data CV
    print("  Features standardized to zero mean, unit variance.")

    # ── Cross-validation on training set─────────────────────────
    sep("5-FOLD CROSS-VALIDATION")

    cv_models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        ("KNN (k=5)",           KNeighborsClassifier(n_neighbors=5),                          True),
        ("Decision Tree",       DecisionTreeClassifier(random_state=RANDOM_STATE),            False),
        ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), False),
        ("Gradient Boosting",   GradientBoostingClassifier(random_state=RANDOM_STATE),        False),
    ]
    for name, m, needs_scale in cv_models:
        cross_validate(name, m, X_train, y_train,
                       scaled_X=X_train_scaled, needs_scaling=needs_scale)

    # ── KNN: try multiple k─────────────────
    sep("KNN — CHOOSING K  ('Best K for Loan Data')")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_k, best_score = 1, 0
    print(f"  {'k':>3}  {'CV-acc':>7}  {'CV-AUC':>7}")
    for k in [1, 3, 5, 7, 11, 15, 21, 31]:
        m = KNeighborsClassifier(n_neighbors=k)
        acc = cross_val_score(m, X_train_scaled, y_train, cv=cv, scoring="accuracy").mean()
        auc = cross_val_score(m, X_train_scaled, y_train, cv=cv, scoring="roc_auc").mean()
        print(f"  {k:>3}  {acc:>7.4f}  {auc:>7.4f}")
        if auc > best_score:
            best_score, best_k = auc, k
    print(f"  → Best k = {best_k}  (CV-AUC = {best_score:.4f})")

    # ── Hyperparameter tuning ─────────────────────
    best_tree = tune_decision_tree(X_train, y_train)

    # ── Final models (trained on full training set) ───────────────────────
    sep("TRAINING FINAL MODELS")
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, C=1.0,
                                                   random_state=RANDOM_STATE), True),
        "KNN (k={})".format(best_k): (KNeighborsClassifier(n_neighbors=best_k), True),
        "Decision Tree (tuned)":     (best_tree,                                 False),
        "Random Forest":             (RandomForestClassifier(n_estimators=200,
                                          max_features="sqrt",
                                          random_state=RANDOM_STATE),            False),
        "Gradient Boosting":         (GradientBoostingClassifier(n_estimators=200,
                                          learning_rate=0.05, max_depth=3,
                                          random_state=RANDOM_STATE),            False),
    }

    # Voting ensemble
    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ],
        voting="soft",
    )
    models["Voting (soft)"] = (voting, False)

    # Train + evaluate
    results = {}
    for name, (mdl, needs_scale) in models.items():
        metrics, y_pred, y_proba, fitted = evaluate_model(
            name, mdl, X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled, needs_scale
        )
        results[name] = {"model": fitted, "metrics": metrics,
                         "y_pred": y_pred, "y_proba": y_proba}

    # ── Results table ─────────────────────────────────────────────────────
    sep("FINAL TEST-SET RESULTS ")
    rdf = pd.DataFrame([r["metrics"] for r in results.values()]).set_index("Model")
    print(rdf.round(4).to_string())

    best_model_name = rdf["ROC-AUC"].idxmax()
    print(f"\n  → Best model by ROC-AUC: {best_model_name}  "
          f"(AUC = {rdf.loc[best_model_name, 'ROC-AUC']:.4f})")

    # ── Detailed report for best model ────────────────────────────────────
    sep(f"CLASSIFICATION REPORT — {best_model_name}")
    print(classification_report(
        y_test, results[best_model_name]["y_pred"],
        target_names=["Loss", "Win"], digits=4,
    ))

    # ── Logistic Regression coefficients (interpretation, Week 9a) ────────
    sep("LOGISTIC REGRESSION — COEFFICIENT INTERPRETATION")
    lr = results["Logistic Regression"]["model"]
    coefs = pd.DataFrame({
        "feature":     feature_names,
        "coefficient": lr.coef_[0],
        "abs_coef":    np.abs(lr.coef_[0]),
    }).sort_values("abs_coef", ascending=False)
    print("  (Features were standardized; coefficients comparable in magnitude.)")
    print(coefs[["feature", "coefficient"]].to_string(index=False))

    # ── Visualizations ────────────────────────────────────────────────────
    sep("GENERATING PLOTS")
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_feature_importance(results["Random Forest"]["model"], feature_names)
    plot_model_comparison(results)

    sep("DONE  —  plots saved to plots/")


if __name__ == "__main__":
    main()
