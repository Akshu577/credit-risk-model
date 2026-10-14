"""Train and evaluate credit default models on the German Credit dataset (UCI / OpenML 'credit-g').

Usage: python src/train.py
Outputs: reports/metrics.md, reports/feature_importance.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEED = 42

# NumPy 2.x on macOS Accelerate emits spurious matmul RuntimeWarnings; results are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# The dataset's documented cost matrix: approving a bad borrower costs 5x
# more than rejecting a good one.
COST_FALSE_APPROVE = 5
COST_FALSE_REJECT = 1


def load_data():
    data = fetch_openml("credit-g", version=1, as_frame=True)
    X = data.frame.drop(columns="class")
    y = (data.frame["class"] == "bad").astype(int)  # 1 = default risk
    return X, y


def build_models(X):
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    linear_prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    tree_prep = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ])
    return {
        "Logistic Regression": Pipeline([
            ("prep", linear_prep),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", tree_prep),
            ("clf", HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=300,
                class_weight="balanced", random_state=SEED)),
        ]),
    }


def business_cost(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return fn * COST_FALSE_APPROVE + fp * COST_FALSE_REJECT


def best_threshold(y_true, proba):
    grid = np.linspace(0.05, 0.95, 91)
    costs = [business_cost(y_true, proba, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def main():
    REPORTS.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lines = ["# Credit Risk Model - Results", "",
             f"Dataset: German Credit (1,000 applicants, {y.mean():.0%} labelled bad risk). "
             "Test set: 25% stratified hold-out.", "",
             "| Model | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | Threshold | Cost @0.5 | Cost @tuned |",
             "|---|---|---|---|---|---|---|"]

    best_name, best_auc, best_model = None, -1, None
    for name, model in build_models(X).items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        # Tune the threshold on out-of-fold training predictions, never on the test set.
        oof = np.zeros(len(y_train))
        for tr, va in cv.split(X_train, y_train):
            m = build_models(X)[name].fit(X_train.iloc[tr], y_train.iloc[tr])
            oof[va] = m.predict_proba(X_train.iloc[va])[:, 1]
        thr = best_threshold(y_train.values, oof)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        lines.append(
            f"| {name} | {cv_auc.mean():.3f} ± {cv_auc.std():.3f} | {auc:.3f} | "
            f"{average_precision_score(y_test, proba):.3f} | {thr:.2f} | "
            f"{business_cost(y_test, proba, 0.5)} | {business_cost(y_test, proba, thr)} |")
        if auc > best_auc:
            best_name, best_auc, best_model = name, auc, model

    imp = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                                 n_repeats=20, random_state=SEED)
    importance = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("importance", ascending=False))
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)

    lines += ["", f"Best model by test ROC-AUC: **{best_name}**", "",
              "Cost = 5 x (bad borrowers approved) + 1 x (good borrowers rejected).", "",
              "## Top 8 drivers (permutation importance, ROC-AUC drop)", ""]
    lines += [f"- {r.feature}: {r.importance:.3f}" for r in importance.head(8).itertuples()]
    (REPORTS / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
