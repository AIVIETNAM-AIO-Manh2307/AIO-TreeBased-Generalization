"""
train.py — Cross-Dataset Generalization Baseline

Thiết kế: train trên Cleveland (source domain), đánh giá trên:
  - phần Cleveland giữ lại (in-domain test)
  - Hungary / Switzerland / VA Long Beach (unseen target domains)

Gộp từ train_single_bagging.py + train_boosting.py để in ĐÚNG MỘT
bảng Baseline Matrix cho cả 6 model.
"""

import sys
import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
import xgboost as xgb


from preprocessing import load_and_clean, split_by_dataset, build_preprocessor

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ----------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------
CSV_PATH = "heart_disease_uci.csv"
FEATURE_SET = "full"
SOURCE = "Cleveland"
TARGETS = ["Hungary", "Switzerland", "VA Long Beach"]
# Tên hiển thị trong bảng kết quả (theo đúng spec Baseline Matrix)
TARGET_LABELS = {
    "Hungary": "Hungary",
    "Switzerland": "Switzerland",
    "VA Long Beach": "Long Beach",
}
TEST_SIZE = 0.2
SEED = 42
RESULTS_CSV = "results.csv"


# ----------------------------------------------------------------------
# Chuẩn bị dữ liệu
# ----------------------------------------------------------------------
df = load_and_clean(CSV_PATH)
datasets = split_by_dataset(df, feature_set=FEATURE_SET)

X_source, y_source = datasets[SOURCE]

X_train, X_test, y_train, y_test = train_test_split(
    X_source,
    y_source,
    test_size=TEST_SIZE,
    random_state=SEED,
)

# Preprocessor CHỈ fit trên train Cleveland -> không leak sang test/target.
preprocessor = build_preprocessor(feature_set=FEATURE_SET)
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
target_data = {
    name: (preprocessor.transform(datasets[name][0]), datasets[name][1])
    for name in TARGETS
}


# ----------------------------------------------------------------------
# Danh sách model
# ----------------------------------------------------------------------
MODELS = {
    "Decision Tree": (
        "Single Tree",
        DecisionTreeClassifier(random_state=SEED),
    ),
    "Random Forest": (
        "Bagging",
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
    ),
    "AdaBoost": (
        "Boosting",
        AdaBoostClassifier(random_state=SEED),
    ),
    "Gradient Boosting": (
        "Boosting",
        GradientBoostingClassifier(random_state=SEED),
    ),
    "XGBoost": (
        "Boosting",
        xgb.XGBClassifier(random_state=SEED, eval_metric="logloss"),
    ),
    "LightGBM": (
        "Boosting",
        lgb.LGBMClassifier(random_state=SEED, verbose=-1),
    ),
}


# ----------------------------------------------------------------------
# Train & evaluate
# ----------------------------------------------------------------------
def train_and_evaluate(model_name, family, model):
    print(f"\n{'=' * 46}\n Model: {model_name}  [{family}]")

    model.fit(X_train_p, y_train)

    f1_in = f1_score(y_test, model.predict(X_test_p), average="binary")
    f1_targets = {
        name: f1_score(y_true, model.predict(X_t), average="binary")
        for name, (X_t, y_true) in target_data.items()
    }

    cross_scores = list(f1_targets.values())
    mean_cross = sum(cross_scores) / len(cross_scores)
    drop = f1_in - mean_cross
    rel_drop = drop / f1_in if f1_in > 0 else float("nan")
    worst_target = min(cross_scores)
    worst_name = min(f1_targets, key=f1_targets.get)

    print(f"- In-Domain F1 (Cleveland): {f1_in:.4f}")
    for name in TARGETS:
        print(f"- {TARGET_LABELS[name] + ' F1':<25s}: {f1_targets[name]:.4f}")
    print(f"-> MEAN CROSS F1          : {mean_cross:.4f}")
    print(f"-> PERFORMANCE DROP       : {drop:.4f}  ({rel_drop * 100:.1f}%)")
    print(f"-> WORST TARGET F1        : {worst_target:.4f}  ({TARGET_LABELS[worst_name]})")

    row = {
        "Family": family,
        "Model": model_name,
        "In-Domain F1": f1_in,
    }
    row.update({f"{TARGET_LABELS[name]} F1": f1_targets[name] for name in TARGETS})
    row.update({
        "Mean Cross F1": mean_cross,
        "Performance Drop": drop,
        "Relative Drop (%)": rel_drop * 100,
        "Worst Target F1": worst_target,
        "Worst Target": TARGET_LABELS[worst_name],
    })
    return row


results = [
    train_and_evaluate(name, family, model)
    for name, (family, model) in MODELS.items()
]
df_results = pd.DataFrame(results)


# ----------------------------------------------------------------------
# Bảng kết quả
# ----------------------------------------------------------------------
full_cols = (
    ["Family", "Model", "In-Domain F1"]
    + [f"{TARGET_LABELS[name]} F1" for name in TARGETS]
    + ["Mean Cross F1", "Performance Drop"]
)
summary_cols = [
    "Model",
    "In-Domain F1",
    "Mean Cross F1",
    "Performance Drop",
    "Relative Drop (%)",
    "Worst Target F1",
]

header = (
    f"BASELINE MATRIX  |  train={SOURCE}  |  feature_set={FEATURE_SET}  |  seed={SEED}"
)

print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
print("\n[1] Baseline Matrix — F1 theo từng domain\n")
print(df_results[full_cols].round(4).to_markdown(index=False))

print("\n[2] Performance vs Robustness\n")
print(df_results[summary_cols].round(4).to_markdown(index=False))

df_results.to_csv(RESULTS_CSV, index=False)
print(f"\nĐã lưu kết quả -> {RESULTS_CSV}")
