from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

TARGET_COL_ORIGINAL = "num"   # nhãn gốc đa lớp 0-4 
TARGET_COL = "target"          # nhãn nhị phân: 0 = không bệnh, 1 = có bệnh
SOURCE_COL = "dataset"          # cột phân biệt 4 nguồn (Cleveland/Hungary/Switzerland/VA)
ID_COL = "id"

NUMERIC_FULL = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
NUMERIC_REDUCED = ["age", "trestbps", "chol", "thalch", "oldpeak"]

CATEGORICAL_FULL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
CATEGORICAL_REDUCED = ["sex", "cp", "fbs", "restecg", "exang"]

FULL_FEATURES = NUMERIC_FULL + CATEGORICAL_FULL
REDUCED_FEATURES = NUMERIC_REDUCED + CATEGORICAL_REDUCED

FEATURE_SETS = {
    "full": {
        "numeric": NUMERIC_FULL,
        "categorical": CATEGORICAL_FULL,
        "all": FULL_FEATURES,
    },
    "reduced": {
        "numeric": NUMERIC_REDUCED,
        "categorical": CATEGORICAL_REDUCED,
        "all": REDUCED_FEATURES,
    },
}

SOURCE_ORDER = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Đọc CSV gốc và chuẩn hoá các lỗi dữ liệu:
      - chol == 0        -> NaN  (100% Switzerland, ~25% VA Long Beach)
      - trestbps == 0     -> NaN  (giá trị không có ý nghĩa)
      - oldpeak < 0       -> NaN  (giá trị bất thường)
      - chuẩn hoá kiểu bool (fbs, exang) về 0/1
    """
    df = pd.read_csv(csv_path)

    df = df.copy()
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    df.loc[df["oldpeak"] < 0, "oldpeak"] = np.nan
    
    for col in ["fbs", "exang"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).astype("float")

    df[TARGET_COL] = (df[TARGET_COL_ORIGINAL] > 0).astype(int)

    return df


def build_preprocessor(feature_set: str = "full") -> ColumnTransformer:
    """
    Trả về ColumnTransformer gồm:
      - numeric   : SimpleImputer(median)
      - categorical: SimpleImputer(most_frequent) + OrdinalEncoder

    feature_set: "full" hoặc "reduced"
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set phải là 'full' hoặc 'reduced', nhận: {feature_set}")

    numeric_cols = FEATURE_SETS[feature_set]["numeric"]
    categorical_cols = FEATURE_SETS[feature_set]["categorical"]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def split_by_dataset(df: pd.DataFrame, feature_set: str = "full"):
    """
    Trả về dict {source_name: (X, y)} cho từng nguồn trong SOURCE_ORDER.
    X chỉ chứa các cột feature tương ứng với feature_set (chưa impute/encode).
    """
    cols = FEATURE_SETS[feature_set]["all"]
    result = {}
    for src in SOURCE_ORDER:
        sub = df[df[SOURCE_COL] == src]
        X = sub[cols].copy()
        y = sub[TARGET_COL].copy()
        result[src] = (X, y)
    return result


if __name__ == "__main__":
    CSV_PATH = "./heart_disease_uci.csv"
    df = load_and_clean(CSV_PATH)

    print("Shape sau khi làm sạch:", df.shape)
    print("\nSố NaN mới sinh ra sau chuẩn hoá lỗi (chol/trestbps/oldpeak):")
    print(df[["chol", "trestbps", "oldpeak"]].isnull().sum())

    print(f"\nPhân phối target NHỊ PHÂN ('{TARGET_COL}') toàn bộ (0=không bệnh, 1=có bệnh):")
    print(df[TARGET_COL].value_counts().sort_index())

    print(f"\nPhân phối target nhị phân theo từng nguồn:")
    print(pd.crosstab(df[SOURCE_COL], df[TARGET_COL]).reindex(SOURCE_ORDER))

    print("\n% dương tính (target=1) theo từng nguồn:")
    print((df.groupby(SOURCE_COL)[TARGET_COL].mean() * 100).round(1).reindex(SOURCE_ORDER))

    for fs in ["full", "reduced"]:
        print(f"\n--- feature_set = '{fs}' ---")
        print("Features:", FEATURE_SETS[fs]["all"])
        splits = split_by_dataset(df, fs)
        for src, (X, y) in splits.items():
            print(f"  {src:15s} X={X.shape}  y_classes={sorted(y.unique())}")