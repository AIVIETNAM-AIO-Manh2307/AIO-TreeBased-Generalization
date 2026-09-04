import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb


from preprocessing import load_and_clean, split_by_dataset, build_preprocessor

#Read data
df = load_and_clean("heart_disease_uci.csv")
datasets = split_by_dataset(df, feature_set="full")

X_cleveland, y_cleveland = datasets["Cleveland"]
X_hungary, y_hungary = datasets["Hungary"]
X_switzerland, y_switzerland = datasets["Switzerland"]
X_va, y_va = datasets["VA Long Beach"]

X_train_cle, X_test_cle, y_train_cle, y_test_cle = train_test_split(
    X_cleveland, y_cleveland, test_size=0.2, random_state=42
)

preprocessor = build_preprocessor(feature_set="full")


X_train_cle_processed = preprocessor.fit_transform(X_train_cle)

X_test_cle_processed = preprocessor.transform(X_test_cle)
X_hungary_processed = preprocessor.transform(X_hungary)
X_switzerland_processed = preprocessor.transform(X_switzerland)
X_va_processed = preprocessor.transform(X_va)


def train_and_evaluate(model_name, model):
    print(f"\n{'='*40}\n Model: {model_name}")
    
    # Huấn luyện
    model.fit(X_train_cle_processed, y_train_cle)
    
    # Dự đoán (Bắt buộc average='binary')
    f1_cleveland = f1_score(y_test_cle, model.predict(X_test_cle_processed), average='binary')
    f1_hungary = f1_score(y_hungary, model.predict(X_hungary_processed), average='binary')
    f1_switzerland = f1_score(y_switzerland, model.predict(X_switzerland_processed), average='binary')
    f1_va = f1_score(y_va, model.predict(X_va_processed), average='binary')
    
    # Tính thống kê chéo
    mean_cross = (f1_hungary + f1_switzerland + f1_va) / 3
    mean_drop = f1_cleveland - mean_cross

    worst_target = min(f1_hungary, f1_switzerland, f1_va)

    print(f"- In-Domain F1 (Cleveland): {f1_cleveland:.4f}")
    print(f"- Hungary F1              : {f1_hungary:.4f}")
    print(f"- Switzerland F1          : {f1_switzerland:.4f}")
    print(f"- Long Beach VA F1        : {f1_va:.4f}")
    print(f"-> MEAN CROSS F1          : {mean_cross:.4f}")
    print(f"-> MEAN DROP    : {mean_drop:.4f}")
    return [model_name, f1_cleveland, f1_hungary, f1_switzerland, f1_va, mean_cross, mean_drop, worst_target]


models = {
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1)
}

results = []
for name, model in models.items():
    res = train_and_evaluate(name, model)
    results.append(res)

columns = [
    "Model", 
    "In-Domain F1", 
    "Hungary F1",       
    "Switzerland F1",   
    "Long Beach F1",    
    "Mean Cross F1", 
    "Performance Drop", 
    "Worst Target F1"
]

df_results = pd.DataFrame(results, columns=columns)

final_baseline_matrix = df_results[["Model", "In-Domain F1", "Mean Cross F1", "Performance Drop", "Worst Target F1"]]

print("\n" + "="*60)
print("BẢNG BASELINE MATRIX CỦA BOOSTING")
print("="*60)
print(final_baseline_matrix.to_markdown(index=False))

