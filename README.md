# AIO-TreeBased-Generalization

**Đánh giá khả năng tổng quát hoá (generalization) của các mô hình tree-based (Decision Tree, Bagging, Boosting) trên bài toán chẩn đoán bệnh tim khi train và test trên các domain/nguồn dữ liệu khác nhau.**

> Project Conquer — Module 3
> Dataset: [UCI Heart Disease (Kaggle)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

---

## 1. Bài toán

Bộ dữ liệu UCI Heart Disease thực chất được gộp từ **4 bệnh viện/nguồn khác nhau** (Cleveland, Hungary, Switzerland, VA Long Beach), mỗi nguồn có quy trình thu thập, thiết bị đo và tỷ lệ bệnh nhân khác nhau. Trong thực tế, một mô hình chẩn đoán thường được huấn luyện tại một bệnh viện rồi triển khai ở nơi khác — đây chính là bài toán **cross-dataset (cross-domain) generalization**.

Project trả lời câu hỏi:
> Một mô hình tree-based học tốt trên Cleveland (nơi dữ liệu sạch và đầy đủ nhất) có còn giữ được hiệu năng khi áp dụng sang Hungary, Switzerland, VA Long Beach hay không? Và họ mô hình nào (Single Tree / Bagging / Boosting) chống chịu tốt nhất với domain shift?

---

## 2. Cấu trúc project

```
├── data/
│   └── heart_disease_uci.csv       
├── docs/
│   ├── EDA_Report.md            
│   ├── preprocessing.md
├── scripts/                      
│   ├── eda.py                      
│   ├── preprocessing.py
│   ├── train.py                    
│   ├── train_bagging.py
│   └── train_boosting.py
├── results/
│   └── results.csv                 
└── README.md                       
```

---

## 3. Khám phá dữ liệu (EDA)

Chi tiết đầy đủ trong [`EDA_Report.md`](EDA_Report.md). Một số phát hiện chính:

| Nguồn | N mẫu | Tuổi TB | % Nam | % Target dương tính | % missing `ca` | % missing `thal` | % missing `slope` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cleveland | 304 | 54.4 | 68.1 | 45.7 | 1.6 | 1.0 | 0.3 |
| Hungary | 293 | 47.9 | 72.4 | 36.2 | 99.0 | 90.4 | 64.5 |
| Switzerland | 123 | 55.3 | 91.9 | **93.5** | 95.9 | 42.3 | 13.8 |
| VA Long Beach | 200 | 59.4 | 97.0 | 74.5 | 99.0 | 83.0 | 51.0 |

- **Missing-not-at-random theo domain**: `ca`, `thal`, `slope` gần như chỉ tồn tại ở Cleveland (missing < 2%) nhưng thiếu 40–99% ở 3 nguồn còn lại — bản thân việc missing đã là tín hiệu phân biệt nguồn dữ liệu.
- **Label shift**: tỷ lệ dương tính chênh lệch rất lớn giữa các nguồn (36.2% → 93.5%), khiến accuracy thô không phản ánh đúng hiệu năng cross-domain; cần ưu tiên F1/AUROC theo từng lớp.
- **Covariate shift**: nhịp tim tối đa (`thalch`) giảm dần qua Cleveland → Hungary → Switzerland/VA; cholesterol (`chol`) có lỗi mã hoá nghiêm trọng (Switzerland 100% giá trị `chol = 0`); tỷ lệ nam giới tăng dần qua các nguồn (68% → 97%).
- **Tương quan với target** (đo trên Cleveland): `ca` (r≈0.52) và `oldpeak` (r≈0.51) là hai đặc trưng dự đoán mạnh nhất — nhưng cũng chính là hai đặc trưng missing nặng nhất ở các nguồn khác, gây khó khăn trực tiếp cho việc generalize.

---

## 4. Tiền xử lý dữ liệu

Chi tiết trong [`preprocessing.md`](preprocessing.md) / [`preprocessing.py`](preprocessing.py).

- **Sửa lỗi dữ liệu → NaN**: `chol == 0` (202 dòng), `trestbps == 0` (60 dòng), `oldpeak < 0` (74 dòng) — đều là giá trị không có ý nghĩa sinh lý.
- **Nhị phân hoá target**: `num > 0` → `target = 1` (có bệnh), vì Hungary chỉ ghi nhận `num ∈ {0,1}` nên không thể dùng bài toán đa lớp cho so sánh cross-domain công bằng.
- **Hai bộ đặc trưng** để so sánh:
  - `full` (13 đặc trưng, gồm cả `ca`, `thal`, `slope`)
  - `reduced` (10 đặc trưng, loại bỏ 3 cột missing nặng theo domain)
- **Impute & encode nằm trong `ColumnTransformer`**, chỉ `fit` trên tập train (Cleveland) để tránh data leakage sang các domain target.
- **`split_by_dataset`** trả về dữ liệu thô (chưa impute/encode) theo từng nguồn, phục vụ đánh giá leave-one-dataset-out.

---

## 5. Huấn luyện & Kết quả

**Thiết kế thực nghiệm** ([`train.py`](train.py)): train trên **Cleveland** (80/20 split), đánh giá trên phần Cleveland giữ lại (in-domain) và trên 3 domain chưa từng thấy: Hungary, Switzerland, VA Long Beach — dùng bộ đặc trưng `full`, F1-score (binary), `random_state=42`.

6 mô hình được so sánh theo 3 họ: **Single Tree**, **Bagging**, **Boosting**.

### Baseline Matrix

| Family | Model | In-Domain F1 | Hungary F1 | Switzerland F1 | Long Beach F1 | Mean Cross F1 | Performance Drop | Relative Drop (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Single Tree | Decision Tree | 0.806 | 0.426 | 0.543 | 0.463 | 0.477 | 0.329 | 40.8% |
| Bagging | Random Forest | 0.842 | 0.696 | 0.700 | 0.707 | 0.701 | 0.141 | 16.8% |
| Boosting | AdaBoost | 0.867 | 0.725 | 0.812 | 0.718 | 0.752 | 0.115 | 13.2% |
| Boosting | Gradient Boosting | 0.793 | 0.708 | 0.746 | 0.701 | 0.718 | 0.075 | **9.4%** |
| Boosting | XGBoost | 0.780 | 0.658 | 0.830 | 0.747 | 0.745 | 0.035 | **4.5%** |
| Boosting | LightGBM | 0.847 | 0.712 | 0.800 | 0.723 | 0.745 | 0.103 | 12.1% |


---

## 6. Hướng dẫn chạy project

```bash
# 1. Cài đặt thư viện
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm tabulate

# 2. Chạy EDA (xuất eda_overview.png, eda_detail.png)
python eda.py

# 3. Kiểm tra pipeline tiền xử lý
python preprocessing.py

# 4. Huấn luyện & xuất Baseline Matrix (results.csv)
python train.py
```

