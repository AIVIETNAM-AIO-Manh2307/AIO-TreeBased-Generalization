# Preprocessing Pipeline — UCI Heart Disease
---

## 1.
`preprocessing.py`:

1. Làm sạch các lỗi dữ liệu.
2. Tạo biến mục tiêu nhị phân từ nhãn đa lớp gốc.
3. Xây dựng **2 bộ đặc trưng** (full / reduced) để phục vụ so sánh.
4. Cung cấp cơ chế tách dữ liệu theo từng nguồn (domain).

---

## 2. Phương pháp xử lý

### 2.1. Chuẩn hoá giá trị lỗi thành `NaN`

| Cột | Điều kiện lỗi | Số dòng bị ảnh hưởng | Lý do |
|---|---|---:|---|
| `chol` | `== 0` | **202** | Cholesterol = 0 không có ý nghĩa sinh lý. |
| `trestbps` | `== 0` | **60** | Huyết áp = 0 không có ý nghĩa, lỗi mã hoá missing. |
| `oldpeak` | `< 0` | **74** | ST depression âm không có ý nghĩa. |

### 2.2. Chuẩn hoá kiểu dữ liệu boolean (`fbs`, `exang`)

Cột đọc từ CSV ở dạng chuỗi `"TRUE"/"FALSE"` được ép về kiểu `boolean`.

### 2.3. Biến mục tiêu: chuyển từ đa lớp (0–4) sang nhị phân

```python
df["target"] = (df["num"] > 0).astype(int)   # 0 = không bệnh, 1 = có bệnh (mọi mức độ)
```

dataset **Hungary chỉ ghi nhận `num ∈ {0, 1}`** — không có bất kỳ mẫu nào ở mức 2/3/4.
- Train trên Hungary → mô hình không bao giờ học được lớp 2/3/4.
- Test trên Hungary → nhãn thật không có lớp 2/3/4 để đối chiếu.

Sau khi nhị phân, **cả 4 nguồn đều có đầy đủ 2 lớp `{0, 1}`**.

### 2.4. Hai bộ đặc trưng (`full` / `reduced`)

| Bộ | Đặc trưng | Số lượng |
|---|---|---|
| **full** | `age, trestbps, chol, thalch, oldpeak, ca, sex, cp, fbs, restecg, exang, slope, thal` | 13 |
| **reduced** | `age, trestbps, chol, thalch, oldpeak, sex, cp, fbs, restecg, exang` | 10 |

**Lý do:** EDA cho thấy `ca`, `thal`, `slope` missing 90–99% ở Hungary/Switzerland/VA Long Beach, chỉ đầy đủ ở Cleveland. Nếu dùng chung bộ `full` cho cross-dataset:
- Khi **train trên Cleveland → test trên nguồn khác**: mô hình học có thể phụ thuộc mạnh vào `ca`/`thal` (2 đặc trưng tương quan cao nhất với target, r≈0.52), nhưng tập test gần như toàn `NaN` ở các cột này → giá trị impute (median/mode) sẽ chiếm hầu hết, biến đặc trưng "quan trọng nhất" thành hằng số nhiễu.
- Bộ `reduced`: đặc trưng đủ dữ liệu thật ở cả 4 nguồn, generalization đo được phản ánh đúng khả năng học covariate shift chứ không bị nhiễu bởi missing-by-domain.

### 2.5. Impute & encode đặt **bên trong** `ColumnTransformer`, không impute trước khi split 

```python
numeric_pipe     = SimpleImputer(strategy="median")
categorical_pipe = SimpleImputer(strategy="most_frequent") + OrdinalEncoder
```

### 2.6. Tách theo nguồn (`split_by_dataset`)

Hàm trả về `{source_name: (X, y)}` cho từng nguồn, giữ X ở dạng **chưa impute/encode** — việc này cố ý để dành riêng bước fit-transform cho vòng lặp huấn luyện sau (leave-one-dataset-out), tránh preprocessing "cứng" trước khi biết domain nào sẽ là train/test trong từng fold.
