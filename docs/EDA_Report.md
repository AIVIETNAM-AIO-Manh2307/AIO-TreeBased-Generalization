# EDA — UCI Heart Disease Dataset
### Dataset: *https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data*
---
## 1. Tổng quan dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Số dòng | 920 |
| Số cột | 16 (14 feature + `id` + `num`) |
| Số nguồn (`dataset`) | 4: **Cleveland, Hungary, Switzerland, VA Long Beach** |
| Trùng lặp | 0 dòng trùng, 0 `id` trùng |
| Biến mục tiêu | `num` — 0 (không bệnh) đến 4 (mức độ nặng)|

**cột `dataset` tách thành 4 tập con độc lập theo location**, dùng làm 4 "domain" khác nhau để đánh giá cross-dataset generalization.

| Nguồn | N mẫu | Tuổi TB | % Nam | % Target dương tính | % missing `ca` | % missing `thal` | % missing `slope` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cleveland | 304 | 54.4 | 68.1 | 45.7 | 1.6 | 1.0 | 0.3 |
| Hungary | 293 | 47.9 | 72.4 | 36.2 | 99.0 | 90.4 | 64.5 |
| Switzerland | 123 | 55.3 | 91.9 | **93.5** | 95.9 | 42.3 | 13.8 |
| VA Long Beach | 200 | 59.4 | 97.0 | 74.5 | 99.0 | 83.0 | 51.0 |

---

## 2. Missing values

![EDA Overview](eda_overview.png)

**Phát hiện chính:**
- `ca` (66.4%), `thal` (52.8%), `slope` (33.6%) missing rất nặng **toàn cục**, nhưng tập trung gần như hoàn toàn ở 3 nguồn ngoài Cleveland.
- Cleveland gần như đầy đủ dữ liệu (missing < 2% ở mọi cột).
- Hungary/Switzerland/VA Long Beach thiếu 90–99% giá trị ở `ca`, và 40–90% ở `thal`, `slope`.
- `fbs`, `oldpeak`, `trestbps`, `exang`, `thalch` missing nặng riêng ở VA Long Beach (~26–28%) và `fbs` missing 61% ở Switzerland.

**Kết luận:** Nếu dùng đủ 14 feature gốc, mô hình học trên Cleveland sẽ *không thể* generalize sang 3 nguồn còn lại vì các đặc trưng quan trọng nhất theo phân tích tương quan (`ca`, `thal`) gần như không tồn tại ở đó. Đây là một dạng "**missing-not-at-random theo domain**" — bản thân việc missing đã là một tín hiệu phân biệt nguồn dữ liệu, dễ khiến model overfit vào domain thay vì học đặc trưng bệnh lý thật.

→ **Giả thuyết**:2 kịch bản baseline riêng biệt:
1. Dùng tập feature đầy đủ, giới hạn train/test trong Cleveland (upper-bound, không cross-domain thật).
2. Dùng tập **feature con** ít missing ở cả 4 nguồn (`age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak`) để đánh giá cross-dataset generalization công bằng.

---

## 3. Label shift giữa các nguồn

Tỷ lệ dương tính (`num > 0`) chênh lệch rất lớn:
- Hungary: 36.2%
- Cleveland: 45.7%
- VA Long Beach: 74.5%
- Switzerland: **93.5%**

Đây là **prior probability shift** rõ rệt — một mô hình train trên Hungary (36% dương tính) rồi test trên Switzerland (93.5% dương tính) sẽ có accuracy/F1 bị bóp méo nếu không hiệu chỉnh threshold hoặc dùng metric phù hợp (nên ưu tiên AUROC, balanced accuracy, hoặc F1 theo từng class thay vì accuracy thô).

Ngoài ra, Switzerland chỉ có 123 mẫu và chỉ 8 mẫu âm tính — cần lưu ý khi tính toán confidence interval hoặc so sánh giữa các domain.

---

## 4. Covariate shift 

![EDA Detail](eda_detail.png)

Một số quan sát:
- **`thalch` (nhịp tim tối đa)**: giảm dần rõ rệt qua Cleveland → Hungary → Switzerland/VA Long Beach (trung vị ~150 → ~140 → ~120). Đây là covariate shift thực sự, không chỉ do missing.
- **`chol` (cholesterol)**: Switzerland có **100% giá trị `chol = 0`** (123/123 dòng) và VA Long Beach có 49/200 dòng `chol = 0`.Cholesterol = 0 không có ý nghĩa sinh lý —> **bắt buộc phải xử lý** (chuyển 0 → NaN) trước khi train.
- **`oldpeak` âm**: 12 dòng có `oldpeak < 0` (11 ở Switzerland, 1 ở VA).
- **Giới tính**: tỷ lệ nam giới tăng mạnh qua các nguồn — Cleveland 68%, Hungary 72%, Switzerland 92%, VA Long Beach 97%. Đây cũng là một dạng selection bias giữa các domain, ảnh hưởng đến các đặc trưng liên quan đến giới tính.
- **Loại đau ngực (`cp`)**: tỷ lệ "asymptomatic" cao hơn hẳn ở Switzerland (80%) so với Hungary (42%) — phân phối đặc trưng phân loại cũng lệch theo nguồn.

---

## 5. Tương quan với biến mục tiêu (trên Cleveland)

Top đặc trưng tương quan với `num`:
- `ca` (số mạch máu chính bị hẹp): r ≈ 0.52
- `oldpeak` (ST depression): r ≈ 0.51
- `thalch` (nhịp tim tối đa, tương quan âm): r ≈ -0.42
- `age`: r ≈ 0.23

Vấn đề: hai đặc trưng dự đoán mạnh nhất (`ca`, `oldpeak`) lại chính là hai đặc trưng missing nặng nhất ở 3 nguồn còn lại → càng củng cố việc cần thử nghiệm nhiều tập feature khác nhau khi đánh giá cross-dataset generalization.

---