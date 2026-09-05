import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

CSV_PATH = './heart_disease_uci.csv'
SOURCE_ORDER = ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach']

df = pd.read_csv(CSV_PATH)
df['target_bin'] = (df['num'] > 0).astype(int)   # nhị phân hoá: 0 = không bệnh, 1 = có bệnh


def section_overview(df):
    print("=" * 70)
    print("1. TỔNG QUAN DỮ LIỆU")
    print("=" * 70)
    print("SHAPE:", df.shape)

    print("\nCOLUMNS & DTYPES:")
    print(df.dtypes)

    print("\nHEAD:")
    print(df.head())

    print("\nSỐ MẪU THEO NGUỒN (cột 'dataset'):")
    print(df['dataset'].value_counts())

    print("\nPHÂN PHỐI BIẾN MỤC TIÊU 'num' (0=không bệnh, 1-4=mức độ nặng):")
    print(df['num'].value_counts().sort_index())

    print("\nDÒNG TRÙNG LẶP:", df.duplicated().sum())
    print("ID TRÙNG LẶP:", df['id'].duplicated().sum())

def section_missing(df):
    print("\n" + "=" * 70)
    print("2. MISSING VALUES")
    print("=" * 70)

    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(1)
    summary = pd.DataFrame({'missing_count': miss, 'missing_pct': miss_pct})
    cols_with_na = summary[summary['missing_count'] > 0].index.tolist()

    print("MISSING TOÀN CỤC (count, %):")
    print(summary.loc[cols_with_na].sort_values('missing_pct', ascending=False))

    print("\nMISSING (%) THEO TỪNG NGUỒN DATASET — quan trọng cho cross-dataset:")
    missing_by_source = (
        df.groupby('dataset')[cols_with_na]
        .apply(lambda x: x.isnull().mean() * 100)
        .round(1)
        .reindex(SOURCE_ORDER)
    )
    print(missing_by_source)
    return cols_with_na

def section_stats(df):
    print("\n" + "=" * 70)
    print("3. THỐNG KÊ MÔ TẢ")
    print("=" * 70)

    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    print("DESCRIBE (numeric):")
    print(df[num_cols].describe().round(2))

    print("\nCATEGORICAL VALUE COUNTS:")
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    for c in cat_cols:
        print(f"\n--- {c} ---")
        print(df[c].value_counts(dropna=False))

    print("\nPHÂN PHỐI TARGET (num) THEO NGUỒN:")
    print(pd.crosstab(df['dataset'], df['num']).reindex(SOURCE_ORDER))

    print("\n% TARGET DƯƠNG TÍNH (num>0) THEO NGUỒN — phát hiện label shift:")
    print((df.groupby('dataset')['target_bin'].mean() * 100).round(1).reindex(SOURCE_ORDER))

    print("\nTUỔI THEO NGUỒN:")
    print(df.groupby('dataset')['age'].describe()[['mean', 'std', 'min', 'max']]
          .round(1).reindex(SOURCE_ORDER))

    print("\nGIỚI TÍNH (%) THEO NGUỒN:")
    print((pd.crosstab(df['dataset'], df['sex'], normalize='index') * 100)
          .round(1).reindex(SOURCE_ORDER))

def section_anomalies(df):
    print("\n" + "=" * 70)
    print("4. BẤT THƯỜNG / LỖI DỮ LIỆU KHẢ NGHI")
    print("=" * 70)

    print("Giá trị 0 đáng ngờ (khả năng là placeholder cho missing):")
    for col in ['trestbps', 'chol']:
        print(f"\n{col} == 0, theo nguồn:")
        print(df[df[col] == 0].groupby('dataset').size())

    print("\noldpeak < 0 (bất thường về mặt sinh lý):", (df['oldpeak'] < 0).sum())
    print(df[df['oldpeak'] < 0]['dataset'].value_counts())

    print("\nSố lượng giá trị non-null của ca/thal/slope theo nguồn:")
    for col in ['ca', 'thal', 'slope']:
        print(f"\n{col}:")
        print(df.groupby('dataset')[col].apply(lambda x: x.notna().sum()))

def plot_overview(df, out_path='eda_overview.png'):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) Số mẫu theo nguồn
    sizes = df['dataset'].value_counts().reindex(SOURCE_ORDER)
    axes[0, 0].bar(sizes.index, sizes.values,
                   color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    axes[0, 0].set_title('Số lượng mẫu theo nguồn dataset')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=20)

    # (2) % target dương tính theo nguồn (label shift)
    rates = (df.groupby('dataset')['target_bin'].mean() * 100).reindex(SOURCE_ORDER)
    axes[0, 1].bar(rates.index, rates.values,
                   color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    axes[0, 1].set_title('% Bệnh nhân dương tính (num>0) theo nguồn\n(Label shift giữa các dataset)')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].tick_params(axis='x', rotation=20)
    for i, v in enumerate(rates.values):
        axes[0, 1].text(i, v + 1, f"{v:.1f}%", ha='center')

    # (3) Heatmap missing theo nguồn
    cols_na = ['trestbps', 'chol', 'fbs', 'restecg', 'thalch',
               'exang', 'oldpeak', 'slope', 'ca', 'thal']
    miss_matrix = (df.groupby('dataset')[cols_na]
                   .apply(lambda x: x.isnull().mean() * 100)
                   .reindex(SOURCE_ORDER))
    sns.heatmap(miss_matrix, annot=True, fmt='.0f', cmap='Reds',
                ax=axes[0, 2], cbar_kws={'label': '% missing'})
    axes[0, 2].set_title('Tỷ lệ missing (%) theo cột & nguồn')

    # (4) Phân phối tuổi theo nguồn
    for src in SOURCE_ORDER:
        sns.kdeplot(df[df['dataset'] == src]['age'], label=src,
                    ax=axes[1, 0], fill=True, alpha=0.2)
    axes[1, 0].set_title('Phân phối tuổi theo nguồn')
    axes[1, 0].legend(fontsize=8)

    # (5) Phân phối cholesterol theo nguồn (lộ rõ lỗi chol=0)
    for src in SOURCE_ORDER:
        sns.kdeplot(df[df['dataset'] == src]['chol'], label=src,
                    ax=axes[1, 1], fill=True, alpha=0.2)
    axes[1, 1].set_title('Phân phối Cholesterol theo nguồn\n(lưu ý: chol=0 là dữ liệu lỗi ở Switzerland/VA)')
    axes[1, 1].legend(fontsize=8)

    # (6) Phân bố giới tính theo nguồn
    sex_ct = (pd.crosstab(df['dataset'], df['sex'], normalize='index') * 100).reindex(SOURCE_ORDER)
    sex_ct.plot(kind='bar', stacked=True, ax=axes[1, 2], color=['#C44E52', '#4C72B0'])
    axes[1, 2].set_title('Phân bố giới tính theo nguồn')
    axes[1, 2].set_ylabel('%')
    axes[1, 2].tick_params(axis='x', rotation=20)
    axes[1, 2].legend(title='Sex')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[Saved] {out_path}")

def plot_detail(df, out_path='eda_detail.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (1) Boxplot trestbps theo nguồn
    sns.boxplot(data=df, x='dataset', y='trestbps', order=SOURCE_ORDER,
                hue='dataset', legend=False, ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Resting Blood Pressure theo nguồn')
    axes[0, 0].tick_params(axis='x', rotation=20)

    # (2) Boxplot thalch (max heart rate) theo nguồn
    sns.boxplot(data=df, x='dataset', y='thalch', order=SOURCE_ORDER,
                hue='dataset', legend=False, ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Max Heart Rate (thalch) theo nguồn')
    axes[0, 1].tick_params(axis='x', rotation=20)

    # (3) Ma trận tương quan (chỉ Cleveland - tập đầy đủ nhất)
    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']
    corr = df[df['dataset'] == 'Cleveland'][num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=axes[1, 0], vmin=-1, vmax=1)
    axes[1, 0].set_title('Correlation matrix (Cleveland - dataset đầy đủ nhất)')

    # (4) Phân bố loại đau ngực (cp) theo nguồn
    cp_ct = (pd.crosstab(df['dataset'], df['cp'], normalize='index') * 100).reindex(SOURCE_ORDER)
    cp_ct.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Set3')
    axes[1, 1].set_title('Loại đau ngực (cp) theo nguồn (%)')
    axes[1, 1].tick_params(axis='x', rotation=20)
    axes[1, 1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {out_path}")

def section_summary_table(df):
    print("\n" + "=" * 70)
    print("7. BẢNG TỔNG HỢP THEO NGUỒN (dùng cho báo cáo)")
    print("=" * 70)

    summary = pd.DataFrame({
        'N_samples': df.groupby('dataset').size(),
        'Age_mean': df.groupby('dataset')['age'].mean().round(1),
        '%_Male': (df.groupby('dataset')['sex'].apply(lambda x: (x == 'Male').mean()) * 100).round(1),
        '%_target_positive': (df.groupby('dataset')['target_bin'].mean() * 100).round(1),
        '%_missing_ca': (df.groupby('dataset')['ca'].apply(lambda x: x.isnull().mean()) * 100).round(1),
        '%_missing_thal': (df.groupby('dataset')['thal'].apply(lambda x: x.isnull().mean()) * 100).round(1),
        '%_missing_slope': (df.groupby('dataset')['slope'].apply(lambda x: x.isnull().mean()) * 100).round(1),
    }).reindex(SOURCE_ORDER)

    print(summary.to_markdown())
    return summary

if __name__ == '__main__':
    section_overview(df)
    section_missing(df)
    section_stats(df)
    section_anomalies(df)
    plot_overview(df, 'eda_overview.png')
    plot_detail(df, 'eda_detail.png')
    section_summary_table(df)