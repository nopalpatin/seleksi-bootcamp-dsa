import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("train.csv")

# --- 5A: Analisis saat suhu minimum < Q1 dan sinar matahari < rata-rata ---
q1_temp = df['temperature_2m_min'].quantile(0.25)
avg_sun = df['sunshine_duration'].mean()

filtered_df_5a = df[(df['temperature_2m_min'] < q1_temp) & (df['sunshine_duration'] < avg_sun)]
avg_consumption_per_cluster_5a = filtered_df_5a.groupby('cluster_id')['electricity_consumption'].mean()
print("Rata-rata konsumsi listrik per cluster (dengan suhu rendah dan sinar matahari minim):")
print(avg_consumption_per_cluster_5a)

# --- 5B: Deteksi Outlier ---
Q1 = df['electricity_consumption'].quantile(0.25)
Q3 = df['electricity_consumption'].quantile(0.75)
IQR = Q3 - Q1

outliers_df = df[(df['electricity_consumption'] < (Q1 - 1.5 * IQR)) | 
                 (df['electricity_consumption'] > (Q3 + 1.5 * IQR))]

top_outliers = outliers_df.sort_values(by='electricity_consumption', ascending=False).head(5)
print("\n5 Outlier Konsumsi Listrik Tertinggi:")
print(top_outliers[['cluster_id', 'date', 'electricity_consumption']])

# --- Visualisasi boxplot per cluster ---
plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster_id', y='electricity_consumption', data=df)
plt.title("Boxplot Konsumsi Listrik per Cluster (untuk deteksi outlier)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
