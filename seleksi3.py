import pandas as pd
import matplotlib.pyplot as plt

# Baca data
df = pd.read_csv("train.csv")

# Ubah kolom 'date' ke datetime
df['date'] = pd.to_datetime(df['date'])

# Tambahkan kolom bulan dan hari dalam minggu
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Senin, 6=Minggu

# Agregasi
monthly_avg = df.groupby('month')['electricity_consumption'].mean()
weekly_avg = df.groupby('day_of_week')['electricity_consumption'].mean()

# Visualisasi
plt.figure(figsize=(12, 5))

# Plot bulanan
plt.subplot(1, 2, 1)
monthly_avg.plot(kind='bar', color='skyblue')
plt.title('Rata-rata Konsumsi Listrik per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Konsumsi Listrik')

# Plot mingguan
plt.subplot(1, 2, 2)
weekly_avg.plot(kind='bar', color='salmon')
plt.title('Rata-rata Konsumsi Listrik per Hari dalam Minggu')
plt.xlabel('Hari (0=Senin)')
plt.ylabel('Konsumsi Listrik')

plt.tight_layout()
plt.show()
