import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import seaborn as sns

# Membuat dataset mainan
hair_color = np.array(['blonde', 'brunette', 'red', 'black', 'brunette', 'black', 'red', 'black'])
eye_color = np.array(['amber', 'gray', 'green', 'hazel', 'amber', 'gray', 'green', 'hazel'])
skin_color = np.array(['fair', 'brown', 'brown', 'brown', 'fair', 'brown', 'fair', 'fair'])
person = ['P1','P2','P3','P4','P5','P6','P7','P8']
data = pd.DataFrame({'person':person, 'hair_color':hair_color, 'eye_color':eye_color, 'skin_color':skin_color})
data = data.set_index('person')
print("Dataset awal:\n", data)

# Menyiapkan data untuk K-Modes
X = data.values

# Menjalankan algoritma K-Modes dengan 3 kluster dan inisialisasi random
km = KModes(n_clusters=3, init='random', n_init=5, verbose=1)
clusters = km.fit_predict(X)

# Menambahkan hasil klasterisasi ke dataset
data['cluster'] = clusters

print("\nDataset dengan klasterisasi K-Modes:\n", data)

# Visualisasi hasil klasterisasi
# Menggunakan seaborn untuk membuat plot
data_reset = data.reset_index()

# Menggunakan seaborn untuk membuat plot, menggambarkan kluster berdasarkan dua fitur kategorikal
plt.figure(figsize=(10, 6))
sns.scatterplot(x='hair_color', y='eye_color', hue='cluster', data=data_reset, palette='viridis', s=100)
plt.title('K-Modes Clustering')
plt.show()
