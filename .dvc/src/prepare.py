import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split


mlflow.set_experiment("dvc")

print("--- MULAI TAHAP 1: PREPROCESSING ---")

# Bikin folder processed pakai alamat lengkap biar Python tidak nyasar
processed_dir = r"c:\Users\gabriels\project\.dvc\data\processed"
os.makedirs(processed_dir, exist_ok=True)

# 1. Load Data Mentah pakai alamat lengkap yang tadi sudah kita temukan
print("Membaca data raw...")
df = pd.read_csv(r"c:\Users\gabriels\project\.dvc\data\raw\dataset.csv")

# 2. Bikin fitur tambahan (Panjang teks)
df['text_length'] = df['content_text'].astype(str).apply(len)

# Encoding: Ubah label Human jadi 1, AI jadi 0
df['author_type_encoded'] = df['author_type'].map({'Human': 1, 'AI': 0})

# 3. Pisahkan Fitur (X) dan Target (y)
kolom_buang = ['text_id', 'content_text', 'author_type', 'model_source', 'author_type_encoded']
X = df.drop(columns=kolom_buang)
y = df['author_type_encoded']

# 4. Data Splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Simpan Data yang Sudah Diproses ke folder yang aman
print("Menyimpan data hasil split ke folder data/processed/...")
X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

print("Pre-processing selesai!\n")