import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# =====================================================================
# BAGIAN 1: PREDICTION MODEL DESCRIPTION (Line 4 - 13 dari Algorithm 1)
# =====================================================================
def Image_Forgery_Predictor_Model():
    """
    Membangun arsitektur CNN ringan berdasarkan Algorithm 1.
    Input berupa feature image berukuran 128x128x3.
    """
    model = models.Sequential()
    
    # First convo. layer: 32 filters (size 3x3, stride size one, activation: "relu")
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(128, 128, 3), padding='same'))
    
    # Second convo. layer: 32 filters (size 3x3, stride size one, activation: "relu")
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    
    # Third convo. layer: 32 filters (size 3x3, stride size one, activation: "relu")
    # Catatan: Di teks paragraf paper (Section 3) disebut ukuran 7x7, tapi di Algorithm 1 ditulis 3x3.
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    
    # Max-pooling of size 2x2
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Meratakan (flatten) tensor sebelum masuk ke Dense layer
    model.add(layers.Flatten())
    
    # Dense layer of 256 neurons with "relu" activation function
    model.add(layers.Dense(256, activation='relu'))
    
    # Two neurons (output neurons) with "sigmoid" activation
    model.add(layers.Dense(2, activation='sigmoid'))
    
    return model

# =====================================================================
# FUNGSI BANTUAN: JPEG COMPRESSION & DIFFERENCE (Line 17-19 & 27-29)
# =====================================================================
def extract_recompression_feature(image, quality_factor=98):
    """
    Melakukan ekstraksi fitur Adiff berdasarkan kompresi ganda.
    Berdasarkan paper, akurasi tertinggi didapat pada quality factor 98.
    """
    # 1. JPEG Compression (A_recompressed)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    
    if not result:
        raise ValueError("Gagal melakukan kompresi JPEG")
        
    recompressed_image = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    # 2. A_diff = A - A_recompressed
    # Menggunakan cv2.absdiff untuk menghindari nilai negatif pada piksel (underflow)
    diff_image = cv2.absdiff(image, recompressed_image)
    
    # 3. A_reshaped_diff = reshape(A_diff, (128, 128, 3))
    reshaped_diff = cv2.resize(diff_image, (128, 128))
    
    return reshaped_diff

# =====================================================================
# BAGIAN 2: MODEL TRAINING (Line 14 - 23 dari Algorithm 1)
# =====================================================================
def train_model(X_train_images, Y_train_labels, total_epochs=60):
    """
    X_train_images: List/Array gambar RGB asli
    Y_train_labels: List/Array label One-Hot Encoded. 
                    Misal: [1, 0] untuk Tampered, [0, 1] untuk Authentic
    """
    print("Mengekstrak fitur pelatihan...")
    X_train_features = []
    
    # Looping untuk mengonversi gambar asli (A) menjadi matriks selisih kompresi (A_diff)
    # Line 16-21 Algorithm 1
    for i, img in enumerate(X_train_images):
        feature = extract_recompression_feature(img, quality_factor=98)
        X_train_features.append(feature)
        
    X_train_features = np.array(X_train_features, dtype='float32') / 255.0 # Normalisasi piksel
    Y_train_labels = np.array(Y_train_labels)
    
    # Inisialisasi Model
    model = Image_Forgery_Predictor_Model()
    
    # Sesuai line 22 di Algorithm 1, menggunakan Adam_optimizer
    # Nilai learning rate 1e-5 didapat dari eksperimen di dalam paper (Section 4.2)
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Memulai proses training...")
    # modify_model(training_error) -> diwakilkan oleh model.fit() di Keras
    model.fit(X_train_features, Y_train_labels, epochs=total_epochs, batch_size=64)
    
    return model

# =====================================================================
# BAGIAN 3: IMAGE FORGERY PREDICTION (Line 24 - 32 dari Algorithm 1)
# =====================================================================
def predict_forgery(model, input_image_path):
    """
    Fungsi untuk menebak apakah gambar baru itu asli atau palsu.
    """
    # Baca input image
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print("Gambar tidak ditemukan!")
        return
    
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Line 27-29: Recompress, Diff, Reshape
    input_image_reshaped_diff = extract_recompression_feature(input_image, quality_factor=98)
    
    # Ekspansi dimensi karena model meminta input (batch_size, 128, 128, 3)
    input_tensor = np.expand_dims(input_image_reshaped_diff, axis=0) / 255.0
    
    # Line 30: Predicted_label = Image_Forgery_Predictor_Model(Input_Image_reshaped_diff)
    predicted_label = model.predict(input_tensor)[0]
    
    print(f"Probabilitas Output [Tampered, Authentic]: {predicted_label}")
    
    # Line 31-32: Logika If dari Algorithm 1
    if predicted_label[0] > predicted_label[1]:
        print(">> KESIMPULAN: Gambar terdeteksi PALSU (Tampered)")
    elif predicted_label[1] > predicted_label[0]:
        print(">> KESIMPULAN: Gambar terdeteksi ASLI (Untampered)")
    else:
        print(">> KESIMPULAN: Probabilitas seri (Uncertain)")

# ======================= CONTOH PENGGUNAAN =======================
if __name__ == "__main__":
    print("Arsitektur CNN berhasil dibuat. Menunggu proses integrasi dataset...")
    
    # Kamu bisa memanggil modelnya dengan cara:
    # my_model = Image_Forgery_Predictor_Model()
    # my_model.summary()

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_casia_dataset(dataset_dir, max_samples_per_class=None):
    """
    Fungsi untuk membaca dataset CASIA dari folder.
    
    Struktur folder yang diharapkan:
    dataset_dir/
      ├── Au/   (berisi gambar asli)
      └── Tp/   (berisi gambar palsu)
    """
    X_data = []
    Y_labels = []
    
    au_dir = os.path.join(dataset_dir, "Au")
    tp_dir = os.path.join(dataset_dir, "Tp")
    
    print(f"Membaca dataset dari: {dataset_dir}")
    
    # 1. Membaca Gambar Authentic (Asli) -> Label: [0, 1]
    if os.path.exists(au_dir):
        au_files = os.listdir(au_dir)
        if max_samples_per_class:
            au_files = au_files[:max_samples_per_class]
            
        for file_name in au_files:
            file_path = os.path.join(au_dir, file_name)
            img = cv2.imread(file_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X_data.append(img_rgb)
                Y_labels.append([0, 1])
    else:
        print(f"Folder {au_dir} tidak ditemukan!")

    # 2. Membaca Gambar Tampered (Palsu) -> Label: [1, 0]
    if os.path.exists(tp_dir):
        tp_files = os.listdir(tp_dir)
        if max_samples_per_class:
            tp_files = tp_files[:max_samples_per_class]
            
        for file_name in tp_files:
            file_path = os.path.join(tp_dir, file_name)
            img = cv2.imread(file_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X_data.append(img_rgb)
                Y_labels.append([1, 0])
    else:
        print(f"Folder {tp_dir} tidak ditemukan!")

    print(f"Total gambar berhasil dimuat: {len(X_data)}")
    return X_data, Y_labels

# ======================= EKSEKUSI TRAINING =======================
if __name__ == "__main__":
    # Ganti string di bawah dengan lokasi folder dataset CASIA 2.0 di komputermu
    CASIA_DIR = r"C:\Users\revia\Documents\Project\PCA\Image Forgery Detection\CASIA2.0_revised"
    
    # TIP: Set max_samples_per_class=100 dulu untuk memastikan tidak ada error.
    # Jika sudah aman dan RAM memadai, ubah menjadi None untuk melatih seluruh dataset.
    print("--- TAHAP 1: PERSIAPAN DATA ---")
    X_images, Y_labels = load_casia_dataset(CASIA_DIR, max_samples_per_class=None) 
    
    if len(X_images) > 0:
        # Membagi data menjadi 80% Training dan 20% Testing (Sesuai eksperimen di Jurnal)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_images, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels
        )
        
        print(f"\nJumlah Data Latih (Train): {len(X_train)}")
        print(f"Jumlah Data Uji (Test): {len(X_test)}")
        
        print("\n--- TAHAP 2: PROSES TRAINING ---")
        # Memanggil fungsi dari kode sebelumnya (ubah total_epochs sesuai kebutuhan)
        trained_model = train_model(X_train, Y_train, total_epochs=10)
        
        print("\n--- TAHAP 3: MENYIMPAN MODEL ---")
        model_save_path = "model_deteksi_citra_agro.h5"
        trained_model.save(model_save_path)
        print(f"Model berhasil disimpan di: {model_save_path}")
        
        print("\n--- TAHAP 4: PENGUJIAN PREDIKSI ---")
        # Menguji model dengan gambar pertama dari data uji
        print("Menguji prediksi pada salah satu gambar testing...")
        
        # Karena X_test berisi array gambar langsung (bukan path), 
        # kita sesuaikan sedikit cara prediksi untuk testing array
        test_img = X_test[0]
        test_feature = extract_recompression_feature(test_img, quality_factor=98)
        test_tensor = np.expand_dims(test_feature, axis=0) / 255.0
        
        pred_label = trained_model.predict(test_tensor)[0]
        aktual_label = Y_test[0]
        
        print(f"Label Aktual: {'Palsu' if aktual_label[0] == 1 else 'Asli'} ({aktual_label})")
        print(f"Probabilitas Prediksi: {pred_label}")
    else:
        print("Dataset kosong. Periksa kembali path/lokasi folder dataset kamu.")