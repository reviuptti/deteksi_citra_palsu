import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import requests  # <-- Ditambahkan untuk mengunduh file

import tensorflow as tf
from tensorflow.keras.models import load_model

# ================= KONFIGURASI HALAMAN =================
st.set_page_config(layout="wide", page_title="Deteksi Forgery (Ali et al., 2022)")

st.title("🔍 Image Forgery Detection System")
st.markdown("""
**Berdasarkan Algoritma:** *Image Forgery Detection Using Deep Learning by Recompressing Images (Ali et al., 2022)*.
Sistem ini mengekstraksi inkonsistensi kompresi (noise) dan menganalisisnya menggunakan Convolutional Neural Network (CNN).
""")

# ================= MUAT MODEL DARI GITHUB RELEASES =================
@st.cache_resource
def load_forgery_model():
    model_name = "model_deteksi_citra_agro.h5"
    
    # ⚠️ GANTI TAUTAN INI DENGAN LINK DOWNLOAD DARI GITHUB RELEASE ANDA
    model_url = "https://github.com/reviuptti/deteksi_citra_palsu/releases/download/v1.0/model_deteksi_citra_agro.h5"

    # Jika model belum ada di direktori aplikasi (misal saat baru di-deploy)
    if not os.path.exists(model_name):
        with st.spinner("Mengunduh model (300 MB) untuk pertama kali. Mohon tunggu, proses ini butuh beberapa menit..."):
            try:
                # Unduh dengan stream agar tidak memakan RAM berlebih
                response = requests.get(model_url, stream=True)
                response.raise_for_status() # Pastikan link valid (tidak error 404)
                
                with open(model_name, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("✅ Model berhasil diunduh!")
            except Exception as e:
                st.error(f"❌ Gagal mengunduh model: {e}")
                return None
    
    # Memuat model
    try:
        return load_model(model_name)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

cnn_model = load_forgery_model()

if cnn_model is None:
    st.warning("⚠️ Model belum siap. Pastikan link GitHub Release sudah benar dan dapat diakses publik.")

# ================= FITUR RESET =================
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def reset_app():
    st.session_state.uploader_key += 1

st.sidebar.button("🔄 Reset Pilihan", on_click=reset_app)

# ================= FUNGSI EKSTRAKSI & ANALISIS =================

def extract_recompression_steps(image_rgb, quality_factor=98):
    """
    Algoritma 1 dari Paper:
    Menghasilkan gambar kompresi ulang, gambar selisih (Adiff), dan Tensor 128x128.
    """
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Kompresi JPEG ulang (A_recompressed)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    recomp_bgr = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    recomp_rgb = cv2.cvtColor(recomp_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Gambar Selisih (A_diff = |A - A_recompressed|)
    diff_rgb = cv2.absdiff(image_rgb, recomp_rgb)
    
    # 3. Reshape untuk Input CNN (A_reshaped_diff)
    tensor_input = cv2.resize(diff_rgb, (128, 128))
    
    return recomp_rgb, diff_rgb, tensor_input

def plot_histogram(img_rgb):
    """Grafik Distribusi Intensitas Piksel"""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, linewidth=1.5, alpha=0.8)
        ax.set_xlim([0, 256])
    ax.set_title("Distribusi Intensitas Piksel (RGB)", fontsize=10)
    ax.set_xlabel("Nilai Piksel", fontsize=8)
    ax.set_ylabel("Frekuensi", fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    return fig

def plot_prediction_bar(auth_prob, tamp_prob):
    """Grafik Bar untuk Probabilitas Prediksi Model"""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = ['ASLI (Authentic)', 'PALSU (Tampered)']
    probabilities = [auth_prob, tamp_prob]
    colors = ['#28a745', '#dc3545']
    
    bars = ax.bar(labels, probabilities, color=colors, width=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probabilitas (%)", fontsize=10)
    ax.set_title("Tingkat Keyakinan Prediksi CNN", fontsize=10)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ================= ANTARMUKA UTAMA =================
st.sidebar.header("📂 Upload Panel")
uploaded_files = st.sidebar.file_uploader(
    "Unggah Gambar (Pilih > 1 untuk membandingkan)", 
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], 
    accept_multiple_files=True,
    key=st.session_state.uploader_key
)

if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        st.markdown(f"### 📷 Analisis Gambar {idx+1}: `{file.name}`")
        
        # Baca Gambar
        file_bytes = file.getvalue()
        img_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_rgb = np.array(img_pil)
        
        # Ekstraksi Fitur Paper Ali dkk.
        recomp_img, diff_img, tensor_input = extract_recompression_steps(img_rgb, quality_factor=98)
        
        # Prediksi CNN
        if cnn_model is not None:
            # Normalisasi & Prediksi
            test_tensor = np.expand_dims(tensor_input, axis=0) / 255.0
            prediction = cnn_model.predict(test_tensor)[0]
            
            # Sesuai training: Index 0 = Tampered, Index 1 = Authentic
            tamp_prob = prediction[0] * 100
            auth_prob = prediction[1] * 100
            
            # Kolom Hasil Utama
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown("#### Kesimpulan Analisis")
                if auth_prob > tamp_prob:
                    st.success(f"✅ **CITRA ASLI**\n\nKeakuratan: {auth_prob:.2f}%")
                else:
                    st.error(f"🚨 **CITRA PALSU (DIMANIPULASI)**\n\nKeakuratan: {tamp_prob:.2f}%")
            
            with res_col2:
                st.markdown("#### Grafik Hasil Analisis Prediksi")
                fig_bar = plot_prediction_bar(auth_prob, tamp_prob)
                st.pyplot(fig_bar)

        # --- TAMPILAN VISUALISASI PROSES & GRAFIK TAMBAHAN ---
        st.markdown("#### Bedah Tahapan Ekstraksi Fitur (CNN Input)")
        st.info("Area yang dimanipulasi akan memiliki inkonsistensi sejarah kompresi, sehingga akan menonjol (terang) pada Gambar Selisih (Adiff).")
        
        step1, step2, step3, step4 = st.columns(4)
        with step1:
            st.image(img_rgb, caption="1. Citra Input (A)", use_container_width=True)
        with step2:
            st.image(recomp_img, caption="2. Kompresi Ulang (Q=98)", use_container_width=True)
        with step3:
            diff_display = cv2.convertScaleAbs(diff_img, alpha=25.0, beta=0)
            st.image(diff_display, caption="3. Gambar Selisih (Adiff)", use_container_width=True)
        with step4:
            tensor_display = cv2.convertScaleAbs(tensor_input, alpha=25.0, beta=0)
            st.image(tensor_display, caption="4. Tensor CNN (128x128)", use_container_width=True)
        
        # Grafik Analisis Statistik
        with st.expander("📊 Lihat Analisis Statistik Piksel (Histogram)"):
            c_hist1, c_hist2 = st.columns([1, 2])
            with c_hist1:
                st.write("**Penjelasan:**")
                st.write("Distribusi piksel ini membantu menganalisis tekstur gambar. Pada gambar yang mengalami manipulasi (*splicing/copy-move*), terkadang ditemukan anomali berupa lonjakan yang tidak natural pada kurva warnanya dibandingkan dengan gambar asli.")
            with c_hist2:
                fig_hist = plot_histogram(img_rgb)
                st.pyplot(fig_hist)

        st.divider()
else:
    st.info("Silakan unggah gambar dari bilah menu di sebelah kiri.")
