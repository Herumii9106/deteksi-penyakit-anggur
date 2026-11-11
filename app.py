import glob
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Deteksi Penyakit Daun Anggur", page_icon="🍇", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)
DEFAULT_CLASSES = ["Black_Rot", "ESCA", "Healthy", "Leaf_Blight"]  # urutan alfabet = urutan Keras
MODEL_CANDIDATES = ["model_anggur_terbaik.keras", "model_anggur_terbaik.h5"]
CM_IMAGE = os.path.join(BASE_DIR, "hasil_evaluasi_confusion_matrix.png")


# ----------------------------- Model -----------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model(path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def find_default_model():
    # 1) nama yang diharapkan
    for name in MODEL_CANDIDATES:
        full = os.path.join(BASE_DIR, name)
        if os.path.exists(full):
            return full
    # 2) cadangan: file .keras / .h5 apa pun di folder aplikasi
    for pattern in ("*.keras", "*.h5"):
        found = [f for f in sorted(glob.glob(os.path.join(BASE_DIR, pattern)))
                 if not os.path.basename(f).startswith("uploaded_")]
        if found:
            return found[0]
    return None


def has_rescaling_layer(model) -> bool:
    return any(l.__class__.__name__ == "Rescaling" for l in model.layers)


# ----------------------------- Preprocessing (OpenCV) -----------------------------
def denoise(rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (5, 5), 0)


def green_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    return cv2.bitwise_and(rgb, rgb, mask=mask)


def predict(model, rgb: np.ndarray) -> np.ndarray:
    x = cv2.resize(rgb, IMG_SIZE).astype("float32")
    if not has_rescaling_layer(model):
        x = x / 255.0  # model tanpa layer Rescaling butuh normalisasi manual
    return model.predict(x[None, ...], verbose=0)[0]


# ----------------------------- Sidebar -----------------------------
st.sidebar.title("🍇 Pengaturan")

uploaded_model = st.sidebar.file_uploader("Upload model (.keras / .h5)", type=["keras", "h5"])
if uploaded_model is not None:
    model_path = os.path.join(BASE_DIR, "uploaded_" + uploaded_model.name)
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
else:
    model_path = find_default_model()

# Status model (diagnostik)
if model_path:
    size_mb = os.path.getsize(model_path) / 1e6
    st.sidebar.success(f"Model: {os.path.basename(model_path)} ({size_mb:.1f} MB)")
else:
    st.sidebar.error("File model tidak ditemukan di repo")
    st.sidebar.write("File yang terlihat oleh aplikasi:")
    st.sidebar.write(sorted(os.listdir(BASE_DIR)))

class_text = st.sidebar.text_input("Nama kelas (urut alfabet, pisah koma)", ", ".join(DEFAULT_CLASSES))
class_names = [c.strip() for c in class_text.split(",") if c.strip()]

st.sidebar.subheader("Preprocessing sebelum prediksi")
use_blur = st.sidebar.checkbox("Gaussian Blur (denoising)", value=True)
use_mask = st.sidebar.checkbox("Masking warna hijau (HSV)", value=False)
st.sidebar.caption(
    "Aktifkan hanya jika model dilatih dengan preprocessing yang sama. "
    "Masking hijau dapat ikut menghapus bercak penyakit (cokelat/kuning), jadi sebaiknya dimatikan."
)

# ----------------------------- Halaman utama -----------------------------
st.title("Deteksi Penyakit Daun Anggur")
st.caption("Pipeline: Input gambar → Denoising → (opsional) Masking → Resize 224×224 → CNN → Prediksi kelas")

tab_pred, tab_eval, tab_info = st.tabs(["Prediksi", "Evaluasi Model", "Tentang Pipeline"])

with tab_pred:
    file = st.file_uploader("Upload foto daun anggur", type=["jpg", "jpeg", "png", "bmp"])
    if file is None:
        st.info("Upload sebuah gambar untuk melihat tahapan preprocessing dan hasil prediksi.")
    else:
        rgb = np.array(Image.open(file).convert("RGB"))
        stage1 = denoise(rgb) if use_blur else rgb
        stage2 = green_mask(stage1) if use_mask else stage1

        # Hasil prediksi ditampilkan di atas gambar supaya langsung terlihat
        if model_path is None:
            st.warning(
                "Model belum ditemukan. Letakkan `model_anggur_terbaik.keras` di folder repo "
                "atau upload lewat sidebar."
            )
        else:
            try:
                model = load_model(model_path)
                probs = predict(model, stage2)
                if len(class_names) != len(probs):
                    st.error(
                        f"Jumlah kelas di sidebar ({len(class_names)}) tidak sama dengan output model ({len(probs)})."
                    )
                else:
                    best = int(np.argmax(probs))
                    r1, r2 = st.columns([1, 2])
                    with r1:
                        st.subheader(f"Prediksi: {class_names[best]}")
                        st.metric("Keyakinan", f"{probs[best] * 100:.1f}%")
                    with r2:
                        st.bar_chart(pd.Series(probs, index=class_names, name="Probabilitas"))
            except Exception as e:
                st.error(f"Gagal memuat atau menjalankan model: {e}")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.image(rgb, caption="1. Gambar asli", width="stretch")
        c2.image(stage1, caption="2. Setelah denoising" if use_blur else "2. Denoising (nonaktif)", width="stretch")
        c3.image(stage2, caption="3. Setelah masking" if use_mask else "3. Masking (nonaktif)", width="stretch")

with tab_eval:
    st.write("Hasil evaluasi dari notebook (confusion matrix dan classification report).")
    cm_file = st.file_uploader("Upload gambar confusion matrix (opsional)", type=["png", "jpg"], key="cm")
    if cm_file is not None:
        st.image(cm_file, caption="Confusion Matrix", width="stretch")
    elif os.path.exists(CM_IMAGE):
        st.image(CM_IMAGE, caption="Confusion Matrix", width="stretch")
    else:
        st.info("Taruh `hasil_evaluasi_confusion_matrix.png` di folder repo atau upload di atas.")

    report = st.text_area("Tempel classification report (opsional)", height=220)
    if report.strip():
        st.code(report)

with tab_info:
    st.markdown(
        """
**Tahapan pipeline**

1. **Input data mentah**: gambar dibaca dan diubah dari BGR ke RGB.
2. **Denoising**: Gaussian Blur 5×5 untuk mengurangi noise.
3. **Segmentasi ROI (opsional)**: masking HSV warna hijau.
4. **Resize dan normalisasi**: 224×224, piksel diskalakan ke 0–1.
5. **CNN**: 4 blok Conv2D + MaxPooling, Flatten, Dense 512, Dropout 0.5, Softmax.
6. **Evaluasi**: classification report (precision, recall, F1) dan confusion matrix.
"""
    )
