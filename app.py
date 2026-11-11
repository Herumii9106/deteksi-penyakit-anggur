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

# ----------------------------- Informasi penyakit -----------------------------
DISEASE_INFO = {
    "Black_Rot": {
        "label": "Black Rot",
        "emoji": "🟤",
        "color": "#c0392b",
        "ringkas": "Busuk hitam, disebabkan jamur Guignardia bidwellii.",
        "ciri": [
            "Bercak bulat berwarna cokelat dengan tepi lebih gelap pada daun",
            "Sering ada titik hitam kecil di tengah bercak",
            "Dapat menyerang buah: buah mengerut, menghitam, dan mengeras",
        ],
        "penanganan": [
            "Buang dan musnahkan daun atau buah yang terinfeksi",
            "Jaga kebersihan kebun dan sirkulasi udara antar tanaman",
            "Konsultasikan penggunaan fungisida dengan penyuluh pertanian",
        ],
    },
    "ESCA": {
        "label": "ESCA",
        "emoji": "🟠",
        "color": "#e67e22",
        "ringkas": "Penyakit batang dan kayu yang disebabkan kompleks jamur.",
        "ciri": [
            "Pola belang kuning atau merah di antara tulang daun ('tiger stripe')",
            "Tepi daun mengering dan mengering menjalar ke dalam",
            "Pada kasus berat, tanaman dapat layu secara tiba-tiba",
        ],
        "penanganan": [
            "Belum ada obat yang sepenuhnya efektif, pencegahan lebih penting",
            "Pangkas dengan alat yang disterilkan dan lindungi luka pangkasan",
            "Buang tanaman yang terinfeksi berat agar tidak menular",
        ],
    },
    "Healthy": {
        "label": "Healthy (Sehat)",
        "emoji": "🟢",
        "color": "#2e9e5b",
        "ringkas": "Daun tidak menunjukkan tanda penyakit.",
        "ciri": [
            "Warna hijau merata",
            "Tidak ada bercak, belang, atau bagian yang mengering",
        ],
        "penanganan": [
            "Lanjutkan perawatan rutin: penyiraman, pemupukan, pemangkasan",
            "Pantau daun secara berkala untuk mendeteksi gejala sejak dini",
        ],
    },
    "Leaf_Blight": {
        "label": "Leaf Blight",
        "emoji": "🟣",
        "color": "#8e44ad",
        "ringkas": "Hawar daun (bercak daun Isariopsis), disebabkan jamur.",
        "ciri": [
            "Bercak cokelat tua sampai kehitaman dengan bentuk tidak beraturan",
            "Daun menguning di sekitar bercak, lalu rontok lebih awal",
            "Biasanya lebih parah pada daun tua dan saat lembap",
        ],
        "penanganan": [
            "Kumpulkan dan buang daun yang gugur",
            "Kurangi kelembapan dengan pemangkasan dan jarak tanam yang baik",
            "Konsultasikan fungisida yang sesuai dengan penyuluh pertanian",
        ],
    },
}

# Hasil uji dari Colab (ubah angka ini jika model dilatih ulang)
REPORT_TABLE = pd.DataFrame(
    {
        "Kelas": ["Black_Rot", "ESCA", "Healthy", "Leaf_Blight"],
        "Precision": [0.99, 1.00, 1.00, 0.98],
        "Recall": [0.98, 1.00, 1.00, 0.99],
        "F1-score": [0.98, 1.00, 1.00, 0.99],
        "Jumlah data uji": [450, 450, 450, 450],
    }
)

# ----------------------------- Tampilan (CSS) -----------------------------
st.markdown(
    """
<style>
.block-container{padding-top:2rem;max-width:1200px;}
.hero{background:linear-gradient(120deg,#4a1d6e 0%,#7b3fa0 50%,#2e9e5b 100%);padding:28px 34px;border-radius:20px;color:#fff;margin-bottom:22px;box-shadow:0 8px 24px rgba(0,0,0,.18);}
.hero-title{font-size:2.1rem;font-weight:800;line-height:1.2;margin:0;color:#fff;}
.hero-sub{margin:8px 0 0;opacity:.92;font-size:1rem;color:#fff;}
.pill{display:inline-block;padding:4px 14px;margin:14px 8px 0 0;border-radius:999px;background:rgba(255,255,255,.18);font-size:.8rem;color:#fff;}
.result-card{border:1px solid rgba(128,128,128,.3);border-radius:18px;padding:20px 22px;background:rgba(128,128,128,.08);margin-bottom:14px;}
.result-tag{font-size:.75rem;letter-spacing:.08em;opacity:.7;}
.result-name{display:inline-block;margin:8px 0 4px;padding:6px 16px;border-radius:12px;color:#fff;font-size:1.5rem;font-weight:700;}
.result-conf{font-size:2.4rem;font-weight:800;line-height:1.1;}
.result-sub{font-size:.85rem;opacity:.7;}
.bar-row{margin:10px 0;}
.bar-label{display:flex;justify-content:space-between;font-size:.9rem;margin-bottom:4px;}
.bar-track{height:10px;border-radius:999px;background:rgba(128,128,128,.25);overflow:hidden;}
.bar-fill{height:100%;border-radius:999px;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------- Model -----------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model(path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def find_default_model():
    for name in MODEL_CANDIDATES:
        full = os.path.join(BASE_DIR, name)
        if os.path.exists(full):
            return full
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


def predict(model, rgb224: np.ndarray) -> np.ndarray:
    x = rgb224.astype("float32")
    if not has_rescaling_layer(model):
        x = x / 255.0  # model tanpa layer Rescaling butuh normalisasi manual
    return model.predict(x[None, ...], verbose=0)[0]


def prob_bars(probs, names) -> str:
    html = ""
    for i in np.argsort(probs)[::-1]:
        pct = probs[i] * 100
        info = DISEASE_INFO.get(names[i], {})
        label = info.get("label", names[i])
        color = info.get("color", "#7b3fa0")
        html += (
            f'<div class="bar-row"><div class="bar-label"><span>{label}</span><span>{pct:.1f}%</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div></div>'
        )
    return html


# ----------------------------- Sidebar -----------------------------
st.sidebar.title("🍇 Pengaturan")

st.sidebar.subheader("Preprocessing")
use_blur = st.sidebar.checkbox("Gaussian Blur (denoising)", value=True)
use_mask = st.sidebar.checkbox("Masking warna hijau (HSV)", value=False)
st.sidebar.caption(
    "Samakan dengan preprocessing saat training. Model ini dilatih dengan blur dan tanpa masking. "
    "Masking hijau dapat menghapus bercak penyakit (cokelat/kuning)."
)

with st.sidebar.expander("Pengaturan lanjutan"):
    uploaded_model = st.file_uploader("Upload model (.keras / .h5)", type=["keras", "h5"])
    class_text = st.text_input("Nama kelas (urut alfabet, pisah koma)", ", ".join(DEFAULT_CLASSES))

class_names = [c.strip() for c in class_text.split(",") if c.strip()]

if uploaded_model is not None:
    model_path = os.path.join(BASE_DIR, "uploaded_" + uploaded_model.name)
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
else:
    model_path = find_default_model()

if model_path:
    size_mb = os.path.getsize(model_path) / 1e6
    st.sidebar.success(f"Model: {os.path.basename(model_path)} ({size_mb:.1f} MB)")
else:
    st.sidebar.error("File model tidak ditemukan di repo")
    st.sidebar.write(sorted(os.listdir(BASE_DIR)))

# ----------------------------- Header -----------------------------
st.markdown(
    """
<div class="hero">
<div class="hero-title">🍇 Deteksi Penyakit Daun Anggur</div>
<div class="hero-sub">Unggah foto daun anggur dan model CNN akan memprediksi apakah daun sehat atau terkena penyakit.</div>
<span class="pill">Black Rot</span><span class="pill">ESCA</span><span class="pill">Leaf Blight</span><span class="pill">Healthy</span>
<span class="pill">OpenCV</span><span class="pill">TensorFlow CNN</span><span class="pill">Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_pred, tab_eval, tab_info, tab_pipe = st.tabs(
    ["🔍 Prediksi", "📊 Evaluasi Model", "🌿 Info Penyakit", "⚙️ Pipeline"]
)

# ----------------------------- Tab Prediksi -----------------------------
with tab_pred:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.subheader("1. Upload foto daun")
        file = st.file_uploader(
            "Pilih foto (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed"
        )
        rgb, stage1, stage2 = None, None, None
        if file is not None:
            rgb = np.array(Image.open(file).convert("RGB"))
            st.image(rgb, caption="Foto yang diupload", width="stretch")
            # Urutan sama dengan saat training: resize → blur → (opsional) masking
            resized = cv2.resize(rgb, IMG_SIZE)
            stage1 = denoise(resized) if use_blur else resized
            stage2 = green_mask(stage1) if use_mask else stage1

    with right:
        st.subheader("2. Hasil analisis")
        if rgb is None:
            st.info("Upload sebuah foto daun di sebelah kiri untuk melihat hasil prediksi.")
        elif model_path is None:
            st.warning(
                "Model belum ditemukan. Letakkan `model_anggur_terbaik.keras` di folder repo "
                "atau upload lewat Pengaturan lanjutan di sidebar."
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
                    name = class_names[best]
                    info = DISEASE_INFO.get(name, {})
                    label = info.get("label", name)
                    color = info.get("color", "#7b3fa0")
                    emoji = info.get("emoji", "🍇")
                    conf = probs[best] * 100

                    st.markdown(
                        f'<div class="result-card"><div class="result-tag">HASIL PREDIKSI</div>'
                        f'<div class="result-name" style="background:{color}">{emoji} {label}</div>'
                        f'<div class="result-conf">{conf:.1f}%</div>'
                        f'<div class="result-sub">tingkat keyakinan model</div></div>',
                        unsafe_allow_html=True,
                    )

                    if use_mask:
                        st.warning(
                            "Masking hijau sedang aktif. Bercak penyakit bisa ikut terhapus sehingga hasil kurang akurat. "
                            "Matikan masking di sidebar jika model dilatih tanpa masking."
                        )
                    if conf < 60:
                        st.warning("Model kurang yakin dengan prediksi ini. Coba foto daun tunggal dengan latar sederhana.")

                    st.markdown("**Probabilitas tiap kelas**")
                    st.markdown(prob_bars(probs, class_names), unsafe_allow_html=True)

                    if info:
                        with st.expander(f"Tentang {label}", expanded=True):
                            st.write(info["ringkas"])
                            st.markdown("**Ciri-ciri pada daun**")
                            st.markdown("\n".join(f"- {c}" for c in info["ciri"]))
                            st.markdown("**Penanganan umum**")
                            st.markdown("\n".join(f"- {p}" for p in info["penanganan"]))
                            st.caption("Informasi umum untuk edukasi, bukan pengganti diagnosis ahli.")
            except Exception as e:
                st.error(f"Gagal memuat atau menjalankan model: {e}")

    if rgb is not None:
        st.divider()
        st.subheader("3. Tahapan preprocessing")
        c1, c2, c3 = st.columns(3)
        c1.image(rgb, caption="Gambar asli", width="stretch")
        c2.image(
            stage1,
            caption="Resize 224×224 + denoising" if use_blur else "Resize 224×224 (denoising nonaktif)",
            width="stretch",
        )
        c3.image(
            stage2,
            caption="Setelah masking hijau" if use_mask else "Masking (nonaktif)",
            width="stretch",
        )

# ----------------------------- Tab Evaluasi -----------------------------
with tab_eval:
    st.subheader("Hasil evaluasi pada data uji")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", "99%")
    m2.metric("Jumlah data uji", "1800")
    m3.metric("Jumlah kelas", "4")
    m4.metric("Rata-rata F1", "0.99")

    st.markdown("**Classification report**")
    st.dataframe(REPORT_TABLE, hide_index=True, width="stretch")
    st.caption(
        "Angka diambil dari hasil uji di Colab. Data uji berasal dari dataset yang sama dengan data latih, "
        "dan dataset dapat berisi gambar hasil augmentasi, sehingga akurasi bisa lebih tinggi daripada pada foto baru."
    )

    st.markdown("**Confusion matrix**")
    cm_file = st.file_uploader("Upload gambar confusion matrix (opsional)", type=["png", "jpg"], key="cm")
    if cm_file is not None:
        st.image(cm_file, caption="Confusion Matrix", width="stretch")
    elif os.path.exists(CM_IMAGE):
        st.image(CM_IMAGE, caption="Confusion Matrix", width="stretch")
    else:
        st.info("Taruh `hasil_evaluasi_confusion_matrix.png` di folder repo atau upload di atas.")

# ----------------------------- Tab Info Penyakit -----------------------------
with tab_info:
    st.subheader("Mengenal kelas penyakit")
    cols = st.columns(2)
    for i, (name, info) in enumerate(DISEASE_INFO.items()):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"### {info['emoji']} {info['label']}")
                st.write(info["ringkas"])
                st.markdown("**Ciri-ciri pada daun**")
                st.markdown("\n".join(f"- {c}" for c in info["ciri"]))
                st.markdown("**Penanganan umum**")
                st.markdown("\n".join(f"- {p}" for p in info["penanganan"]))
    st.caption("Informasi bersifat umum untuk edukasi. Untuk diagnosis dan penanganan, konsultasikan dengan penyuluh pertanian.")

# ----------------------------- Tab Pipeline -----------------------------
with tab_pipe:
    st.subheader("Tahapan pipeline")
    st.markdown(
        """
1. **Input data mentah**: foto dibaca dan diubah ke format RGB.
2. **Resize**: gambar diubah ke 224×224 piksel.
3. **Denoising**: Gaussian Blur 5×5 untuk mengurangi noise.
4. **Segmentasi ROI (opsional)**: masking HSV warna hijau.
5. **Normalisasi**: piksel diskalakan ke 0–1 (di dalam model).
6. **CNN**: 4 blok Conv2D + MaxPooling, Flatten, Dense 512, Dropout 0.5, Softmax 4 kelas.
7. **Evaluasi**: classification report (precision, recall, F1) dan confusion matrix.
"""
    )
