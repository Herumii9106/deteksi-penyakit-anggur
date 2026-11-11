# Deteksi Penyakit Daun Anggur

Aplikasi Streamlit untuk memvisualkan pipeline: preprocessing OpenCV → CNN → prediksi kelas
(Black Rot, ESCA, Healthy, Leaf Blight).

## Isi repo

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `requirements.txt` | Daftar library |
| `model_anggur_terbaik.keras` | Model hasil training dari Colab (tambahkan sendiri) |
| `hasil_evaluasi_confusion_matrix.png` | Confusion matrix dari Colab (opsional) |

## Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Push semua file ke repo GitHub (branch `main`).
2. Buka https://share.streamlit.io lalu login dengan GitHub.
3. Klik **Create app**, pilih repo dan branch, isi **Main file path** dengan `app.py`.
4. Di **Advanced settings**, pilih Python 3.11 atau 3.12.
5. Klik **Deploy**.
