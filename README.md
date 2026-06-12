# 👗 Visual-Based Fashion Recommender System

Sistem rekomendasi produk fashion berbasis konten visual menggunakan CNN sebagai feature extractor dan Cosine Similarity sebagai mesin retrieval. Dibangun sebagai proyek skripsi dengan komparasi 4 arsitektur CNN.

---

## 📁 Struktur Proyek

```
Visual Based Rekomender Sistem/
├── notebooks/
│   ├── 01_data_preparation.py    ← Jalankan di Google Colab (langkah 1)
│   ├── 02_feature_extraction.py  ← Jalankan di Google Colab GPU (langkah 2)
│   └── 03_evaluation.py          ← Jalankan di Google Colab (langkah 3)
├── app/
│   ├── main.py          ← Streamlit app (entry point)
│   ├── recommender.py   ← Engine: cold-start, user profile, cosine sim
│   ├── utils.py         ← Helper: image loading, formatting
│   └── config.py        ← Konfigurasi path & hyperparameter
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan

### Langkah 1 — Persiapan Data (Google Colab)
1. Upload `notebooks/01_data_preparation.py` ke Colab
2. Salin kode ke cell-cell Colab
3. Sesuaikan `BASE_DIR` di Cell 3 dengan path Google Drive kamu
4. Jalankan semua cell → menghasilkan `master_dataset.csv`

### Langkah 2 — Ekstraksi Fitur (Google Colab GPU)
1. Gunakan runtime **GPU** di Colab
2. Upload `notebooks/02_feature_extraction.py`
3. Jalankan semua cell → menghasilkan 4 file `.npy` di Google Drive:
   - `resnet50_features.npy`
   - `vgg19_features.npy`
   - `inceptionv3_features.npy`
   - `mobilenetv3_features.npy`
4. Durasi: ~20-60 menit tergantung ukuran dataset

### Langkah 3 — Evaluasi (Google Colab)
1. Upload `notebooks/03_evaluation.py`
2. Jalankan setelah Langkah 2 selesai
3. Menghasilkan tabel komparasi, bar chart, dan heatmap

### Langkah 4 — Jalankan Streamlit App (Lokal)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Sesuaikan path di app/config.py dengan lokasi file .npy kamu
# (path bisa ke Google Drive yang di-mount, atau download ke lokal)

# 3. Jalankan app
streamlit run app/main.py
```

---

## ⚙️ Konfigurasi

Edit `app/config.py` untuk menyesuaikan:
- `GDRIVE_BASE` — lokasi folder di Google Drive
- `TOP_N` — jumlah rekomendasi yang ditampilkan (default: 10)
- `COLD_START_N` — jumlah item cold-start (default: 8)

---

## 🧠 Model CNN yang Digunakan

| Model | Feature Dim | Params | Input Size |
|---|---|---|---|
| ResNet50 | 2048 | ~25M | 224×224 |
| VGG19 | 512 | ~143M | 224×224 |
| InceptionV3 | 2048 | ~23M | 299×299 |
| MobileNetV3 | 960 | ~5.4M | 224×224 |

> Semua model menggunakan bobot **ImageNet pre-trained** dengan Global Average Pooling (tanpa fine-tuning) sebagai feature extractor murni.

---

## 📊 Metrik Evaluasi

- **Precision@K** — proporsi item relevan di top-K
- **Recall@K** — cakupan item relevan dari seluruh ground truth
- **F1@K** — harmonic mean Precision & Recall
- **NDCG@K** — mempertimbangkan posisi ranking
- **Intra-list Diversity** — keberagaman item dalam satu daftar rekomendasi

Dievaluasi pada K = 5, 10, 20 dengan 100 simulasi user.

---

## 📦 Dataset

**DeepFashion — Category and Attribute Prediction Benchmark**  
[https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)

File yang dibutuhkan:
- `Img/` — folder gambar
- `Anno/list_category_cloth.txt`
- `Anno/list_category_img.txt`
