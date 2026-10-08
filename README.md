# 👗 Visual-Based Fashion Recommender System

Sistem rekomendasi produk fashion berbasis konten visual menggunakan CNN sebagai feature extractor dan Cosine Similarity sebagai mesin retrieval. Dibangun sebagai proyek skripsi dengan komparasi 4 arsitektur CNN.

---

## 📁 Struktur Proyek

```
Visual Based Rekomender Sistem/
├── app/                        ← Aplikasi Streamlit (user study A/B: CNN vs Text)
│   ├── main.py                 ← Entry point aplikasi
│   ├── engine.py               ← Engine dual recommender (CNN VGG19 Exp3 + One-Hot text)
│   ├── data_loader.py          ← Auto-download data dari GitHub Releases saat first run
│   ├── sheets.py               ← Logging respons user study ke Google Sheets
│   └── config.py               ← Konfigurasi path & hyperparameter
├── notebooks/                  ← Pipeline eksperimen (jalankan di Google Colab)
│   ├── 01_data_preparation.py  ← langkah 1: bangun master_dataset.csv
│   ├── 02_feature_extraction.py← langkah 2: ekstraksi fitur CNN (GPU)
│   ├── 03_evaluation.py        ← langkah 3: komparasi 4 arsitektur CNN
│   ├── 04x_experiment3*.py     ← fine-tuning & partial unfreeze (VGG19 dkk)
│   ├── 05x_retrieval_evaluation.py ← evaluasi retrieval per eksperimen
│   └── 06–08                   ← baseline text-CBF & visualisasi vektor
├── evaluation/                 ← Hasil metrik (CSV) & grafik (PNG) tiap eksperimen
├── baseline_text_cbf/          ← Baseline text-CBF: one-hot, analisis leakage atribut
├── grid_search/                ← Script & hasil tuning hyperparameter
├── prepare_deploy_data.py      ← Bangun arsip data untuk deploy (GitHub Releases)
├── requirements.txt
└── README.md
```

> **Catatan:** folder `data/`, `dataset/`, `features/`, `models/` (file `.npy`/`.zip` berukuran GB) sengaja **tidak** di-push ke GitHub karena melewati batas 100 MB per file. Data runtime diambil otomatis oleh `app/data_loader.py` dari GitHub Releases.

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
- `COLD_START_K` — jumlah item cold-start hasil K-Means (default: 8)
- `REC_TOP_N` — jumlah rekomendasi per set (default: 8)
- Path fitur & dataset (`FEATURES_DIR`, `DATASET_DIR`) — di dev lokal menunjuk ke `features/` dan `dataset/`, di cloud otomatis pakai arsip dari GitHub Releases.

---

## 🧠 Model CNN yang Digunakan

| Model | Feature Dim | Params | Input Size |
|---|---|---|---|
| ResNet50 | 2048 | ~25M | 224×224 |
| VGG19 | 512 | ~143M | 224×224 |
| InceptionV3 | 2048 | ~23M | 299×299 |
| MobileNetV3 | 960 | ~5.4M | 224×224 |

> Eksperimen: Exp1 memakai bobot **ImageNet pre-trained** sebagai feature extractor murni; Exp2–Exp3 melakukan fine-tuning (termasuk partial unfreeze). Model terbaik yang dipakai di aplikasi: **VGG19 Exp3 (partial unfreeze, 512-D)**, dibandingkan baseline **One-Hot text-CBF (1158-D)**.

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

**DeepFashion — In-shop Clothes Retrieval Benchmark**  
[https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

File yang dibutuhkan (diletakkan di `dataset/In-shop Clothes Retrieval Benchmark/`):
- `Img/` — folder gambar
- `Anno/list_item_inshop.txt`
- `Anno/list_eval_partition.txt`
- `Anno/list_bbox_inshop.txt`

Dataset tidak di-commit ke repo (ukurannya >13 GB). Unduh manual dari link di atas, atau jalankan `notebooks/01_data_preparation.py` di Colab dengan dataset yang sudah ada di Google Drive.
