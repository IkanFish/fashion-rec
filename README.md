# 👗 Visual-Based Fashion Recommender System

Sistem rekomendasi produk fashion berbasis konten visual menggunakan CNN sebagai feature extractor dan Cosine Similarity sebagai mesin retrieval. Dibangun sebagai proyek skripsi dengan komparasi 4 arsitektur CNN.

---

## 📁 Struktur Proyek

```
Visual Based Rekomender Sistem/
├── app/                        ← Aplikasi Streamlit final (user study A/B: CNN vs Text)
│   ├── main.py                 ← Entry point aplikasi
│   ├── engine.py               ← Engine dual recommender (CNN VGG19 Exp3 + One-Hot text)
│   ├── data_loader.py          ← Auto-download data dari GitHub Releases saat first run
│   ├── sheets.py               ← Logging respons user study ke Google Sheets
│   └── config.py               ← Konfigurasi path & hyperparameter
├── experiments/                ← Pipeline eksperimen final (urut 01 → 08)
│   ├── 01_data_preparation/    ← bangun master_dataset.csv dari DeepFashion
│   ├── 02_baseline/            ← ekstraksi fitur bobot ImageNet mentah (tanpa fine-tuning)
│   ├── 03_lightweight_finetuning/ ← fine-tuning head saja + eval klasifikasi/retrieval/rekomendasi
│   ├── 04_partial_unfreeze/    ← unfreeze sebagian layer + eval (model final: VGG19 Exp3)
│   ├── 05_retrieval_analysis/  ← analisis retrieval exp3 per kategori (hit/miss, recall)
│   ├── 06_text_cbf_onehot/     ← baseline teks one-hot + analisis leakage atribut
│   ├── 07_nn_visualization/    ← visualisasi tetangga terdekat per kategori
│   └── 08_user_study/          ← data mentah A/B testing + notebook analisis
├── prepare_deploy_data.py      ← Bangun arsip data untuk deploy (GitHub Releases)
├── visualisasi_arsitektur_cnn.ipynb
├── grafik_peningkatan_recommender.ipynb
├── requirements.txt
└── README.md
```

Setiap folder eksperimen memiliki konvensi yang sama:

```
0X_nama/
├── kode/            ← script training/evaluasi (Colab-ready)
├── hasil_evaluasi/  ← CSV metrik + grafik PNG + ringkasan (SUMMARY_*.md)
├── fitur_vektor/    ← (TIDAK di-push) output .npy / .h5 berukuran besar
└── logs/            ← log training per model
```

> **Catatan:** folder `data/`, `dataset/`, `features/`, `models/` dan file `.npy`/`.h5`/`.zip` (berukuran GB) sengaja **tidak** di-push ke GitHub karena melewati batas 100 MB per file. Data runtime diambil otomatis oleh `app/data_loader.py` dari GitHub Releases.

---

## � Alur Pipeline Eksperimen

```
01 Data Preparation
   └→ 02 Baseline (bobot ImageNet mentah)
        └→ 03 Lightweight Fine Tuning
             └→ 04 Partial Unfreeze  ← model final: VGG19 Exp3
                  └→ 05 Analisis Retrieval
06 Text CBF One-Hot (baseline pembanding, paralel)
        └→ 07 Visualisasi NN
             └→ Pembangunan App (app/)
                  └→ 08 User Study (A/B testing)
```

Matriks evaluasi per tahap:

| Tahap | Klasifikasi | Retrieval | Rekomendasi |
|---|:---:|:---:|:---:|
| 02 Baseline | — (bobot ImageNet mentah) | ✅ | ✅ |
| 03 Lightweight Fine Tuning | ✅ | ✅ | ✅ |
| 04 Partial Unfreeze | ✅ | ✅ | ✅ |
| 06 Text CBF One-Hot | — | ✅ | ✅ |
| 08 User Study | — | — | ✅ (online, A/B testing) |

---

## 🚀 Cara Menjalankan

### Langkah 1 — Persiapan Data (Google Colab)
1. Upload `experiments/01_data_preparation/kode/01_data_preparation_revisi.ipynb` ke Colab
2. Sesuaikan `BASE_DIR` dengan path Google Drive kamu
3. Jalankan semua cell → menghasilkan `master_dataset.csv` (salinan final tersedia di `experiments/01_data_preparation/dataset_output/`)

### Langkah 2 — Baseline: Ekstraksi Fitur (Colab GPU)
1. Upload `experiments/02_baseline/kode/baseline_feature_extraction.ipynb`
2. Jalankan → menghasilkan fitur 4 arsitektur CNN dengan bobot ImageNet (tanpa fine-tuning)
3. Evaluasi: `baseline_retrieval_evaluation.ipynb` dan `baseline_recommender_eval.ipynb`

### Langkah 3 — Lightweight Fine Tuning (Colab GPU)
1. Training: `experiments/03_lightweight_finetuning/kode/lightweight_training.ipynb`
2. Ekstraksi fitur: `lightweight_feature_extraction.ipynb`
3. Evaluasi: `lightweight_klasifikasi_eval.ipynb` → `lightweight_retrieval_eval.ipynb` → `recommender_eval_lightweight.ipynb`

### Langkah 4 — Partial Unfreeze (Colab GPU)
1. Training: `experiments/04_partial_unfreeze/kode/partial_unfreeze_training.ipynb`
2. Ekstraksi fitur: `partial_unfreeze_feature_extraction.ipynb`
3. Evaluasi: `partial_unfreeze_klasifikasi_eval.ipynb` → `partial_unfreeze_retrieval_eval.ipynb` → `recommender_eval_partial_unfreeze.ipynb`
4. Analisis per kategori: `experiments/05_retrieval_analysis/kode/05_retrieval_evaluation.py`

### Langkah 5 — Baseline Teks One-Hot
1. `experiments/06_text_cbf_onehot/kode/build_clean_onehot.ipynb` → matriks one-hot
2. `analyze_leakage.py` → deteksi atribut proxy label
3. `evaluate_text_cbf_onehot.py` / `evaluate_retrieval_onehot.py` → evaluasi

### Langkah 6 — Visualisasi & User Study
1. Visualisasi: `experiments/07_nn_visualization/kode/08_category_nn_visualization.ipynb`
2. Analisis user study: `experiments/08_user_study/kode/09_user_study_analysis.ipynb` (data: `08_user_study/data/fashion_user_study.csv`)

### Langkah 7 — Jalankan Streamlit App (Lokal)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Dev lokal: siapkan features/ & dataset/ sesuai path di app/config.py,
#    atau jalankan prepare_deploy_data.py lalu upload arsip ke GitHub Releases
#    (app akan auto-download saat first run)

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
