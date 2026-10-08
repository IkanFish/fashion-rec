# Baseline: Text-Based Content-Based Filtering (One-Hot Encoding)

## Deskripsi
Baseline konvensional menggunakan **One-Hot Encoding + Cosine Similarity** berbasis metadata atribut dari dataset DeepFashion In-Shop. Baseline ini digunakan untuk perbandingan apple-to-apple terhadap pendekatan visual CNN.

## Fitur yang Digunakan
- **463 atribut pakaian** dari `list_attr_cloth.txt` + `list_attr_items.txt`
- **Informasi warna** dari `list_color_cloth.txt`
- **Saringan Kebocoran Data (Data Leakage Filter):** 10 atribut yang berkorelasi dominan (>70%) memihak langsung pada kategori pakaian target dihapus dari pemrosesan.
- **TANPA nama kategori** (karena ground truth berbasis kategori)

## Protokol Evaluasi (Identik dengan CNN)
- 100 simulated users, `seed=42`
- 3 liked items per user (cold-start)
- Category-based ground truth
- Metrik: Precision@K, Recall@K, F1@K, NDCG@K, Diversity
- K = 5, 10, 20

## Cara Menjalankan
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan evaluasi
python evaluate_text_cbf_onehot.py
```

## Tidak Memerlukan GPU
Semua operasi berjalan di CPU. Estimasi waktu: 1-2 menit.
