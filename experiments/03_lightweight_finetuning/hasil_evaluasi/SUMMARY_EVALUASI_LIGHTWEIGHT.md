# Evaluasi Eksperimen 2: Lightweight Fine-Tuning

## 1. Klasifikasi
- **Dataset**: `full_dataset.csv` subset train dengan validation split 20% (5.174 sampel validasi)
- **Metode**: Evaluasi model fine-tuned menggunakan classification report dan confusion matrix
- **Sumber hasil**: `*_classification_report.txt`

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** (Accuracy Terbaik) | **0.7100** | 0.6900 | **0.6200** | **0.6500** | **0.7100** | **0.7100** | **0.7100** |
| **MobileNetV3** | 0.7000 | **0.7000** | 0.6100 | 0.6400 | 0.7000 | 0.7000 | 0.7000 |
| InceptionV3 | 0.6800 | 0.6600 | 0.6000 | 0.6200 | 0.6800 | 0.6800 | 0.6800 |
| VGG19 | 0.6400 | 0.6100 | 0.5200 | 0.5500 | 0.6300 | 0.6400 | 0.6300 |

---

## 2. Image Retrieval
- **Dataset**: `full_dataset.csv` (14.218 query, 12.612 gallery)
- **Fitur**: Ekstraksi fitur visual dari model lightweight fine-tuned
- **Metode**: Cosine similarity pada fitur L2-normalized
- **Relevansi**: Exact item matching (`item_id` query == `item_id` gallery)
- **Sumber hasil**: `retrieval_evaluation_lightweight.csv`
- **Catatan**: Nilai pada CSV tersimpan dalam persen, sedangkan tabel berikut dinormalisasi ke format desimal agar konsisten dengan summary baseline.

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3** (Terbaik) | **0.3407** | **0.5371** | **0.6145** | **0.6904** | **0.7846** | **0.2027** |
| ResNet50 | 0.2948 | 0.4805 | 0.5600 | 0.6345 | 0.7314 | 0.1666 |
| InceptionV3 | 0.2239 | 0.3944 | 0.4802 | 0.5660 | 0.6748 | 0.1275 |
| VGG19 | 0.2165 | 0.3712 | 0.4477 | 0.5234 | 0.6239 | 0.1154 |

---

## 3. Image Retrieval per Kategori
- **Metrik**: Recall@5 per kategori
- **Sumber hasil**: `retrieval_category_evaluation_lightweight.csv`
- **Catatan**: Nilai pada CSV tersimpan dalam persen, sedangkan tabel berikut dinormalisasi ke format desimal.

| Kategori | ResNet50 | VGG19 | InceptionV3 | MobileNetV3 |
| :--- | :---: | :---: | :---: | :---: |
| Blouses_Shirts | 0.3777 | 0.2459 | 0.2870 | **0.4532** |
| Cardigans | 0.4372 | 0.3668 | 0.3266 | **0.4925** |
| Denim | 0.4398 | 0.4188 | 0.3717 | **0.4607** |
| Dresses | 0.5502 | 0.4193 | 0.4392 | **0.6039** |
| Graphic_Tees | 0.1644 | 0.0849 | 0.1205 | **0.2356** |
| Jackets_Coats | 0.3376 | 0.2624 | 0.3119 | **0.4514** |
| Jackets_Vests | **0.6111** | 0.5000 | 0.5889 | 0.6000 |
| Leggings | **0.6765** | 0.5294 | 0.5147 | 0.6471 |
| Pants | 0.5877 | 0.5190 | 0.5031 | **0.6061** |
| Rompers_Jumpsuits | 0.5473 | 0.3971 | 0.4362 | **0.6029** |
| Shirts_Polos | 0.5550 | 0.4541 | 0.4450 | **0.6101** |
| Shorts | 0.5076 | 0.4283 | 0.4155 | **0.5396** |
| Skirts | 0.6547 | 0.5749 | 0.6075 | **0.7313** |
| Suiting | **0.5000** | 0.4286 | 0.3571 | **0.5000** |
| Sweaters | 0.4464 | 0.3112 | 0.3616 | **0.5279** |
| Sweatshirts_Hoodies | 0.4762 | 0.3719 | 0.4036 | **0.5079** |
| Tees_Tanks | 0.4891 | 0.3745 | 0.4051 | **0.5425** |

---

## 4. Sistem Rekomendasi
- **Dataset**: `master_dataset_gallery.csv` (3.981 item representatif, bebas leak)
- **Fitur**: Fitur gallery dari model lightweight fine-tuned
- **Metode**: Simulasi 100 user (3 liked items/user), cosine similarity pada fitur L2-normalized
- **Relevansi**: Intra-category relevance
- **Sumber hasil**: `recommender_evaluation_lightweight.csv`

| Model | K | Precision | Recall | F1 | NDCG | Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 | 5 | 0.5020 | 0.0199 | 0.0356 | 0.5066 | 0.2332 |
| ResNet50 | 10 | 0.4790 | 0.0386 | 0.0631 | 0.4885 | 0.2488 |
| ResNet50 | 20 | 0.4550 | 0.0760 | 0.1072 | 0.4708 | 0.2648 |
| VGG19 | 5 | 0.3900 | 0.0196 | 0.0343 | 0.3954 | 0.2893 |
| VGG19 | 10 | 0.3690 | 0.0352 | 0.0557 | 0.3792 | **0.3174** |
| **VGG19** (Diversity Terbaik) | 20 | 0.3680 | 0.0715 | 0.0961 | 0.3775 | **0.3392** |
| **InceptionV3** (Precision Terbaik) | 5 | **0.5160** | 0.0246 | 0.0431 | 0.5248 | 0.1974 |
| **InceptionV3** | 10 | **0.4800** | **0.0485** | **0.0760** | **0.4972** | 0.2113 |
| **InceptionV3** (Recall/F1/NDCG Terbaik) | 20 | **0.4595** | **0.0862** | **0.1184** | **0.4789** | 0.2295 |
| **MobileNetV3** (NDCG@5 Terbaik) | 5 | 0.5080 | **0.0265** | **0.0461** | **0.5249** | 0.2650 |
| MobileNetV3 | 10 | 0.4680 | 0.0477 | 0.0750 | 0.4917 | 0.2826 |
| MobileNetV3 | 20 | 0.4315 | 0.0804 | 0.1102 | 0.4609 | 0.3039 |
