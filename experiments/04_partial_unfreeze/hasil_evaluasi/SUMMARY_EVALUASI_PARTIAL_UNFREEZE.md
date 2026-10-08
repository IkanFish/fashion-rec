# Evaluasi Eksperimen 3: Partial Unfreeze Fine-Tuning

## 1. Klasifikasi
- **Dataset**: `full_dataset.csv` subset train dengan validation split 20% (5.174 sampel validasi)
- **Metode**: Evaluasi model partial unfreeze fine-tuned menggunakan classification report
- **Sumber hasil**: `partial_unfreeze_classification_results.csv` dan `*_classification_report.txt`

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VGG19** (Accuracy Terbaik) | **0.7457** | 0.7148 | **0.6715** | 0.6832 | **0.7500** | **0.7500** | **0.7400** |
| **ResNet50** (Macro F1 Terbaik) | 0.7410 | **0.7271** | 0.6635 | **0.6900** | 0.7400 | 0.7400 | **0.7400** |
| MobileNetV3 | 0.7319 | 0.7186 | 0.6433 | 0.6732 | 0.7300 | 0.7300 | 0.7300 |
| InceptionV3 | 0.7296 | 0.7132 | 0.6374 | 0.6626 | 0.7200 | 0.7300 | 0.7200 |

---

## 2. Image Retrieval
- **Dataset**: `full_dataset.csv` (14.218 query, 12.612 gallery)
- **Fitur**: Ekstraksi fitur visual dari model partial unfreeze fine-tuned
- **Metode**: Cosine similarity pada fitur L2-normalized
- **Relevansi**: Exact item matching (`item_id` query == `item_id` gallery)
- **Sumber hasil**: `partial_unfreeze_retrieval_overall_results.csv`
- **Catatan**: Nilai pada CSV tersimpan dalam persen, sedangkan tabel berikut dinormalisasi ke format desimal agar konsisten dengan summary baseline.

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3** (Terbaik) | **0.3887** | **0.5962** | **0.6771** | **0.7468** | **0.8294** | **0.2375** |
| VGG19 | 0.3292 | 0.5223 | 0.6059 | 0.6859 | 0.7768 | 0.1948 |
| ResNet50 | 0.3242 | 0.5224 | 0.5997 | 0.6776 | 0.7699 | 0.1901 |
| InceptionV3 | 0.2506 | 0.4297 | 0.5129 | 0.5948 | 0.7052 | 0.1425 |

---

## 3. Image Retrieval per Kategori
- **Metrik**: Recall@5 per kategori
- **Sumber hasil**: `retrieval_category_evaluation_partial_unfreeze.csv`
- **Catatan**: Nilai pada CSV tersimpan dalam persen, sedangkan tabel berikut dinormalisasi ke format desimal.

| Kategori | ResNet50 | VGG19 | InceptionV3 | MobileNetV3 |
| :--- | :---: | :---: | :---: | :---: |
| Blouses_Shirts | 0.4374 | 0.4394 | 0.3223 | **0.5411** |
| Cardigans | 0.4774 | 0.4874 | 0.3844 | **0.5980** |
| Denim | 0.4660 | 0.4503 | 0.4136 | **0.5026** |
| Dresses | 0.5786 | 0.5944 | 0.4719 | **0.6607** |
| Graphic_Tees | 0.2411 | 0.2192 | 0.1397 | **0.2877** |
| Jackets_Coats | 0.4183 | 0.4275 | 0.3431 | **0.5193** |
| Jackets_Vests | 0.6444 | **0.6667** | 0.5667 | **0.6667** |
| Leggings | 0.6029 | **0.6324** | 0.4853 | 0.6176 |
| Pants | 0.6098 | **0.6331** | 0.5362 | 0.6172 |
| Rompers_Jumpsuits | 0.6070 | 0.5905 | 0.4774 | **0.6502** |
| Shirts_Polos | 0.5872 | 0.5596 | 0.4725 | **0.6468** |
| Shorts | 0.5436 | 0.5468 | 0.4516 | **0.5837** |
| Skirts | 0.6775 | 0.7427 | 0.6401 | **0.7720** |
| Suiting | 0.4286 | 0.5000 | 0.2857 | **0.6429** |
| Sweaters | 0.4957 | 0.4732 | 0.3895 | **0.5987** |
| Sweatshirts_Hoodies | 0.5442 | 0.5102 | 0.4422 | **0.6077** |
| Tees_Tanks | 0.5252 | 0.5117 | 0.4456 | **0.5968** |

---

## 4. Sistem Rekomendasi
- **Dataset**: `master_dataset_gallery.csv` (3.981 item representatif, bebas leak)
- **Fitur**: Fitur gallery dari model partial unfreeze fine-tuned
- **Metode**: Simulasi 100 user (3 liked items/user), cosine similarity pada fitur L2-normalized
- **Relevansi**: Intra-category relevance
- **Sumber hasil**: `recommender_evaluation_partial_unfreeze.csv`

| Model | K | Precision | Recall | F1 | NDCG | Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 | 5 | 0.6100 | 0.0313 | 0.0546 | 0.6086 | 0.2080 |
| ResNet50 | 10 | 0.5900 | 0.0581 | 0.0920 | 0.5947 | 0.2263 |
| ResNet50 | 20 | 0.5760 | 0.1084 | 0.1496 | 0.5864 | 0.2473 |
| **VGG19** (Precision/Recall/F1/NDCG Terbaik) | 5 | **0.6780** | **0.0377** | **0.0655** | **0.6742** | 0.2512 |
| **VGG19** | 10 | **0.6480** | **0.0701** | **0.1096** | **0.6544** | 0.2728 |
| **VGG19** | 20 | **0.5910** | **0.1149** | **0.1575** | **0.6156** | 0.2994 |
| InceptionV3 | 5 | 0.6300 | 0.0307 | 0.0540 | 0.6452 | 0.1672 |
| InceptionV3 | 10 | 0.5870 | 0.0519 | 0.0841 | 0.6095 | 0.1797 |
| InceptionV3 | 20 | 0.5600 | 0.0997 | 0.1405 | 0.5853 | 0.1950 |
| **MobileNetV3** (Diversity Terbaik) | 5 | 0.6060 | 0.0337 | 0.0585 | 0.6192 | **0.2746** |
| **MobileNetV3** | 10 | 0.5350 | 0.0542 | 0.0854 | 0.5646 | **0.2943** |
| **MobileNetV3** | 20 | 0.4920 | 0.0920 | 0.1269 | 0.5269 | **0.3197** |
