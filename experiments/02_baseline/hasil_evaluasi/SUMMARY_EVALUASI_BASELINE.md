# Evaluasi Eksperimen 1: Baseline (Pre-trained ImageNet)

## 1. Klasifikasi
**N/A** - Model baseline tidak di-fine-tune, sehingga evaluasi klasifikasi tidak berlaku.

---

## 2. Image Retrieval
- **Dataset**: `full_dataset.csv` (14.218 query, 12.612 gallery)
- **Fitur**: Ekstraksi fitur visual dari model pre-trained ImageNet tanpa fine-tuning
- **Metode**: Cosine similarity pada fitur L2-normalized
- **Relevansi**: Exact item matching (`item_id` query == `item_id` gallery)
- **Sumber hasil**: `retrieval_evaluation_baseline.csv`

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3** (Terbaik) | **0.3389** | **0.5186** | **0.5969** | **0.6677** | **0.1944** |
| ResNet50 | 0.2946 | 0.4605 | 0.5282 | 0.5978 | 0.1555 |
| InceptionV3 | 0.2673 | 0.4352 | 0.5068 | 0.5801 | 0.1433 |
| VGG19 | 0.2244 | 0.3710 | 0.4354 | 0.5043 | 0.1137 |

---

## 3. Sistem Rekomendasi
- **Dataset**: `master_dataset_gallery.csv` (3.981 item representatif, bebas leak)
- **Fitur**: Fitur gallery dari model pre-trained ImageNet tanpa fine-tuning
- **Metode**: Simulasi 100 user (3 liked items/user), cosine similarity pada fitur L2-normalized
- **Relevansi**: Intra-category relevance
- **Sumber hasil**: `recommender_evaluation_baseline.csv`

| Model | K | Precision | Recall | F1 | NDCG | Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 | 5 | 0.2860 | 0.0111 | 0.0201 | 0.2979 | 0.1186 |
| ResNet50 | 10 | 0.2700 | 0.0201 | 0.0333 | 0.2826 | 0.1270 |
| ResNet50 | 20 | 0.2635 | 0.0434 | 0.0608 | 0.2742 | 0.1341 |
| VGG19 | 5 | 0.2700 | 0.0112 | 0.0200 | 0.2672 | 0.1076 |
| VGG19 | 10 | 0.2570 | 0.0231 | 0.0373 | 0.2590 | 0.1146 |
| VGG19 | 20 | 0.2440 | 0.0412 | 0.0575 | 0.2503 | 0.1220 |
| **InceptionV3** (Precision Terbaik) | 5 | **0.3380** | 0.0124 | 0.0227 | **0.3400** | 0.1508 |
| **InceptionV3** | 10 | **0.3120** | 0.0254 | 0.0417 | **0.3215** | 0.1590 |
| **InceptionV3** | 20 | **0.2885** | 0.0474 | **0.0676** | **0.3026** | 0.1676 |
| MobileNetV3 | 5 | 0.3180 | 0.0165 | 0.0286 | 0.3286 | 0.1600 |
| MobileNetV3 | 10 | 0.2920 | 0.0288 | 0.0448 | 0.3073 | 0.1697 |
| **MobileNetV3** (Diversity Terbaik) | 20 | 0.2780 | **0.0484** | 0.0673 | 0.2931 | **0.1792** |
