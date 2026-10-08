# ─────────────────────────────────────────────
# CELL 1: Import Library & Setup
# ─────────────────────────────────────────────
import os, time, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✅ Library loaded!")


# ─────────────────────────────────────────────
# CELL 2: Konfigurasi Path & Model
# ─────────────────────────────────────────────
if os.name == 'posix':
    BASE_DIR = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
else:
    BASE_DIR = r'D:\Antigravity\Visual Based Rekomender Sistem'

FULL_CSV = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')
if not os.path.exists(FULL_CSV):
    FULL_CSV = os.path.join(BASE_DIR, 'skripsi', 'persiapan_skripsi', 'data preperation', 'dataset_output', 'full_dataset.csv')

IMG_DIR  = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')
FEAT_DIR = os.path.join(BASE_DIR, 'features', 'exp2_lightweight')
EVAL_DIR = os.path.join(BASE_DIR, 'skripsi', 'persiapan_skripsi', 'lightweight', 'hasil_evaluasi')

os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_NAMES = ['resnet50', 'vgg19', 'inceptionv3', 'mobilenetv3']
K_VALUES    = [1, 5, 10, 20, 50]

print("✅ Path configuration loaded!")
print(f"📁 Fitur dibaca dari : {FEAT_DIR}")
print(f"📁 Hasil disimpan di : {EVAL_DIR}")


# ─────────────────────────────────────────────
# CELL 3: Load Metadata Query & Gallery
# ─────────────────────────────────────────────
print("\n📦 Memuat metadata dataset full_dataset.csv...")
df_full = pd.read_csv(FULL_CSV)

df_query   = df_full[df_full['split'] == 'query'].reset_index(drop=True)
df_gallery = df_full[df_full['split'] == 'gallery'].reset_index(drop=True)

print(f"📊 Total Query   : {len(df_query):,}")
print(f"📊 Total Gallery : {len(df_gallery):,}")


# ─────────────────────────────────────────────
# CELL 4: Fungsi Evaluasi Retrieval (Recall@K & mAP)
# ─────────────────────────────────────────────
def evaluate_retrieval_batch(query_features, gallery_features, 
                             query_item_ids, gallery_item_ids, 
                             query_categories, k_values=[1, 5, 10, 20, 50]):
    """
    Menghitung Recall@K, mAP, dan Recall@5 per kategori pakaian.
    Memproses similarity matrix secara berjarak (batch) untuk hemat RAM.
    """
    n_queries = len(query_features)
    recall_at_k = {k: 0 for k in k_values}
    average_precisions = []
    
    # Tracking Recall@5 per Kategori
    category_recall5 = {}
    category_counts  = {}

    batch_size = 500 # Proses 500 query per batch
    
    for start in tqdm(range(0, n_queries, batch_size), desc="Calculating Cosine Similarity & Recall@K"):
        end = min(start + batch_size, n_queries)
        q_batch = query_features[start:end]
        
        # Cosine Similarity (Dot product dari vektor L2-normalized)
        sim_batch = q_batch @ gallery_features.T
        
        for idx in range(end - start):
            i = start + idx
            query_id  = query_item_ids[i]
            cat_label = query_categories[i]
            
            # Ground truth: gallery indices yang memiliki item_id yang sama
            gt_mask = (gallery_item_ids == query_id)
            n_relevant = gt_mask.sum()
            
            if n_relevant == 0:
                continue # Skip jika tidak ada pasangan di gallery
                
            sorted_indices = np.argsort(-sim_batch[idx])
            
            # Recall@K
            has_hit_k5 = False
            for k in k_values:
                top_k_ids = gallery_item_ids[sorted_indices[:k]]
                if query_id in top_k_ids:
                    recall_at_k[k] += 1
                    if k == 5:
                        has_hit_k5 = True
                        
            # Accumulate per Category for Recall@5
            if cat_label not in category_recall5:
                category_recall5[cat_label] = 0
                category_counts[cat_label]  = 0
            category_counts[cat_label] += 1
            if has_hit_k5:
                category_recall5[cat_label] += 1

            # mAP Calculation
            ap = 0.0
            n_correct = 0
            for rank, g_idx in enumerate(sorted_indices):
                if gt_mask[g_idx]:
                    n_correct += 1
                    ap += n_correct / (rank + 1)
            ap /= n_relevant
            average_precisions.append(ap)

    n_eval = len(average_precisions)
    
    summary_results = {}
    for k in k_values:
        summary_results[f'Recall@{k}'] = (recall_at_k[k] / n_eval * 100) if n_eval > 0 else 0.0
        
    summary_results['mAP'] = (np.mean(average_precisions) * 100) if n_eval > 0 else 0.0
    summary_results['n_queries'] = n_eval
    
    # Category Recall@5 breakdown (%)
    cat_summary = {}
    for cat, total in category_counts.items():
        cat_summary[cat] = (category_recall5[cat] / total * 100) if total > 0 else 0.0
        
    return summary_results, cat_summary


# ─────────────────────────────────────────────
# CELL 5: Evaluasi Semua Model Fine-Tuned (Exp 2)
# ─────────────────────────────────────────────
print("\n" + "="*65)
print("  🚀 EVALUASI RETRIEVAL (LIGHTWEIGHT FINE-TUNING)")
print("="*65)

overall_metrics = []
category_metrics = []

for model_name in MODEL_NAMES:
    print(f"\n🧠 Evaluasi Model: {model_name.upper()}")
    
    q_feat_path = os.path.join(FEAT_DIR, f'{model_name}_query_features.npy')
    g_feat_path = os.path.join(FEAT_DIR, f'{model_name}_gallery_features.npy')
    q_idx_path  = os.path.join(FEAT_DIR, f'{model_name}_query_valid_idx.npy')
    g_idx_path  = os.path.join(FEAT_DIR, f'{model_name}_gallery_valid_idx.npy')
    
    if not (os.path.exists(q_feat_path) and os.path.exists(g_feat_path)):
        print(f"  ⚠️ Warning: File fitur {model_name} tidak ditemukan di {FEAT_DIR}! Skipping...")
        continue
        
    q_feats = np.load(q_feat_path)
    g_feats = np.load(g_feat_path)
    q_valid = np.load(q_idx_path)
    g_valid = np.load(g_idx_path)
    
    q_item_ids   = df_query.iloc[q_valid]['item_id'].values
    g_item_ids   = df_gallery.iloc[g_valid]['item_id'].values
    q_categories = df_query.iloc[q_valid]['category'].values
    
    print(f"  📌 Query Matrix   : {q_feats.shape}")
    print(f"  📌 Gallery Matrix : {g_feats.shape}")
    
    summary, cat_summary = evaluate_retrieval_batch(
        q_feats, g_feats, q_item_ids, g_item_ids, q_categories, K_VALUES
    )
    
    row_summary = {'model': model_name.upper(), **summary}
    overall_metrics.append(row_summary)
    
    for cat, r5 in cat_summary.items():
        category_metrics.append({
            'model': model_name.upper(),
            'category': cat,
            'Recall@5': r5
        })
        
    print(f"  ✅ Recall@1 : {summary['Recall@1']:.2f}%")
    print(f"  ✅ Recall@5 : {summary['Recall@5']:.2f}%")
    print(f"  ✅ Recall@10: {summary['Recall@10']:.2f}%")
    print(f"  ✅ Recall@20: {summary['Recall@20']:.2f}%")
    print(f"  ✅ Recall@50: {summary['Recall@50']:.2f}%")
    print(f"  ✅ mAP      : {summary['mAP']:.2f}%")
    
    del q_feats, g_feats
    gc.collect()


# ─────────────────────────────────────────────
# CELL 6: Simpan Laporan Evaluasi & Visualisasi
# ─────────────────────────────────────────────
if overall_metrics:
    df_results = pd.DataFrame(overall_metrics)
    csv_path = os.path.join(EVAL_DIR, 'retrieval_evaluation_lightweight.csv')
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*65)
    print("  📊 SUMMARY HASIL RETRIEVAL (LIGHTWEIGHT FINE-TUNING)")
    print("="*65)
    print(df_results.to_string(index=False))
    
    # Plot Perbandingan Recall@K
    plt.figure(figsize=(10, 6))
    for res in overall_metrics:
        m_name = res['model']
        r_vals = [res[f'Recall@{k}'] for k in K_VALUES]
        plt.plot(K_VALUES, r_vals, marker='o', linewidth=2, label=m_name)
        
    plt.title('Perbandingan Recall@K (Lightweight Fine-Tuning)', fontsize=14)
    plt.xlabel('K (Top-K Items)')
    plt.ylabel('Recall (%)')
    plt.xticks(K_VALUES)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    chart_path = os.path.join(EVAL_DIR, 'retrieval_recall_comparison_lightweight.png')
    plt.savefig(chart_path, dpi=300)
    plt.show()
    plt.close()
    
    print(f"\n💾 Tabel hasil disimpan di : {csv_path}")
    print(f"💾 Grafik perbandingan di   : {chart_path}")
else:
    print("\n⚠️ Tidak ada fitur model yang dievaluasi.")
