"""
=============================================================
  VISUAL-BASED FASHION RECOMMENDER SYSTEM
  Notebook 03 — Evaluation
  Jalankan di Google Colab
=============================================================
Tujuan:
  - Evaluasi performa rekomendasi tiap model CNN
  - Metrik: Precision@K, Recall@K, F1@K, NDCG@K, Diversity
  - Ground truth berbasis kategori (intra-category relevance)
  - Buat tabel perbandingan dan visualisasi antar model
=============================================================
"""

# ─────────────────────────────────────────────
# CELL 1: Mount & Imports
# ─────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# CELL 2: Konfigurasi Path
# ─────────────────────────────────────────────
BASE_DIR   = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
FEAT_DIR   = os.path.join(BASE_DIR, 'features')
EVAL_DIR   = os.path.join(BASE_DIR, 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_NAMES = ['resnet50', 'vgg19', 'inceptionv3', 'mobilenetv3']
K_VALUES    = [5, 10, 20]      # Evaluate at K = 5, 10, 20
N_USERS     = 100              # Jumlah simulasi user untuk evaluasi
N_LIKED     = 3                # Item awal yang "disukai" tiap user (cold-start phase)

print("✅ Config loaded.")


# ─────────────────────────────────────────────
# CELL 3: Load Data
# ─────────────────────────────────────────────
df = pd.read_csv(os.path.join(FEAT_DIR, 'image_index.csv'))
print(f"📊 Total items: {len(df):,}")
print(f"📊 Categories : {df['category'].nunique()}")

# Pastikan semua kategori mempunyai cukup item
cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 20].index.tolist()
df = df[df['category'].isin(valid_cats)].reset_index(drop=True)
print(f"📊 Items after filtering (≥20 per category): {len(df):,}")


# ─────────────────────────────────────────────
# CELL 4: Metrik Evaluasi
# ─────────────────────────────────────────────
def precision_at_k(recommended_ids, relevant_ids, k):
    """Precision@K: proporsi item relevan di top-K rekomendasi."""
    top_k = recommended_ids[:k]
    hits  = len(set(top_k) & set(relevant_ids))
    return hits / k


def recall_at_k(recommended_ids, relevant_ids, k):
    """Recall@K: proporsi item relevan yang berhasil ditemukan di top-K."""
    if len(relevant_ids) == 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits  = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def f1_at_k(recommended_ids, relevant_ids, k):
    """F1@K: harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(recommended_ids, relevant_ids, k)
    r = recall_at_k(recommended_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def dcg_at_k(recommended_ids, relevant_ids, k):
    """DCG@K: Discounted Cumulative Gain."""
    top_k = recommended_ids[:k]
    dcg   = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # posisi 1-indexed, log base 2
    return dcg


def ndcg_at_k(recommended_ids, relevant_ids, k):
    """NDCG@K: Normalized DCG."""
    actual_dcg  = dcg_at_k(recommended_ids, relevant_ids, k)
    # Ideal DCG: semua item relevan di posisi teratas
    ideal_order = list(relevant_ids)[:k]
    ideal_dcg   = dcg_at_k(ideal_order, relevant_ids, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def intra_list_diversity(recommended_ids, feature_matrix, item_id_to_idx):
    """
    Diversity: rata-rata jarak (1 - cosine_sim) antar item di dalam daftar rekomendasi.
    Semakin tinggi = semakin beragam.
    """
    indices = [item_id_to_idx[i] for i in recommended_ids if i in item_id_to_idx]
    if len(indices) < 2:
        return 0.0
    vecs = feature_matrix[indices]
    # Cosine similarity dari vektor L2-normalized = dot product
    sim_matrix = vecs @ vecs.T
    n          = len(indices)
    diversity  = 0.0
    count      = 0
    for i in range(n):
        for j in range(i+1, n):
            diversity += 1 - sim_matrix[i, j]
            count += 1
    return diversity / count if count > 0 else 0.0


# ─────────────────────────────────────────────
# CELL 5: Recommendation Helper
# ─────────────────────────────────────────────
def build_user_profile(liked_indices, feature_matrix):
    """Rata-rata feature vector dari item yang disukai user."""
    vecs    = feature_matrix[liked_indices]
    profile = np.mean(vecs, axis=0)
    norm    = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


def get_recommendations(user_profile, feature_matrix, exclude_indices, top_n=10):
    """
    Cosine similarity antara user_profile dan seluruh database.
    Exclude item yang sudah dilihat/disukai user.
    Return: list of (item_idx, score)
    """
    scores      = feature_matrix @ user_profile          # dot product (L2-normalized)
    scores[exclude_indices] = -1.0                       # exclude seen items
    ranked_idx  = np.argsort(scores)[::-1]               # descending
    ranked_idx  = [i for i in ranked_idx if i not in set(exclude_indices)]
    ranked_scores = [(i, float(scores[i])) for i in ranked_idx[:top_n]]
    return ranked_scores


# ─────────────────────────────────────────────
# CELL 6: Simulasi User & Evaluasi
# ─────────────────────────────────────────────
def simulate_evaluation(feature_matrix, df, n_users=100, n_liked=3, k_values=[5,10,20]):
    """
    Simulasi N user:
    - Setiap user dipilih dari 1 kategori secara acak
    - n_liked item dari kategori tersebut menjadi "liked items" (query)
    - Sisa item sekategori = ground truth (relevant set)
    - Rekomendasi diambil dari seluruh database, diukur relevansinya
    """
    n_items       = len(df)
    item_id_to_idx= {row['item_id']: idx for idx, row in df.iterrows()}
    categories    = df['category'].unique()

    results = {k: {'precision': [], 'recall': [], 'f1': [], 'ndcg': [], 'diversity': []}
               for k in k_values}

    rng = np.random.default_rng(42)

    for user_i in tqdm(range(n_users), desc="Simulating users"):
        # Pilih kategori acak
        cat         = rng.choice(categories)
        cat_items   = df[df['category'] == cat]['item_id'].tolist()
        if len(cat_items) < n_liked + 5:
            continue

        # Pilih liked items
        liked_ids   = rng.choice(cat_items, size=n_liked, replace=False).tolist()
        liked_idx   = [item_id_to_idx[i] for i in liked_ids]

        # Ground truth: semua item kategori yang sama, kecuali liked
        relevant_ids= [i for i in cat_items if i not in set(liked_ids)]

        # Build user profile
        user_profile = build_user_profile(liked_idx, feature_matrix)

        # Dapatkan rekomendasi
        max_k        = max(k_values)
        recs         = get_recommendations(user_profile, feature_matrix,
                                           exclude_indices=liked_idx, top_n=max_k)
        rec_ids      = [df.iloc[idx]['item_id'] for idx, _ in recs]
        rec_idx_list = [idx for idx, _ in recs]

        # Hitung metrik untuk tiap K
        for k in k_values:
            p  = precision_at_k(rec_ids, relevant_ids, k)
            r  = recall_at_k(rec_ids, relevant_ids, k)
            f1 = f1_at_k(rec_ids, relevant_ids, k)
            nd = ndcg_at_k(rec_ids, relevant_ids, k)
            dv = intra_list_diversity(rec_ids[:k], feature_matrix, item_id_to_idx)

            results[k]['precision'].append(p)
            results[k]['recall'].append(r)
            results[k]['f1'].append(f1)
            results[k]['ndcg'].append(nd)
            results[k]['diversity'].append(dv)

    # Rata-rata
    summary = {}
    for k in k_values:
        summary[k] = {m: np.mean(v) for m, v in results[k].items()}
    return summary


# ─────────────────────────────────────────────
# CELL 7: Jalankan Evaluasi untuk Semua Model
# ─────────────────────────────────────────────
all_results = {}

for model_name in MODEL_NAMES:
    feat_path = os.path.join(FEAT_DIR, f'{model_name}_features_exp3.npy')
    idx_path  = os.path.join(FEAT_DIR, f'{model_name}_valid_idx_exp3.npy')

    if not os.path.exists(feat_path):
        print(f"⚠️ Feature file tidak ditemukan untuk {model_name}, skip.")
        continue

    print(f"\n{'='*50}")
    print(f"  📊 Evaluating: {model_name.upper()}")
    print(f"{'='*50}")

    # Load features
    feat_mat  = np.load(feat_path)
    valid_idx = np.load(idx_path).tolist()

    # Subset df sesuai valid_idx
    df_sub    = df.iloc[valid_idx].reset_index(drop=True)
    df_sub['item_id'] = range(len(df_sub))

    # Evaluasi
    summary   = simulate_evaluation(
        feature_matrix= feat_mat,
        df            = df_sub,
        n_users       = N_USERS,
        n_liked       = N_LIKED,
        k_values      = K_VALUES,
    )
    all_results[model_name] = summary

    # Print per model
    for k, metrics in summary.items():
        print(f"  K={k}: P={metrics['precision']:.4f} | R={metrics['recall']:.4f} | "
              f"F1={metrics['f1']:.4f} | NDCG={metrics['ndcg']:.4f} | Div={metrics['diversity']:.4f}")

    del feat_mat
    import gc; gc.collect()


# ─────────────────────────────────────────────
# CELL 8: Buat Tabel Perbandingan
# ─────────────────────────────────────────────
rows = []
for model_name, result in all_results.items():
    for k, metrics in result.items():
        rows.append({
            'Model'    : model_name,
            'K'        : k,
            'Precision': round(metrics['precision'],  4),
            'Recall'   : round(metrics['recall'],     4),
            'F1'       : round(metrics['f1'],         4),
            'NDCG'     : round(metrics['ndcg'],       4),
            'Diversity': round(metrics['diversity'],  4),
        })

df_eval = pd.DataFrame(rows)
print("\n📊 TABEL EVALUASI LENGKAP:")
print(df_eval.to_string(index=False))

# Simpan
eval_csv = os.path.join(EVAL_DIR, 'evaluation_results.csv')
df_eval.to_csv(eval_csv, index=False)
print(f"\n💾 Tersimpan: {eval_csv}")


# ─────────────────────────────────────────────
# CELL 9: Visualisasi — Bar Chart per Metrik per K
# ─────────────────────────────────────────────
metrics_to_plot = ['Precision', 'Recall', 'F1', 'NDCG', 'Diversity']
colors          = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

for k in K_VALUES:
    df_k = df_eval[df_eval['K'] == k]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 4))
    fig.suptitle(f'Evaluasi Sistem Rekomendasi @ K={k}', fontsize=13, fontweight='bold')

    for ax, metric in zip(axes, metrics_to_plot):
        bars = ax.bar(df_k['Model'], df_k[metric], color=colors, width=0.5)
        ax.set_title(f'{metric}@{k}', fontsize=11)
        ax.set_ylim(0, max(df_k[metric].max() * 1.2, 0.1))
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + 0.002,
                    f'{bar.get_height():.4f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f'evaluation_k{k}.png'), dpi=150, bbox_inches='tight')

print("✅ Semua visualisasi tersimpan.")


# ─────────────────────────────────────────────
# CELL 10: Heatmap — Semua Metrik × Model × K
# ─────────────────────────────────────────────
for metric in metrics_to_plot:
    pivot = df_eval.pivot(index='Model', columns='K', values=metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_title(f'{metric} — Heatmap (Model × K)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f'heatmap_{metric.lower()}.png'), dpi=150)


# ─────────────────────────────────────────────
# CELL 11: Kesimpulan Otomatis
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  📌 KESIMPULAN EVALUASI")
print("="*55)
k10 = df_eval[df_eval['K'] == 10]
for metric in ['F1', 'NDCG', 'Diversity']:
    best_row = k10.loc[k10[metric].idxmax()]
    print(f"  Best {metric}@10 : {best_row['Model'].upper()} ({best_row[metric]:.4f})")
print("="*55)
print("\n➡️ Lanjut ke Streamlit App")
