"""
=============================================================
  CATEGORY-LEVEL RETRIEVAL EVALUATION
  CNN vs Text-Based — Head-to-Head Comparison
=============================================================
Tujuan:
  - Mengevaluasi kualitas RUANG VEKTOR kedua paradigma
  - Mengukur kemampuan vektor mengelompokkan item ke kategori yang benar
  - Pendekatan: Category-Level Retrieval (bukan item_id)
  - Ground truth: kategori pakaian (17 kelas)
  - Metrik: CategoryPrecision@K, Category-mAP

Mengapa BUKAN item_id retrieval?
  - CNN: 1 vektor per FOTO → item_id retrieval valid
  - Text: 1 vektor per ITEM_ID → semua foto satu item mendapat vektor IDENTIK
  - Evaluasi item_id pada text = trivial (selalu 100%) → bukan evaluasi valid

  Oleh karena itu, kita gunakan Category-Level Retrieval yang ADIL
  untuk kedua paradigma. Ground truth = kategori (17 kelas).
  Kedua paradigma menghasilkan 1 vektor per item (CNN sudah di-aggregate
  ke master_features yang merupakan per-item, bukan per-foto).

Output:
  - CSV: comparison table semua model
  - Bar charts per K
  - Heatmap Model × K
  - Per-category breakdown
  - Kesimpulan
=============================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
ANNO_DIR   = os.path.join(ROOT_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Anno')
ATTR_DIR   = os.path.join(ANNO_DIR, 'attributes')
EVAL_DIR   = os.path.join(ROOT_DIR, 'evaluation', 'vector_space_comparison')
os.makedirs(EVAL_DIR, exist_ok=True)

K_VALUES   = [5, 10, 20]

# Sumber fitur CNN — master features (per-item, bukan per-foto)
CNN_SOURCES = {
    'ResNet50 (Exp3)': {
        'feat': os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'resnet50_features_exp3.npy'),
        'idx':  os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'resnet50_valid_idx_exp3.npy'),
    },
    'VGG19 (Exp3)': {
        'feat': os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'vgg19_features_exp3.npy'),
        'idx':  os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'vgg19_valid_idx_exp3.npy'),
    },
    'InceptionV3 (Exp3)': {
        'feat': os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'inceptionv3_features_exp3.npy'),
        'idx':  os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'inceptionv3_valid_idx_exp3.npy'),
    },
    'MobileNetV3 (Exp3)': {
        'feat': os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'mobilenetv3_features_exp3.npy'),
        'idx':  os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'mobilenetv3_valid_idx_exp3.npy'),
    },
    'ResNet50 (GridSearch)': {
        'feat': os.path.join(ROOT_DIR, 'grid_search', 'features', 'resnet50_adam_bs24_lr1e-05_uf50', 'master_features.npy'),
        'idx':  os.path.join(ROOT_DIR, 'grid_search', 'features', 'resnet50_adam_bs24_lr1e-05_uf50', 'master_valid_idx.npy'),
    },
    'MobileNetV3 (GridSearch)': {
        'feat': os.path.join(ROOT_DIR, 'grid_search', 'features', 'mobilenetv3_adam_bs24_lr1e-05_uf50', 'master_features.npy'),
        'idx':  os.path.join(ROOT_DIR, 'grid_search', 'features', 'mobilenetv3_adam_bs24_lr1e-05_uf50', 'master_valid_idx.npy'),
    },
}

# Evidence-based leaky keywords (sama persis dengan evaluate_text_cbf_v3.py)
LEAKY_KEYWORDS = {
    'jumpsuit', 'romper', 'sweatshirt', 'shorts',
    'pants', 'blouse', 'hoodie', 'dress',
}

def is_leaky_attribute(attr_name: str) -> bool:
    words = set(attr_name.lower().replace('-', ' ').replace('_', ' ').split())
    return bool(words & LEAKY_KEYWORDS)


# ═══════════════════════════════════════════════
# STEP 1: Load Master Dataset
# ═══════════════════════════════════════════════
print("=" * 70)
print("  STEP 1: Load Master Dataset")
print("=" * 70)

MASTER_CSV = os.path.join(ROOT_DIR, 'dataset', 'master_dataset.csv')
df = pd.read_csv(MASTER_CSV)

cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 20].index.tolist()
df = df[df['category'].isin(valid_cats)].reset_index(drop=True)

print(f"  Total items (setelah filter >=20/cat): {len(df):,}")
print(f"  Kategori valid                      : {df['category'].nunique()}")
print(f"\n  Distribusi per kategori:")
for cat, cnt in df['category'].value_counts().items():
    print(f"    {cat:<30s} {cnt:>5,}")


# ═══════════════════════════════════════════════
# STEP 2: Build Text Feature Matrices
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 2: Build Text Feature Matrices")
print("=" * 70)

# --- Load attributes ---
all_attr_names = []
with open(os.path.join(ATTR_DIR, 'list_attr_cloth.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        if line.strip():
            all_attr_names.append(line.strip())

leaky_indices = set()
for i, attr in enumerate(all_attr_names):
    if is_leaky_attribute(attr):
        leaky_indices.add(i)

item_attrs = {}
with open(os.path.join(ATTR_DIR, 'list_attr_items.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        item_id = parts[0]
        labels  = [int(x) for x in parts[1:]]
        active  = []
        for i, val in enumerate(labels):
            if val == 1 and i not in leaky_indices:
                active.append(all_attr_names[i].split()[0])
        item_attrs[item_id] = active

# --- Load colors ---
item_colors = {}
with open(os.path.join(ATTR_DIR, 'list_color_cloth.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        match = re.search(r'(id_\d+)', parts[0])
        if match:
            item_id = match.group(1)
            if item_id not in item_colors:
                item_colors[item_id] = parts[1]



# --- One-Hot Encoding ---
attr_lists_for_onehot = []
for _, row in df.iterrows():
    item_id = row['item_id']
    attrs   = item_attrs.get(item_id, [])
    color   = item_colors.get(item_id, '')
    parts   = list(attrs)
    if color:
        parts.extend(color.replace('-', ' ').replace('_', ' ').split())
    attr_lists_for_onehot.append(parts if parts else ['unknown'])

mlb = MultiLabelBinarizer()
onehot_matrix = mlb.fit_transform(attr_lists_for_onehot).astype(np.float32)

# L2 normalize One-Hot
oh_norms = np.linalg.norm(onehot_matrix, axis=1, keepdims=True)
oh_norms = np.where(oh_norms == 0, 1e-10, oh_norms)
onehot_matrix_norm = onehot_matrix / oh_norms

print(f"  One-Hot matrix  : {onehot_matrix_norm.shape}")


# ═══════════════════════════════════════════════
# STEP 3: Category-Level Retrieval Function
# ═══════════════════════════════════════════════
def category_retrieval_eval(feature_matrix, categories, k_values=[5, 10, 20], desc="Evaluating"):
    """
    Untuk setiap item, cari Top-K tetangga terdekat (cosine similarity),
    lalu hitung berapa banyak tetangga yang sekategori.

    Karena feature sudah L2-normalized, dot product = cosine similarity.

    Returns:
        summary: dict[k] -> {'CatPrecision': float, 'Cat-mAP': float}
        per_cat_summary: dict[k] -> dict[category] -> float (precision)
    """
    n = len(categories)

    # Cosine similarity matrix via dot product (features sudah L2-normalized)
    sim_matrix = feature_matrix @ feature_matrix.T
    np.fill_diagonal(sim_matrix, -np.inf)  # exclude self

    results = {k: {'cat_precision': [], 'cat_ap': []} for k in k_values}
    per_category = {k: {} for k in k_values}

    for i in tqdm(range(n), desc=desc):
        query_cat  = categories[i]
        ranked_idx = np.argsort(sim_matrix[i])[::-1]

        # Jumlah item sekategori di seluruh dataset (minus diri sendiri)
        n_same_cat = sum(1 for c in categories if c == query_cat) - 1

        for k in k_values:
            top_k     = ranked_idx[:k]
            hits      = sum(1 for j in top_k if categories[j] == query_cat)
            precision = hits / k
            results[k]['cat_precision'].append(precision)

            # Average Precision for this query
            relevant_count = 0
            ap_sum = 0.0
            for rank, j in enumerate(top_k):
                if categories[j] == query_cat:
                    relevant_count += 1
                    ap_sum += relevant_count / (rank + 1)
            ap = ap_sum / min(k, n_same_cat) if relevant_count > 0 else 0.0
            results[k]['cat_ap'].append(ap)

            # Per-category tracking
            if query_cat not in per_category[k]:
                per_category[k][query_cat] = []
            per_category[k][query_cat].append(precision)

    summary = {}
    for k in k_values:
        summary[k] = {
            'CatPrecision': np.mean(results[k]['cat_precision']),
            'Cat-mAP':      np.mean(results[k]['cat_ap']),
        }

    per_cat_summary = {}
    for k in k_values:
        per_cat_summary[k] = {cat: np.mean(vals) for cat, vals in per_category[k].items()}

    return summary, per_cat_summary


# ═══════════════════════════════════════════════
# STEP 4: Evaluate Text-Based Models
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 4: Evaluate Text-Based Models")
print("=" * 70)

categories_array = df['category'].values

all_results    = {}
all_per_cat    = {}



# One-Hot
print("\n📊 One-Hot Encoding:")
summary, per_cat = category_retrieval_eval(onehot_matrix_norm, categories_array, K_VALUES, desc="One-Hot")
all_results['One-Hot'] = summary
all_per_cat['One-Hot'] = per_cat
for k, m in summary.items():
    print(f"  K={k:2d}: CatPrec={m['CatPrecision']:.4f} | Cat-mAP={m['Cat-mAP']:.4f}")


# ═══════════════════════════════════════════════
# STEP 5: Evaluate CNN Models
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 5: Evaluate CNN Models")
print("=" * 70)

# Master dataset (tanpa filter) untuk mapping valid_idx
df_raw = pd.read_csv(MASTER_CSV)

for model_name, paths in CNN_SOURCES.items():
    if not os.path.exists(paths['feat']):
        print(f"\n⚠️ {model_name}: feature file not found, skipping.")
        continue

    print(f"\n📊 {model_name}:")
    feat_mat  = np.load(paths['feat'])
    valid_idx = np.load(paths['idx']).tolist()

    # Map valid indices ke df_raw untuk mendapatkan kategori
    df_sub = df_raw.iloc[valid_idx].reset_index(drop=True)

    # Filter kategori >= 20
    cat_counts_sub = df_sub['category'].value_counts()
    valid_cats_sub = cat_counts_sub[cat_counts_sub >= 20].index.tolist()
    valid_mask     = df_sub['category'].isin(valid_cats_sub)

    df_eval_ready   = df_sub[valid_mask].reset_index(drop=True)
    feat_mat_ready  = feat_mat[valid_mask]
    cats_ready      = df_eval_ready['category'].values

    # L2 normalize (sudah dinormalisasi, tapi pastikan)
    norms = np.linalg.norm(feat_mat_ready, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    feat_mat_ready = feat_mat_ready / norms

    print(f"  Items: {len(df_eval_ready):,} | Features: {feat_mat_ready.shape[1]}")

    summary, per_cat = category_retrieval_eval(feat_mat_ready, cats_ready, K_VALUES, desc=model_name)
    all_results[model_name] = summary
    all_per_cat[model_name] = per_cat

    for k, m in summary.items():
        print(f"  K={k:2d}: CatPrec={m['CatPrecision']:.4f} | Cat-mAP={m['Cat-mAP']:.4f}")

    del feat_mat, feat_mat_ready
    import gc; gc.collect()


# ═══════════════════════════════════════════════
# STEP 6: Build Comparison Table
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 6: Tabel Perbandingan Head-to-Head")
print("=" * 70)

rows = []
for model_name, result in all_results.items():
    paradigm = 'Text-Based' if model_name == 'One-Hot' else 'Visual (CNN)'
    for k, metrics in result.items():
        rows.append({
            'Paradigm':      paradigm,
            'Model':         model_name,
            'K':             k,
            'CatPrecision':  round(metrics['CatPrecision'], 4),
            'Cat-mAP':       round(metrics['Cat-mAP'], 4),
        })

df_comparison = pd.DataFrame(rows)
print(df_comparison.to_string(index=False))

csv_path = os.path.join(EVAL_DIR, 'category_retrieval_comparison.csv')
df_comparison.to_csv(csv_path, index=False)
print(f"\n💾 Saved: {csv_path}")


# ═══════════════════════════════════════════════
# STEP 7: Per-Category Breakdown (K=10)
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 7: Per-Category Breakdown @ K=10")
print("=" * 70)

k_report = 10
per_cat_rows = []
for model_name, per_cat_dict in all_per_cat.items():
    if k_report in per_cat_dict:
        for cat, prec in per_cat_dict[k_report].items():
            per_cat_rows.append({
                'Model': model_name,
                'Category': cat,
                'CatPrecision@10': round(prec, 4),
            })

df_per_cat = pd.DataFrame(per_cat_rows)
pivot_per_cat = df_per_cat.pivot(index='Category', columns='Model', values='CatPrecision@10')

# Sort by mean across all models
pivot_per_cat['Mean'] = pivot_per_cat.mean(axis=1)
pivot_per_cat = pivot_per_cat.sort_values('Mean', ascending=False)
pivot_per_cat = pivot_per_cat.drop(columns=['Mean'])

print(pivot_per_cat.to_string())

csv_per_cat = os.path.join(EVAL_DIR, 'per_category_breakdown_k10.csv')
pivot_per_cat.to_csv(csv_per_cat)
print(f"\n💾 Saved: {csv_per_cat}")


# ═══════════════════════════════════════════════
# STEP 8: Visualisasi
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 8: Visualisasi")
print("=" * 70)

# Color palette
colors_map = {
    'One-Hot': '#e74c3c',
}
cnn_colors = ['#2ecc71', '#3498db', '#9b59b6', '#1abc9c', '#34495e', '#f39c12']
cnn_idx = 0
for name in all_results:
    if name not in colors_map:
        colors_map[name] = cnn_colors[cnn_idx % len(cnn_colors)]
        cnn_idx += 1

# --- 8a: Bar Charts per K ---
for k in K_VALUES:
    df_k = df_comparison[df_comparison['K'] == k].sort_values('CatPrecision', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df_k) * 0.7)))

    bar_colors = [colors_map.get(m, '#95a5a6') for m in df_k['Model']]
    bars = ax.barh(df_k['Model'], df_k['CatPrecision'], color=bar_colors, height=0.5, edgecolor='white')

    for bar, val in zip(bars, df_k['CatPrecision']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Category Precision', fontsize=12)
    ax.set_title(f'Category-Level Retrieval @ K={k}\n(CNN vs Text-Based — Head-to-Head)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, min(df_k['CatPrecision'].max() * 1.2, 1.0))
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, label='50% baseline')

    plt.tight_layout()
    fig_path = os.path.join(EVAL_DIR, f'category_retrieval_k{k}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {fig_path}")

# --- 8b: Heatmap ---
pivot = df_comparison.pivot(index='Model', columns='K', values='CatPrecision')
fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.6 + 1)))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax,
            linewidths=0.5, vmin=0.3, vmax=1.0)
ax.set_title('Category Precision — Heatmap (Model × K)', fontweight='bold', fontsize=13)
plt.tight_layout()
fig_path = os.path.join(EVAL_DIR, 'category_retrieval_heatmap.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {fig_path}")

# --- 8c: Per-Category Heatmap ---
fig, ax = plt.subplots(figsize=(14, max(6, len(pivot_per_cat) * 0.5 + 1)))
sns.heatmap(pivot_per_cat, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
            linewidths=0.5, vmin=0.2, vmax=1.0)
ax.set_title('CatPrecision@10 per Category × Model', fontweight='bold', fontsize=13)
plt.tight_layout()
fig_path = os.path.join(EVAL_DIR, 'per_category_heatmap_k10.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Saved: {fig_path}")

# --- 8d: Grouped Bar Chart — One-Hot (Text) vs Best CNN per Category ---
cnn_models = [m for m in all_results.keys() if m != 'One-Hot']

if cnn_models and k_report in all_per_cat.get('One-Hot', {}):
    cats_sorted = pivot_per_cat.index.tolist()
    
    # Text = One-Hot
    text_vals  = [all_per_cat['One-Hot'][k_report].get(cat, 0) for cat in cats_sorted]
    
    # Best CNN per category
    best_cnn_vals = []
    for cat in cats_sorted:
        best_val = max(all_per_cat[m][k_report].get(cat, 0) for m in cnn_models)
        best_cnn_vals.append(best_val)
    
    x = np.arange(len(cats_sorted))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar(x - width/2, text_vals, width, label='One-Hot (Text)', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x + width/2, best_cnn_vals, width, label='Best CNN', color='#2ecc71', alpha=0.85)
    
    ax.set_ylabel('CatPrecision@10', fontsize=12)
    ax.set_title('Per-Category: One-Hot (Text) vs Best CNN @ K=10', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(cats_sorted, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate difference
    for i, (t, c) in enumerate(zip(text_vals, best_cnn_vals)):
        diff = c - t
        color = '#2ecc71' if diff > 0 else '#e74c3c'
        ax.annotate(f'{diff:+.2f}', (x[i] + width/2, max(t, c) + 0.02),
                   ha='center', fontsize=7, color=color, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(EVAL_DIR, 'text_vs_cnn_per_category_k10.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {fig_path}")


# ═══════════════════════════════════════════════
# STEP 9: Kesimpulan & Analisis
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  📌 KESIMPULAN: Kualitas Ruang Vektor")
print("=" * 70)

for k in K_VALUES:
    k_data = df_comparison[df_comparison['K'] == k]
    
    best_overall = k_data.loc[k_data['CatPrecision'].idxmax()]
    best_text    = k_data[k_data['Paradigm'] == 'Text-Based'].sort_values('CatPrecision', ascending=False).iloc[0]
    best_cnn_row = k_data[k_data['Paradigm'] == 'Visual (CNN)']
    
    print(f"\n  ── K={k} ──────────────────────────────────────────")
    print(f"  Best Overall : {best_overall['Model']} ({best_overall['CatPrecision']:.4f})")
    print(f"  Best Text    : {best_text['Model']} ({best_text['CatPrecision']:.4f})")
    
    if len(best_cnn_row) > 0:
        best_cnn_row = best_cnn_row.sort_values('CatPrecision', ascending=False).iloc[0]
        print(f"  Best CNN     : {best_cnn_row['Model']} ({best_cnn_row['CatPrecision']:.4f})")
        gap = best_cnn_row['CatPrecision'] - best_text['CatPrecision']
        print(f"  Gap CNN-Text : {gap:+.4f}")
    else:
        print("  Best CNN     : N/A (no CNN models evaluated)")

# Summary paragraph
print("\n" + "─" * 70)
print("  📝 RINGKASAN")
print("─" * 70)

k10_data = df_comparison[df_comparison['K'] == 10]
all_text = k10_data[k10_data['Paradigm'] == 'Text-Based'].sort_values('CatPrecision', ascending=False)
all_cnn  = k10_data[k10_data['Paradigm'] == 'Visual (CNN)'].sort_values('CatPrecision', ascending=False)

print("\n  Ranking @K=10 (Category Precision):")
ranked = k10_data.sort_values('CatPrecision', ascending=False)
for i, (_, row) in enumerate(ranked.iterrows()):
    marker = "🟢" if row['Paradigm'] == 'Visual (CNN)' else "🔴"
    print(f"    {i+1}. {marker} {row['Model']:<30s} {row['CatPrecision']:.4f}")

print("""
  Interpretasi:
  - Category Precision mengukur kemampuan ruang vektor mengelompokkan
    item ke KATEGORI yang benar berdasarkan kemiripan vektor.
  - Semakin tinggi → vektor semakin baik dalam menangkap struktur
    kategori, yang krusial untuk recommender system.
  - CNN menggunakan fitur visual (pixel), text menggunakan atribut metadata.
  - Hasil ini menunjukkan SEBERAPA BAIK masing-masing paradigma
    merepresentasikan data sebelum masuk ke recommender system.
""")

print("=" * 70)
print("  ✅ EVALUASI RUANG VEKTOR SELESAI")
print("=" * 70)
