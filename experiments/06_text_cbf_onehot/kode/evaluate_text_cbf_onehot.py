"""
=============================================================
  BASELINE: TEXT-BASED CBF — ONE-HOT ENCODING (CLEAN)
  Binary Attribute Encoding + Cosine Similarity

  TUJUAN:
  Mengevaluasi One-Hot Encoding sebagai baseline text-based.
  Data input (list_attr_all.csv) sudah dibersihkan dari atribut
  leaky menggunakan statistical dominance filter (>70%).
  Lihat: build_clean_onehot.ipynb untuk proses pembersihan.
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR  = os.path.join(BASE_DIR, 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

K_VALUES  = [5, 10, 20]
N_USERS   = 100
N_LIKED   = 3

# ─────────────────────────────────────────────
# STEP 1: Load Pre-processed Dataset (list_attr_all.csv)
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Load Pre-processed Dataset")
print("=" * 60)

csv_path = os.path.join(EVAL_DIR, 'list_attr_all.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}. Please run export_onehot_profiles.py first.")

df = pd.read_csv(csv_path)
print(f"  Total items: {len(df):,}")
print(f"  Categories : {df['category'].nunique()}")

# ─────────────────────────────────────────────
# STEP 2: Build One-Hot Encoded Feature Matrix
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2: Build One-Hot Matrix via MultiLabelBinarizer")
print("=" * 60)

# Convert comma-separated string back to a list of features
df['features_list'] = df['active_features'].apply(
    lambda x: [f.strip() for f in str(x).split(',')] if pd.notnull(x) and str(x).strip() else []
)

mlb = MultiLabelBinarizer(sparse_output=True)
onehot_matrix = mlb.fit_transform(df['features_list'])

n_items, n_features = onehot_matrix.shape
print(f"  Matrix shape: {onehot_matrix.shape}")
print(f"  Non-zero entries: {onehot_matrix.nnz:,}")
print(f"  Sparsity: {1 - onehot_matrix.nnz / (n_items * n_features):.4%}")

row_sums = np.array(onehot_matrix.sum(axis=1)).flatten()
print(f"\n  Avg active features per item: {row_sums.mean():.1f}")
print(f"  Min: {row_sums.min():.0f}, Max: {row_sums.max():.0f}")
print(f"  Items with 0 features: {(row_sums == 0).sum()}")

# ─────────────────────────────────────────────
# STEP 3: Evaluasi — Metrik
# ─────────────────────────────────────────────
def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / k

def recall_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / len(rel) if rel else 0.0

def f1_at_k(rec, rel, k):
    p, r = precision_at_k(rec, rel, k), recall_at_k(rec, rel, k)
    return 2*p*r/(p+r) if (p+r) else 0.0

def dcg_at_k(rec, rel, k):
    return sum(1/np.log2(i+2) for i, item in enumerate(rec[:k]) if item in rel)

def ndcg_at_k(rec, rel, k):
    actual = dcg_at_k(rec, rel, k)
    ideal = dcg_at_k(list(rel)[:k], rel, k)
    return actual/ideal if ideal else 0.0

def diversity_score(indices, mat):
    if len(indices) < 2: return 0.0
    sim = cosine_similarity(mat[indices])
    n = len(indices)
    return sum(1-sim[i,j] for i in range(n) for j in range(i+1,n)) / (n*(n-1)/2)

# ─────────────────────────────────────────────
# STEP 4: Simulate & Evaluate
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: Simulating 100 Users (seed=42)")
print("=" * 60)

categories = df['category'].unique()
results = {k: {'precision':[], 'recall':[], 'f1':[], 'ndcg':[], 'diversity':[]} for k in K_VALUES}
rng = np.random.default_rng(42)

for _ in tqdm(range(N_USERS), desc="Simulating users"):
    cat = rng.choice(categories)
    cat_idx = df[df['category'] == cat].index.tolist()
    if len(cat_idx) < N_LIKED + 5:
        continue

    liked_pos = rng.choice(len(cat_idx), size=N_LIKED, replace=False)
    liked = [cat_idx[p] for p in liked_pos]
    relevant = [i for i in cat_idx if i not in set(liked)]

    profile = np.mean(onehot_matrix[liked].toarray(), axis=0).reshape(1, -1)

    scores = cosine_similarity(profile, onehot_matrix).flatten()
    scores[liked] = -1.0
    ranked = [i for i in np.argsort(scores)[::-1] if i not in set(liked)][:max(K_VALUES)]

    for k in K_VALUES:
        results[k]['precision'].append(precision_at_k(ranked, relevant, k))
        results[k]['recall'].append(recall_at_k(ranked, relevant, k))
        results[k]['f1'].append(f1_at_k(ranked, relevant, k))
        results[k]['ndcg'].append(ndcg_at_k(ranked, relevant, k))
        results[k]['diversity'].append(diversity_score(ranked[:k], onehot_matrix))

summary = {k: {m: np.mean(v) for m, v in results[k].items()} for k in K_VALUES}

# ─────────────────────────────────────────────
# STEP 5: Results
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  HASIL: TEXT-BASED CBF — ONE-HOT ENCODING")
print("=" * 60)

for k, m in summary.items():
    print(f"  K={k:2d}: P={m['precision']:.4f} | R={m['recall']:.4f} | "
          f"F1={m['f1']:.4f} | NDCG={m['ndcg']:.4f} | Div={m['diversity']:.4f}")

rows = [{'Model': 'One-Hot Encoding', 'K': k,
         'Precision': round(m['precision'],4), 'Recall': round(m['recall'],4),
         'F1': round(m['f1'],4), 'NDCG': round(m['ndcg'],4),
         'Diversity': round(m['diversity'],4)} for k, m in summary.items()]

df_eval = pd.DataFrame(rows)
csv_path = os.path.join(EVAL_DIR, 'text_cbf_onehot_results.csv')
df_eval.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")

# ─────────────────────────────────────────────
# STEP 6: Perbandingan dengan CNN (P@5 Reference)
# ─────────────────────────────────────────────
oh_p5  = summary[5]['precision']
oh_p10 = summary[10]['precision']
oh_p20 = summary[20]['precision']

print("\n" + "-" * 70)
print("  PERBANDINGAN: TEXT-BASED (One-Hot) vs CNN (Exp3 Partial Unfreeze)")
print("-" * 70)
print(f"\n  {'Method':<45} {'P@5':>8} {'P@10':>8} {'P@20':>8}")
print(f"  {'-'*70}")
print(f"  {'One-Hot Encoding (statistical filter)':<45} {oh_p5:>8.4f} {oh_p10:>8.4f} {oh_p20:>8.4f}")
print(f"  {'-'*70}")
print(f"  {'MobileNetV3  (CNN Exp3)':<45} {0.5840:>8.4f} {0.6200:>8.4f} {0.6600:>8.4f}")
print(f"  {'InceptionV3  (CNN Exp3)':<45} {0.6020:>8.4f} {0.6340:>8.4f} {0.6740:>8.4f}")
print(f"  {'ResNet50     (CNN Exp3)':<45} {0.7240:>8.4f} {0.7340:>8.4f} {0.7560:>8.4f}")
print(f"  {'VGG19        (CNN Exp3)':<45} {0.7300:>8.4f} {0.7480:>8.4f} {0.7700:>8.4f}")
print(f"\n  Catatan: One-Hot adalah baseline non-visual. CNN unggul karena")
print(f"  menangkap fitur visual yang tidak tersedia di metadata teks.")
print("\n" + "-" * 70)
print("  SELESAI")
print("-" * 70)
