import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
EVAL_DIR  = os.path.join(BASE_DIR, 'evaluation')

# ─────────────────────────────────────────────
# STEP 1: Load Text Profiles (One-Hot)
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Load Text Profiles")
print("=" * 60)
csv_path = os.path.join(EVAL_DIR, 'list_attr_all.csv')
df_text = pd.read_csv(csv_path)

df_text['features_list'] = df_text['active_features'].apply(
    lambda x: [f.strip() for f in str(x).split(',')] if pd.notnull(x) and str(x).strip() else []
)

mlb = MultiLabelBinarizer(sparse_output=False) # Dense matrix for fast mapping
onehot_matrix = mlb.fit_transform(df_text['features_list'])

# L2 Normalize the text vectors once so we can just use dot product or cosine easily
norms = np.linalg.norm(onehot_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
onehot_matrix = onehot_matrix / norms

item_to_idx = {item: idx for idx, item in enumerate(df_text['item_id'])}

# ─────────────────────────────────────────────
# STEP 2: Load Query & Gallery Split
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2: Load Query & Gallery Images")
print("=" * 60)
full_csv = os.path.join(ROOT_DIR, 'dataset', 'full_dataset.csv')
df_full = pd.read_csv(full_csv)

df_query = df_full[df_full['split'] == 'query'].reset_index(drop=True)
df_gallery = df_full[df_full['split'] == 'gallery'].reset_index(drop=True)

def build_features(df):
    feats = []
    valid_ids = []
    for item_id in df['item_id']:
        if item_id in item_to_idx:
            feats.append(onehot_matrix[item_to_idx[item_id]])
            valid_ids.append(item_id)
    return np.array(feats), np.array(valid_ids)

q_feats, q_ids = build_features(df_query)
g_feats, g_ids = build_features(df_gallery)

print(f"  Query images mapped  : {len(q_feats):,}")
print(f"  Gallery images mapped: {len(g_feats):,}")

# ─────────────────────────────────────────────
# STEP 3: Hit/Miss Retrieval Evaluation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: Evaluate Hit/Miss (Query vs Gallery)")
print("=" * 60)

k_values = [1, 5, 10, 20]
recall_at_k = {k: 0 for k in k_values}
n_queries = len(q_feats)

# TRICK PENTING:
# Karena banyak baju memiliki teks yang 100% sama (Tie / Seri),
# kita tambahkan noise/acak yang SANGAT KECIL (1e-6) ke Gallery.
# Ini agar np.argsort() memecah nilai seri secara adil dan acak.
np.random.seed(42)
noise = np.random.uniform(0, 1e-6, size=g_feats.shape)
g_feats_noisy = g_feats + noise

batch_size = 1000
for start in tqdm(range(0, n_queries, batch_size), desc="Evaluating Retrieval"):
    end = min(start + batch_size, n_queries)
    batch_q = q_feats[start:end]
    batch_q_ids = q_ids[start:end]
    
    # Hitung kemiripan
    sim = cosine_similarity(batch_q, g_feats_noisy)
    
    for i in range(len(batch_q)):
        query_id = batch_q_ids[i]
        
        # Cek apakah item_id ini memang ada pasangannya di gallery
        if (g_ids == query_id).sum() == 0:
            continue
            
        sorted_indices = np.argsort(-sim[i])
        
        for k in k_values:
            top_k_ids = g_ids[sorted_indices[:k]]
            if query_id in top_k_ids:
                recall_at_k[k] += 1

print("\n" + "=" * 60)
print("  HASIL HIT RATE (RECALL@K) TEXT-BASED ONE-HOT")
print("=" * 60)
for k in k_values:
    print(f"  Recall@{k:2d}: {recall_at_k[k] / n_queries:.4f}  ({(recall_at_k[k] / n_queries)*100:.2f}%)")

print("\n(Catatan: Bandingkan ini dengan CNN yang memiliki Recall@5 di atas 70%!)")
print("=" * 60)
