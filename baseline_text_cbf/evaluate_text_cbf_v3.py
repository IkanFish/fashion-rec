"""
=============================================================
  BASELINE: TEXT-BASED CBF v3 — EVIDENCE-BASED FILTERING
  TF-IDF + Cosine Similarity on DeepFashion Metadata

  PERBEDAAN DENGAN VERSI SEBELUMNYA:
  - v1: Tanpa filter (42 atribut bocor → P@5 inflated)
  - v2: Filter terlalu agresif (42 atribut → P@5 deflated)
  - v3: Filter hanya 10 atribut yang terbukti >70% terkonsentrasi
        di 1 kategori (evidence-based, via analyze_leakage.py)
=============================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
ANNO_DIR  = os.path.join(ROOT_DIR, 'Anno')
ATTR_DIR  = os.path.join(ANNO_DIR, 'attributes')
EVAL_DIR  = os.path.join(BASE_DIR, 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

K_VALUES  = [5, 10, 20]
N_USERS   = 100
N_LIKED   = 3

# ─────────────────────────────────────────────
# EVIDENCE-BASED LEAKY KEYWORDS
# Hanya 8 kata kunci yang >70% terkonsentrasi di 1 kategori
# (dibuktikan oleh analyze_leakage.py)
#
# jumpsuit   → 100.0% Rompers_Jumpsuits
# romper     → 97.3%  Rompers_Jumpsuits
# sweatshirt → 79.1%  Sweatshirts_Hoodies
# shorts     → 78.0%  Shorts
# pants      → 76.9%  Pants
# blouse     → 76.3%  Blouses_Shirts
# hoodie     → 74.8%  Sweatshirts_Hoodies
# dress      → 70.7%  Dresses
# ─────────────────────────────────────────────
LEAKY_KEYWORDS = {
    'jumpsuit', 'romper', 'sweatshirt', 'shorts',
    'pants', 'blouse', 'hoodie', 'dress',
}
# Atribut yang terkena: dress, shorts, pants, blouse, romper,
#   sweatshirt, jumpsuit, hoodie, maxi dress, sleeveless dress
# Total: 10 atribut difilter dari 463


def is_leaky_attribute(attr_name: str) -> bool:
    words = set(attr_name.lower().replace('-', ' ').replace('_', ' ').split())
    return bool(words & LEAKY_KEYWORDS)


# ─────────────────────────────────────────────
# STEP 1: Load Master Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Load Master Dataset")
print("=" * 60)

df = pd.read_csv(os.path.join(ROOT_DIR, 'master_dataset.csv'))
print(f"  Total items: {len(df):,}")
print(f"  Categories : {df['category'].nunique()}")

cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 20].index.tolist()
df = df[df['category'].isin(valid_cats)].reset_index(drop=True)
print(f"  Items after filtering (>=20 per category): {len(df):,}")


# ─────────────────────────────────────────────
# STEP 2: Load & Filter Attributes
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2: Load Attributes (Evidence-Based Filter)")
print("=" * 60)

all_attr_names = []
with open(os.path.join(ATTR_DIR, 'list_attr_cloth.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        if line.strip():
            all_attr_names.append(line.strip())

leaky_indices = set()
leaky_list = []
safe_list = []

for i, attr in enumerate(all_attr_names):
    if is_leaky_attribute(attr):
        leaky_indices.add(i)
        leaky_list.append(attr)
    else:
        safe_list.append(attr)

print(f"  Total attributes: {len(all_attr_names)}")
print(f"  Filtered (leaky): {len(leaky_list)}")
print(f"  Kept (safe): {len(safe_list)}")
print(f"\n  Filtered attributes:")
for attr in leaky_list:
    print(f"    x {attr}")


# ─────────────────────────────────────────────
# STEP 3: Load Attribute Labels per Item
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: Load Attribute Labels (Filtered)")
print("=" * 60)

item_attrs = {}
with open(os.path.join(ATTR_DIR, 'list_attr_items.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        item_id = parts[0]
        labels = [int(x) for x in parts[1:]]

        active = []
        for i, val in enumerate(labels):
            if val == 1 and i not in leaky_indices:
                active.append(all_attr_names[i].split()[0])
        item_attrs[item_id] = active

print(f"  Loaded attributes for {len(item_attrs)} items")
counts = [len(v) for v in item_attrs.values()]
print(f"  Avg per item: {np.mean(counts):.1f}, Min: {min(counts)}, Max: {max(counts)}")


# ─────────────────────────────────────────────
# STEP 4: Load Color Data
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: Load Color Information")
print("=" * 60)

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

print(f"  Loaded colors for {len(item_colors)} items")


# ─────────────────────────────────────────────
# STEP 5: Build Text Profiles
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5: Build Text Profiles")
print("=" * 60)

text_profiles = []
for idx, row in df.iterrows():
    item_id = row['item_id']
    attrs = item_attrs.get(item_id, [])
    color = item_colors.get(item_id, '')

    parts = list(attrs)
    if color:
        parts.extend(color.replace('-', ' ').replace('_', ' ').split())

    text_profiles.append(' '.join(parts) if parts else 'unknown')

print(f"  Built {len(text_profiles)} profiles")
print(f"\n  Samples:")
for i in range(min(3, len(df))):
    print(f"    [{df.iloc[i]['item_id']}] ({df.iloc[i]['category']}): {text_profiles[i][:80]}...")


# ─────────────────────────────────────────────
# STEP 6: TF-IDF Vectorization
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6: TF-IDF Vectorization")
print("=" * 60)

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(text_profiles)

print(f"  Shape: {tfidf_matrix.shape}")
print(f"  Vocab size: {len(vectorizer.vocabulary_)}")

leaked = set(vectorizer.vocabulary_.keys()) & LEAKY_KEYWORDS
if leaked:
    print(f"  WARNING: leaky words in vocab: {leaked}")
else:
    print(f"  VERIFIED: No leaky keywords in vocabulary")


# ─────────────────────────────────────────────
# STEP 7: Evaluation Metrics (identical to CNN)
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

def diversity(indices, mat):
    if len(indices) < 2: return 0.0
    sim = cosine_similarity(mat[indices])
    n = len(indices)
    return sum(1-sim[i,j] for i in range(n) for j in range(i+1,n)) / (n*(n-1)/2)


# ─────────────────────────────────────────────
# STEP 8: Simulate & Evaluate
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8: Simulating 100 Users (seed=42)")
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

    profile = np.mean(tfidf_matrix[liked].toarray(), axis=0).reshape(1, -1)
    scores = cosine_similarity(profile, tfidf_matrix).flatten()
    scores[liked] = -1.0
    ranked = [i for i in np.argsort(scores)[::-1] if i not in set(liked)][:max(K_VALUES)]

    for k in K_VALUES:
        results[k]['precision'].append(precision_at_k(ranked, relevant, k))
        results[k]['recall'].append(recall_at_k(ranked, relevant, k))
        results[k]['f1'].append(f1_at_k(ranked, relevant, k))
        results[k]['ndcg'].append(ndcg_at_k(ranked, relevant, k))
        results[k]['diversity'].append(diversity(ranked[:k], tfidf_matrix))

summary = {k: {m: np.mean(v) for m, v in results[k].items()} for k in K_VALUES}


# ─────────────────────────────────────────────
# STEP 9: Results
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  HASIL: TEXT-BASED CBF v3 (Evidence-Based Filter)")
print("=" * 60)

for k, m in summary.items():
    print(f"  K={k:2d}: P={m['precision']:.4f} | R={m['recall']:.4f} | "
          f"F1={m['f1']:.4f} | NDCG={m['ndcg']:.4f} | Div={m['diversity']:.4f}")

rows = [{'Model': 'TF-IDF v3 (Evidence)', 'K': k,
         'Precision': round(m['precision'],4), 'Recall': round(m['recall'],4),
         'F1': round(m['f1'],4), 'NDCG': round(m['ndcg'],4),
         'Diversity': round(m['diversity'],4)} for k, m in summary.items()]

df_eval = pd.DataFrame(rows)
csv_path = os.path.join(EVAL_DIR, 'text_cbf_v3_evidence_results.csv')
df_eval.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")


# ─────────────────────────────────────────────
# STEP 10: Comparison
# ─────────────────────────────────────────────
v3_p5 = summary[5]['precision']

print("\n" + "=" * 60)
print("  PERBANDINGAN SEMUA METODE (P@5)")
print("=" * 60)
print(f"\n  {'Method':<40} {'P@5'}")
print(f"  {'─'*50}")
print(f"  {'TF-IDF v1 (no filter)':<40} 0.6940")
print(f"  {'TF-IDF v2 (over-filter, 42 attrs)':<40} 0.5620")
print(f"  {'TF-IDF v3 (evidence, 10 attrs)':<40} {v3_p5:.4f}")
print(f"  {'─'*50}")
print(f"  {'MobileNetV3 (CNN Exp3)':<40} 0.5840")
print(f"  {'InceptionV3 (CNN Exp3)':<40} 0.6020")
print(f"  {'ResNet50 (CNN Exp3)':<40} 0.7240")
print(f"  {'VGG19 (CNN Exp3)':<40} 0.7300")

print("\n" + "=" * 60)
print("  SELESAI")
print("=" * 60)
