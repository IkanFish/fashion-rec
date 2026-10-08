import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ANNO_DIR = os.path.join(ROOT_DIR, 'Anno')
ATTR_DIR = os.path.join(ANNO_DIR, 'attributes')
FEAT_DIR = os.path.join(ROOT_DIR, 'features')

os.makedirs(FEAT_DIR, exist_ok=True)

# 1. Leaky Keywords (Evidence-Based from V3 logic)
LEAKY_KEYWORDS = {
    'jumpsuit', 'romper', 'sweatshirt', 'shorts',
    'pants', 'blouse', 'hoodie', 'dress',
}

def is_leaky_attribute(attr_name: str) -> bool:
    words = set(attr_name.lower().replace('-', ' ').replace('_', ' ').split())
    return bool(words & LEAKY_KEYWORDS)

# 2. Load Master Dataset
df = pd.read_csv(os.path.join(ROOT_DIR, 'master_dataset.csv'))
cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 20].index.tolist()
df = df[df['category'].isin(valid_cats)].reset_index(drop=True)

# 3. Load Attributes and Filter Leaky ones
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

# 4. Load Item Attributes
item_attrs = {}
with open(os.path.join(ATTR_DIR, 'list_attr_items.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2: continue
        item_id = parts[0]
        labels = [int(x) for x in parts[1:]]
        
        active = []
        for i, val in enumerate(labels):
            if val == 1 and i not in leaky_indices:
                active.append(all_attr_names[i].split()[0])
        item_attrs[item_id] = active

# 5. Load Item Colors
item_colors = {}
with open(os.path.join(ATTR_DIR, 'list_color_cloth.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2: continue
        match = re.search(r'(id_\d+)', parts[0])
        if match:
            item_id = match.group(1)
            if item_id not in item_colors:
                item_colors[item_id] = parts[1]

# 6. Build Text Profiles
print("Building text profiles...")
text_profiles = []
for idx, row in df.iterrows():
    item_id = row['item_id']
    attrs = item_attrs.get(item_id, [])
    color = item_colors.get(item_id, '')
    
    parts = list(attrs)
    if color:
        parts.extend(color.replace('-', ' ').replace('_', ' ').split())
        
    text_profiles.append(' '.join(parts) if parts else 'unknown')

df['text_profile'] = text_profiles

# 7. Export Text DataFrame
csv_export_path = os.path.join(FEAT_DIR, 'text_profiles.csv')
df[['item_id', 'category', 'text_profile']].to_csv(csv_export_path, index=False)
print(f"Exported text descriptions to: {csv_export_path}")

# 8. TF-IDF Vectorization
print("Running TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(text_profiles)

# Convert from sparse matrix to standard dense numpy array
dense_features = tfidf_matrix.toarray()

# 9. Export .npy feature matrix
npy_export_path = os.path.join(FEAT_DIR, 'tfidf_features_exp5.npy')
np.save(npy_export_path, dense_features)
print(f"Exported TF-IDF feature matrix ({dense_features.shape}) to: {npy_export_path}")

print("Done! Ready for Streamlit integration.")
