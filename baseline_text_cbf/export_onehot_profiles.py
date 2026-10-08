import os
import re
import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
ANNO_DIR    = os.path.join(DATASET_DIR, 'In-shop Clothes Retrieval Benchmark', 'Anno')
ATTR_DIR    = os.path.join(ANNO_DIR, 'attributes')
OUT_DIR     = os.path.join(BASE_DIR, 'evaluation')

# 1. Leaky keywords
LEAKY_KEYWORDS = {'jumpsuit', 'romper', 'sweatshirt', 'shorts', 'pants', 'blouse', 'hoodie', 'dress'}
def is_leaky_attribute(attr_name: str) -> bool:
    words = set(attr_name.lower().replace('-', ' ').replace('_', ' ').split())
    return bool(words & LEAKY_KEYWORDS)

# 2. Load Master Dataset
df = pd.read_csv(os.path.join(DATASET_DIR, 'master_dataset.csv'))
cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 20].index.tolist()
df = df[df['category'].isin(valid_cats)].reset_index(drop=True)

# 3. Load Attributes
all_attr_names = []
with open(os.path.join(ATTR_DIR, 'list_attr_cloth.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        if line.strip():
            all_attr_names.append(line.strip())

safe_attr_indices = []
for i, attr in enumerate(all_attr_names):
    if not is_leaky_attribute(attr):
        safe_attr_indices.append((i, attr.split()[0])) # Ambil kata pertama saja untuk simplisitas

# 4. Load Item Attributes
item_attrs_binary = {}
with open(os.path.join(ATTR_DIR, 'list_attr_items.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            item_id = parts[0]
            labels = [int(x) for x in parts[1:]]
            active_safe = []
            for orig_idx, attr_name in safe_attr_indices:
                if orig_idx < len(labels) and labels[orig_idx] == 1:
                    active_safe.append(attr_name)
            item_attrs_binary[item_id] = active_safe

# 5. Load Colors
item_colors = {}
with open(os.path.join(ATTR_DIR, 'list_color_cloth.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            match = re.search(r'(id_\d+)', parts[0])
            if match:
                item_id = match.group(1)
                color_raw = parts[1]
                color_tokens = color_raw.replace('-', ' ').replace('_', ' ').lower().split()
                item_colors[item_id] = color_tokens

# 6. Build Text Profiles
print("Mengekstrak profil untuk diekspor...")
export_data = []
for idx, row in df.iterrows():
    item_id = row['item_id']
    cat = row['category']
    
    attrs = item_attrs_binary.get(item_id, [])
    colors = item_colors.get(item_id, [])
    
    # Gabungkan atribut dan warna (dengan prefix color:)
    profile_words = attrs + [f"color:{c}" for c in colors]
    
    export_data.append({
        'item_id': item_id,
        'category': cat,
        'active_features_count': len(profile_words),
        'active_features': ', '.join(profile_words)
    })

df_export = pd.DataFrame(export_data)
out_path = os.path.join(OUT_DIR, 'onehot_active_profiles.csv')
df_export.to_csv(out_path, index=False)
print(f"Selesai! File diekspor ke: {out_path}")
