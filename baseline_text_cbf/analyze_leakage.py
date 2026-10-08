"""
Analisis statistik: atribut mana yang benar-benar bocor (leakage)?
Untuk setiap atribut "suspect", cek distribusi kategorinya.
"""
import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTR_DIR = os.path.join(ROOT, 'Anno', 'attributes')

# Load dataset
df = pd.read_csv(os.path.join(ROOT, 'master_dataset.csv'))

# Load attribute names
attr_names = []
with open(os.path.join(ATTR_DIR, 'list_attr_cloth.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        if line.strip():
            attr_names.append(line.strip().split()[0])

# Load attribute labels per item
item_attr_map = {}
with open(os.path.join(ATTR_DIR, 'list_attr_items.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        item_id = parts[0]
        labels = [1 if int(x) == 1 else 0 for x in parts[1:]]
        item_attr_map[item_id] = labels

# Suspect attributes
suspects = ['top', 'dress', 'tee', 'denim', 'shorts', 'graphic', 'skirt',
            'jeans', 'shirt', 'tank', 'sweater', 'cardigan', 'jacket',
            'pants', 'blouse', 'romper', 'leggings', 'sweatshirt', 'jumpsuit',
            'hoodie', 'vest', 'blazer', 'trousers', 'joggers']

header = f"{'Attribute':<15} {'Dominant Category':<25} {'% Dominant':<12} {'#Cats':<7} {'Verdict'}"
print(header)
print("=" * 85)

for suspect in suspects:
    if suspect not in attr_names:
        continue
    idx = attr_names.index(suspect)

    # Find items where this attr is active (=1)
    active_items = []
    for item_id, labels in item_attr_map.items():
        if idx < len(labels) and labels[idx] == 1:
            active_items.append(item_id)

    if not active_items:
        print(f"{suspect:<15} (no active items)")
        continue

    # Get category distribution
    cats = df[df['item_id'].isin(active_items)]['category'].value_counts()
    total = cats.sum()
    dominant_cat = cats.index[0]
    dominant_pct = cats.iloc[0] / total * 100
    n_cats = len(cats)

    if dominant_pct > 90:
        verdict = "JELAS LEAKAGE"
    elif dominant_pct > 70:
        verdict = "STRONG LEAKAGE"
    elif dominant_pct > 50:
        verdict = "PARTIAL"
    else:
        verdict = "AMAN (tersebar)"

    print(f"{suspect:<15} {dominant_cat:<25} {dominant_pct:>6.1f}%      {n_cats:<7} {verdict}")

    # Show top-3 categories for context
    for cat_name, count in cats.head(3).items():
        pct = count / total * 100
        print(f"  {'':>15} -> {cat_name}: {count} ({pct:.1f}%)")
    print()
