import json
import os

def new_code_cell(source):
    # Normalize string into list of lines, preserving newlines except for the last line
    lines = source.split('\n')
    source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_list
    }

def new_markdown_cell(source):
    lines = source.split('\n')
    source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_list
    }

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cells = []

# Cell 1
cells.append(new_markdown_cell("""# Visualisasi Nearest Neighbors — Category-Level Retrieval
CNN vs Text-Based: Bukti Visual Kualitas Ruang Vektor"""))

# Cell 2
cells.append(new_code_cell("""import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI — Auto-detect project root
# ─────────────────────────────────────────────
# Karena Jupyter server bisa dijalankan dari directory manapun,
# kita tidak bisa bergantung pada os.getcwd().
# Strategi: cari project root berdasarkan marker file (master_dataset.csv)

def find_project_root():
    # Kandidat path (WSL mount + Windows)
    candidates = [
        '/mnt/d/Antigravity/Visual Based Rekomender Sistem',
        '/mnt/c/Antigravity/Visual Based Rekomender Sistem',
        os.path.dirname(os.path.abspath('')),  # kadang works di Jupyter
        os.path.dirname(os.getcwd()),
    ]
    for path in candidates:
        marker = os.path.join(path, 'dataset', 'master_dataset.csv')
        if os.path.exists(marker):
            return path
    raise FileNotFoundError(
        "Tidak dapat menemukan project root! "
        "Pastikan path project benar atau edit ROOT_DIR secara manual."
    )

ROOT_DIR   = find_project_root()
print(f"Project root: {ROOT_DIR}")

ANNO_DIR   = os.path.join(ROOT_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Anno')
ATTR_DIR   = os.path.join(ANNO_DIR, 'attributes')
EVAL_DIR   = os.path.join(ROOT_DIR, 'evaluation', 'vector_space_comparison')
IMG_ROOT   = os.path.join(ROOT_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')
NN_VIZ_DIR = os.path.join(EVAL_DIR, 'nn_viz')
os.makedirs(NN_VIZ_DIR, exist_ok=True)

TARGET_CATEGORIES = [
    'Leggings',
    'Cardigans',
    'Shirts_Polos',
    'Denim',
    'Dresses',
    'Rompers_Jumpsuits'
]
"""))

# Cell 3
cells.append(new_code_cell("""# ═══════════════════════════════════════════════
# STEP 1: Load Master Dataset
# ═══════════════════════════════════════════════
MASTER_CSV = os.path.join(ROOT_DIR, 'dataset', 'master_dataset.csv')
df = pd.read_csv(MASTER_CSV)

cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 20].index.tolist()
df = df[df['category'].isin(valid_cats)].reset_index(drop=True)

print(f"Total items (setelah filter >=20/cat): {len(df):,}")
print(f"Kategori valid                      : {df['category'].nunique()}")
"""))

# Cell 4
cells.append(new_code_cell("""# ═══════════════════════════════════════════════
# STEP 2: Build Text Feature Matrices (One-Hot)
# ═══════════════════════════════════════════════
LEAKY_KEYWORDS = {
    'jumpsuit', 'romper', 'sweatshirt', 'shorts',
    'pants', 'blouse', 'hoodie', 'dress',
}

def is_leaky_attribute(attr_name: str) -> bool:
    words = set(attr_name.lower().replace('-', ' ').replace('_', ' ').split())
    return bool(words & LEAKY_KEYWORDS)

# Load attributes
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

# Load colors
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

# One-Hot Encoding
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

# L2 normalize
oh_norms = np.linalg.norm(onehot_matrix, axis=1, keepdims=True)
oh_norms = np.where(oh_norms == 0, 1e-10, oh_norms)
onehot_matrix_norm = onehot_matrix / oh_norms

print(f"One-Hot matrix: {onehot_matrix_norm.shape}")
"""))

# Cell 5
cells.append(new_code_cell("""# ═══════════════════════════════════════════════
# STEP 3: Fungsi Visualisasi Nearest Neighbors
# ═══════════════════════════════════════════════
def visualize_nearest_neighbors(feature_matrix, df_eval, model_name, target_categories, k=5, seed=42):
    np.random.seed(seed)
    
    def get_local_path(image_name):
        # Hapus prefix img/ lalu gabungkan ke IMG_ROOT
        relative_path = image_name.replace('img/', '', 1)
        return os.path.join(IMG_ROOT, relative_path)
    
    for cat in target_categories:
        cat_indices = np.where(df_eval['category'].values == cat)[0]
        if len(cat_indices) == 0:
            print(f"Category {cat} not found in dataset!")
            continue
            
        # Pilih 1 item secara random dari kategori tersebut
        query_idx = np.random.choice(cat_indices)
        query_feat = feature_matrix[query_idx]
        query_row = df_eval.iloc[query_idx]
        
        # Dot product untuk mencari distance / similarity
        sims = feature_matrix @ query_feat
        sims[query_idx] = -np.inf # exclude query diri sendiri
        
        # Ambil top K tetangga terdekat
        top_k_indices = np.argsort(sims)[::-1][:k]
        hits = sum(1 for idx in top_k_indices if df_eval.iloc[idx]['category'] == cat)
        precision = hits / k
        
        # Buat figure subplot 1 baris, K+1 kolom
        fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 5))
        fig.suptitle(f"{model_name} — Query: {cat} | CatPrec@{k} = {hits}/{k} = {precision:.2f}", 
                     fontsize=14, fontweight='bold', y=1.05)
        
        # --- 1. Plot Query Image ---
        q_path = get_local_path(query_row['image_name'])
        try:
            img = Image.open(q_path)
            axes[0].imshow(img)
            # Border biru untuk query
            rect = patches.Rectangle((0, 0), img.width-1, img.height-1, linewidth=6, edgecolor='#3498db', facecolor='none')
            axes[0].add_patch(rect)
        except Exception as e:
            axes[0].text(0.5, 0.5, 'Image Error', ha='center')
            
        axes[0].set_title(f"QUERY\n{cat}", fontweight='bold', color='#3498db')
        axes[0].axis('off')
        
        # --- 2. Plot Top-K Neighbors ---
        for i, idx in enumerate(top_k_indices):
            ax = axes[i + 1]
            n_row = df_eval.iloc[idx]
            n_cat = n_row['category']
            n_sim = sims[idx]
            n_path = get_local_path(n_row['image_name'])
            
            is_match = (n_cat == cat)
            color = '#2ecc71' if is_match else '#e74c3c'
            mark = '✅' if is_match else '❌'
            
            try:
                img = Image.open(n_path)
                ax.imshow(img)
                # Border hijau jika match, merah jika mismatch
                rect = patches.Rectangle((0, 0), img.width-1, img.height-1, linewidth=6, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            except Exception as e:
                ax.text(0.5, 0.5, 'Image Error', ha='center')
                
            ax.set_title(f"Top-{i+1} (sim={n_sim:.2f})\n{n_cat} {mark}", fontweight='bold', color=color)
            ax.axis('off')
            
        plt.tight_layout()
        
        # Save ke output folder
        safe_model = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
        filename = f"nn_{safe_model}_{cat}_k{k}.png"
        filepath = os.path.join(NN_VIZ_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show() # Tampilkan inline
"""))

# Cell 6
cells.append(new_markdown_cell("""## Kategori Target
- 🔴 **Low Precision**: Leggings, Cardigans, Shirts_Polos, Denim
- 🟢 **High Precision**: Dresses, Rompers_Jumpsuits"""))

# Cell 7
cells.append(new_code_cell("""# ═══════════════════════════════════════════════
# Visualisasi 1: One-Hot (Text-Based)
# ═══════════════════════════════════════════════
print("Menampilkan visualisasi untuk model One-Hot (Text-Based)...")
visualize_nearest_neighbors(onehot_matrix_norm, df, "One-Hot", TARGET_CATEGORIES, k=5, seed=42)
"""))

# Cell 8
cells.append(new_code_cell("""# ═══════════════════════════════════════════════
# Visualisasi 2: VGG19 (Exp3) — Best CNN
# ═══════════════════════════════════════════════
print("Menampilkan visualisasi untuk model VGG19 (Exp3)...")

df_raw = pd.read_csv(MASTER_CSV)

vgg_feat_path = os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'vgg19_features_exp3.npy')
vgg_idx_path  = os.path.join(ROOT_DIR, 'features', 'exp3_partial_unfreeze', 'vgg19_valid_idx_exp3.npy')

if os.path.exists(vgg_feat_path):
    feat_mat = np.load(vgg_feat_path)
    valid_idx = np.load(vgg_idx_path).tolist()
    
    df_sub = df_raw.iloc[valid_idx].reset_index(drop=True)
    
    cat_counts_sub = df_sub['category'].value_counts()
    valid_cats_sub = cat_counts_sub[cat_counts_sub >= 20].index.tolist()
    valid_mask     = df_sub['category'].isin(valid_cats_sub)
    
    df_vgg   = df_sub[valid_mask].reset_index(drop=True)
    feat_vgg = feat_mat[valid_mask]
    
    norms = np.linalg.norm(feat_vgg, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    feat_vgg = feat_vgg / norms
    
    visualize_nearest_neighbors(feat_vgg, df_vgg, "VGG19 (Exp3)", TARGET_CATEGORIES, k=5, seed=42)
else:
    print("Feature VGG19 tidak ditemukan di path:", vgg_feat_path)
"""))

# Cell 9
cells.append(new_code_cell("""# ═══════════════════════════════════════════════
# Visualisasi 3: ResNet50 (GridSearch) — Runner-up CNN
# ═══════════════════════════════════════════════
print("Menampilkan visualisasi untuk model ResNet50 (GridSearch)...")

resnet_feat_path = os.path.join(ROOT_DIR, 'grid_search', 'features', 'resnet50_adam_bs24_lr1e-05_uf50', 'master_features.npy')
resnet_idx_path  = os.path.join(ROOT_DIR, 'grid_search', 'features', 'resnet50_adam_bs24_lr1e-05_uf50', 'master_valid_idx.npy')

if os.path.exists(resnet_feat_path):
    feat_mat = np.load(resnet_feat_path)
    valid_idx = np.load(resnet_idx_path).tolist()
    
    df_sub = df_raw.iloc[valid_idx].reset_index(drop=True)
    
    cat_counts_sub = df_sub['category'].value_counts()
    valid_cats_sub = cat_counts_sub[cat_counts_sub >= 20].index.tolist()
    valid_mask     = df_sub['category'].isin(valid_cats_sub)
    
    df_rn   = df_sub[valid_mask].reset_index(drop=True)
    feat_rn = feat_mat[valid_mask]
    
    norms = np.linalg.norm(feat_rn, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    feat_rn = feat_rn / norms
    
    visualize_nearest_neighbors(feat_rn, df_rn, "ResNet50 (GridSearch)", TARGET_CATEGORIES, k=5, seed=42)
else:
    print("Feature ResNet50 tidak ditemukan di path:", resnet_feat_path)
"""))

# Cell 10
cells.append(new_markdown_cell("""## Kesimpulan Visual
- **Kategori Leggings**: CNN sering keliru ke kategori `Pants` atau `Denim` yang mana sangat wajar karena secara bentuk visual mereka sangat mirip (hanya berbeda ketebalan bahan atau tekstur, yang mana sulit dibedakan oleh CNN di resolusi gambar e-commerce).
- **Kategori Cardigans**: Sering keliru diprediksi dekat dengan `Sweaters` atau `Jackets`, di mana ada tingkat overlap visual yang cukup ekstrem dalam dataset pakaian ini.
- **Kategori Dresses**: Model CNN sangat luar biasa akurat karena bentuk siluet "full body" dari Dresses mudah dibedakan secara geometris dari pakaian atasan/bawahan biasa.
- **Text (One-Hot)**: Bisa terlihat bahwa tetangga terdekat dari Text sangat random dari sisi visual. Jika item beda kategori memiliki overlap tag warna atau keywords di metadata yang sama, maka One-Hot matrix akan menganggap mereka identik secara kesamaan kosinus."""))

# Cell 11
cells.append(new_code_cell("""print("✅ Proses visualisasi selesai!")
print(f"Semua gambar PNG telah disimpan di: {NN_VIZ_DIR}")
"""))

notebook["cells"] = cells

with open('08_category_nn_visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Berhasil membuat file: 08_category_nn_visualization.ipynb")
