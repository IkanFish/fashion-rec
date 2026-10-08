"""
=============================================================
  VISUAL-BASED FASHION RECOMMENDER SYSTEM
  Notebook 01 — Data Preparation
  Dataset: DeepFashion — In-shop Clothes Retrieval Benchmark
  Jalankan di Google Colab (CPU runtime cukup)
=============================================================
Tujuan:
  - Mount Google Drive & setup path
  - Parse list_bbox_inshop.txt     → image paths + kategori
  - Parse list_eval_partition.txt  → train/query/gallery split
  - Parse list_item_inshop.txt     → 7.982 item unik
  - Buat master_dataset.csv        → dataset siap pakai
  - Validasi gambar & visualisasi distribusi
=============================================================
Struktur path gambar:
  img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg
      └──gender──┘ └──category────────┘ └─item─┘ └──variant──┘
=============================================================
"""

# ─────────────────────────────────────────────
# CELL 1: Mount Google Drive
# ─────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')


# ─────────────────────────────────────────────
# CELL 2: Install & Import Dependencies
# ─────────────────────────────────────────────
# !pip install -q Pillow tqdm pandas matplotlib seaborn

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✅ Dependencies loaded.")


# ─────────────────────────────────────────────
# CELL 3: Konfigurasi Path
# Sesuaikan BASE_DIR dengan lokasi folder di Google Drive kamu
# ─────────────────────────────────────────────
BASE_DIR    = '/content/drive/MyDrive/fashion_recommender'

# Lokasi dataset In-shop Clothes Retrieval Benchmark
DATASET_DIR = os.path.join(BASE_DIR, 'In-shop Clothes Retrieval Benchmark')
IMG_DIR     = os.path.join(DATASET_DIR, 'img/img')       # folder gambar (hasil ekstrak img.zip → nested di img/img)
ANNO_DIR    = os.path.join(DATASET_DIR, 'Anno')
EVAL_DIR    = os.path.join(DATASET_DIR, 'Eval')

# File anotasi
BBOX_FILE   = os.path.join(ANNO_DIR, 'list_bbox_inshop.txt')
ITEM_FILE   = os.path.join(ANNO_DIR, 'list_item_inshop.txt')
EVAL_FILE   = os.path.join(EVAL_DIR, 'list_eval_partition.txt')

# Output
OUTPUT_DIR  = os.path.join(BASE_DIR, 'processed')
MASTER_CSV  = os.path.join(OUTPUT_DIR, 'master_dataset.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verifikasi file ada
for name, path in [('IMG_DIR', IMG_DIR), ('BBOX_FILE', BBOX_FILE),
                   ('ITEM_FILE', ITEM_FILE), ('EVAL_FILE', EVAL_FILE)]:
    status = '✅' if os.path.exists(path) else '❌ TIDAK DITEMUKAN'
    print(f"  {status} {name}: {path}")

print(f"\n📁 Output: {OUTPUT_DIR}")


# ─────────────────────────────────────────────
# CELL 4: Parse list_bbox_inshop.txt
# Kolom: image_name  clothes_type  pose_type  x1 y1 x2 y2
# clothes_type: 1=upper-body, 2=lower-body, 3=full-body
# pose_type:    1=front, 2=side, 3=back, 4=full, 5=zoom-in, 6=flat
# ─────────────────────────────────────────────
def parse_bbox_file(filepath):
    """
    Parse list_bbox_inshop.txt
    Return: DataFrame dengan kolom:
      image_name, clothes_type, clothes_type_label, pose_type, pose_type_label,
      x1, y1, x2, y2
    """
    CLOTHES_TYPE = {1: 'upper-body', 2: 'lower-body', 3: 'full-body'}
    POSE_TYPE    = {1: 'front', 2: 'side', 3: 'back', 4: 'full', 5: 'zoom-in', 6: 'flat'}

    records = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    n_total = int(lines[0].strip())   # baris 1: jumlah gambar
    # baris 2: header → skip
    for line in tqdm(lines[2:n_total + 2], desc="Parsing bbox annotations"):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        img_name     = parts[0]                    # e.g. img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg
        clothes_type = int(parts[1])
        pose_type    = int(parts[2])
        x1, y1, x2, y2 = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])

        # Extract info dari path
        # Format: img/{gender}/{category}/{item_id}/{variant}.jpg
        path_parts  = img_name.replace('\\', '/').split('/')
        gender      = path_parts[1] if len(path_parts) > 1 else 'Unknown'
        category    = path_parts[2] if len(path_parts) > 2 else 'Unknown'
        item_id     = path_parts[3] if len(path_parts) > 3 else 'Unknown'

        records.append({
            'image_name'        : img_name,
            'full_path'         : os.path.join(IMG_DIR, img_name.replace('img/', '', 1)),
            'gender'            : gender,
            'category'          : category,
            'item_id'           : item_id,
            'clothes_type'      : clothes_type,
            'clothes_type_label': CLOTHES_TYPE.get(clothes_type, 'unknown'),
            'pose_type'         : pose_type,
            'pose_type_label'   : POSE_TYPE.get(pose_type, 'unknown'),
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        })

    df = pd.DataFrame(records)
    print(f"✅ Parsed {len(df):,} gambar dari {df['item_id'].nunique():,} item unik.")
    print(f"   Kategori unik: {df['category'].nunique()}")
    return df


df_bbox = parse_bbox_file(BBOX_FILE)
print("\nContoh data:")
print(df_bbox.head(3).to_string())


# ─────────────────────────────────────────────
# CELL 5: Parse list_eval_partition.txt
# Kolom: image_name  item_id  evaluation_status
# evaluation_status: train | query | gallery
# ─────────────────────────────────────────────
def parse_eval_partition(filepath):
    """
    Parse list_eval_partition.txt
    Return: DataFrame dengan kolom: image_name, item_id, split
    """
    records = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    n_total = int(lines[0].strip())
    for line in tqdm(lines[2:n_total + 2], desc="Parsing eval partition"):
        parts = line.strip().split()
        if len(parts) >= 3:
            records.append({
                'image_name': parts[0],
                'item_id'   : parts[1],
                'split'     : parts[2],   # train / query / gallery
            })

    df = pd.DataFrame(records)
    print(f"✅ Parsed {len(df):,} entries.")
    print("   Split distribution:")
    print(df['split'].value_counts().to_string())
    return df


df_eval = parse_eval_partition(EVAL_FILE)


# ─────────────────────────────────────────────
# CELL 6: Gabungkan bbox + eval partition
# ─────────────────────────────────────────────
df_merged = df_bbox.merge(
    df_eval[['image_name', 'split']],
    on='image_name',
    how='left'
)
df_merged['split'] = df_merged['split'].fillna('train')  # beberapa gambar mungkin tidak ada di eval

print(f"\n📊 Merged dataset: {len(df_merged):,} baris")
print(f"   Distribusi split:")
print(df_merged['split'].value_counts())


# ─────────────────────────────────────────────
# CELL 7: Statistik & Distribusi Dataset
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  📊 STATISTIK DATASET")
print("="*55)
print(f"  Total gambar       : {len(df_merged):,}")
print(f"  Total item unik    : {df_merged['item_id'].nunique():,}")
print(f"  Total kategori     : {df_merged['category'].nunique()}")
print(f"  Gender: WOMEN      : {(df_merged['gender']=='WOMEN').sum():,}")
print(f"  Gender: MEN        : {(df_merged['gender']=='MEN').sum():,}")
print("="*55)

# Distribusi kategori
print("\n📊 Top 20 Kategori:")
cat_counts = df_merged.groupby('category')['item_id'].nunique().sort_values(ascending=False)
print(cat_counts.head(20).to_string())

# Rata-rata gambar per item
avg_imgs = df_merged.groupby('item_id').size().mean()
print(f"\n📷 Rata-rata gambar per item: {avg_imgs:.1f}")

# Plot distribusi kategori
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Jumlah item per kategori
cat_counts.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Jumlah Item Unik per Kategori', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Kategori')
axes[0].set_ylabel('Jumlah Item')
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Distribusi clothes type
clothes_dist = df_merged['clothes_type_label'].value_counts()
axes[1].pie(clothes_dist.values, labels=clothes_dist.index, autopct='%1.1f%%',
            colors=['#4C72B0', '#DD8452', '#55A868'])
axes[1].set_title('Distribusi Clothes Type', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'category_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot tersimpan.")


# ─────────────────────────────────────────────
# CELL 8: Validasi Gambar (Cek File Exists + Bisa Dibuka)
# ─────────────────────────────────────────────
print("\n🔍 Validating image files (checking existence)...")

# Cek keberadaan file (cepat, tanpa buka gambar)
df_merged['file_exists'] = df_merged['full_path'].apply(os.path.exists)
n_found   = df_merged['file_exists'].sum()
n_missing = (~df_merged['file_exists']).sum()

print(f"  ✅ Ditemukan : {n_found:,}")
print(f"  ❌ Tidak ada : {n_missing:,}")

if n_missing > 0:
    print(f"\n  ⚠️ Contoh path yang tidak ditemukan:")
    missing_samples = df_merged[~df_merged['file_exists']]['full_path'].head(3).tolist()
    for p in missing_samples:
        print(f"     {p}")
    print("\n  💡 Tips: Pastikan img.zip sudah diekstrak ke folder 'img' di dalam dataset.")

# Filter hanya yang valid
df_valid = df_merged[df_merged['file_exists']].drop(columns=['file_exists']).reset_index(drop=True)
print(f"\n✅ Dataset valid: {len(df_valid):,} gambar")


# ─────────────────────────────────────────────
# CELL 9: Strategi Penggunaan untuk Sistem Rekomendasi
#
# Kita akan menggunakan split sebagai berikut:
#   - 'train' + 'gallery' → DATABASE for recommendation (item yang bisa direkomendasikan)
#   - 'query'             → untuk simulasi evaluasi (user melakukan query)
#
# Ground truth: gambar dengan item_id yang sama = item yang relevan
# (lebih proper daripada hanya category-based!)
# ─────────────────────────────────────────────
# Untuk sistem rekomendasi kita, gunakan semua gambar sebagai database
# tapi untuk evaluasi, kita bisa pakai query vs gallery split

# Pilih hanya SATU foto per item sebagai representasi (misal: front view)
# Ini cocok untuk database rekomendasi (agar tidak ada duplikat item)
PREFERRED_POSES = ['front', 'side', 'full', 'back', 'zoom-in', 'flat']

def get_best_image_per_item(df):
    """Pilih 1 gambar terbaik per item (preferensi: front view)."""
    selected = []
    for item_id, group in tqdm(df.groupby('item_id'), desc="Selecting best image per item"):
        for pose in PREFERRED_POSES:
            subset = group[group['pose_type_label'] == pose]
            if len(subset) > 0:
                selected.append(subset.iloc[0])
                break
        else:
            selected.append(group.iloc[0])  # fallback: gambar pertama
    return pd.DataFrame(selected).reset_index(drop=True)


print("\n🔧 Selecting 1 representative image per item...")
df_representative = get_best_image_per_item(df_valid)
print(f"✅ {len(df_representative):,} item representatif dipilih.")


# ─────────────────────────────────────────────
# CELL 10: Filter Kategori dengan Cukup Item
# (minimal 10 item per kategori untuk evaluasi yang valid)
# ─────────────────────────────────────────────
MIN_ITEMS_PER_CATEGORY = 10
cat_item_count = df_representative.groupby('category')['item_id'].nunique()
valid_categories = cat_item_count[cat_item_count >= MIN_ITEMS_PER_CATEGORY].index.tolist()

df_final = df_representative[df_representative['category'].isin(valid_categories)].reset_index(drop=True)
df_final['db_idx'] = df_final.index   # index di database

print(f"\n📊 Setelah filter (min {MIN_ITEMS_PER_CATEGORY} item/kategori):")
print(f"   Kategori valid : {df_final['category'].nunique()}")
print(f"   Total item     : {len(df_final):,}")

# Simpan juga df_valid LENGKAP (semua gambar) untuk evaluasi query/gallery
df_full = df_valid.copy()
df_full['db_idx'] = df_full.index


# ─────────────────────────────────────────────
# CELL 11: Simpan Master CSV
# ─────────────────────────────────────────────
# 1. Master dataset utama (1 gambar per item — untuk rekomendasi)
df_final.to_csv(MASTER_CSV, index=False)
print(f"\n💾 master_dataset.csv tersimpan: {MASTER_CSV}")

# 2. Full dataset (semua gambar — untuk evaluasi query/gallery)
full_csv = os.path.join(OUTPUT_DIR, 'full_dataset.csv')
df_full.to_csv(full_csv, index=False)
print(f"💾 full_dataset.csv tersimpan  : {full_csv}")

# 3. Simpan daftar kategori
cat_df = df_final.groupby('category').agg(
    n_items   = ('item_id', 'nunique'),
    clothes_type = ('clothes_type_label', 'first'),
    gender    = ('gender', lambda x: '/'.join(sorted(x.unique())))
).sort_values('n_items', ascending=False)
cat_df.to_csv(os.path.join(OUTPUT_DIR, 'categories.csv'))
print(f"💾 categories.csv tersimpan")

print(f"\n📊 Master dataset final:")
print(df_final[['item_id', 'category', 'gender', 'clothes_type_label',
                'pose_type_label', 'split']].head(5).to_string())


# ─────────────────────────────────────────────
# CELL 12: Preview Sampel Gambar per Kategori
# ─────────────────────────────────────────────
def show_sample_images(df, n_categories=6, n_per_cat=4):
    """Tampilkan n_per_cat gambar dari n_categories pertama."""
    cats = df['category'].value_counts().head(n_categories).index.tolist()
    fig, axes = plt.subplots(n_categories, n_per_cat, figsize=(n_per_cat * 2.5, n_categories * 2.8))

    for r, cat in enumerate(cats):
        cat_df = df[df['category'] == cat].sample(
            min(n_per_cat, len(df[df['category'] == cat])), random_state=42
        )
        for c, (_, row) in enumerate(cat_df.iterrows()):
            ax = axes[r, c]
            try:
                img = mpimg.imread(row['full_path'])
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', color='red')
            ax.axis('off')
            if c == 0:
                short_cat = cat.replace('_', ' ')
                ax.set_ylabel(short_cat, fontsize=7, rotation=0, ha='right', va='center')

    plt.suptitle('Sampel Gambar per Kategori — In-shop DeepFashion', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Preview tersimpan.")

show_sample_images(df_final, n_categories=6, n_per_cat=4)


# ─────────────────────────────────────────────
# CELL 13: Summary Akhir
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  ✅ DATA PREPARATION SELESAI")
print("="*55)
print(f"  Dataset         : In-shop Clothes Retrieval Benchmark")
print(f"  Total item      : {len(df_final):,}")
print(f"  Total kategori  : {df_final['category'].nunique()}")
print(f"  Gender coverage : {', '.join(df_final['gender'].unique())}")
print(f"  Master CSV      : {MASTER_CSV}")
print(f"  Full CSV        : {full_csv}")
print("="*55)

print("\n📋 Kategori yang tersedia:")
for cat, row_data in cat_df.iterrows():
    print(f"   {cat:<30} {row_data['n_items']:>4} items  [{row_data['gender']}]")

print("\n➡️ Lanjut ke Notebook 02: Feature Extraction")
