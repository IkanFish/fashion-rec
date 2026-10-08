# %% [markdown]
# # 🔍 Eksperimen Evaluasi Retrieval — Sisi Visi Komputer
# 
# **Tujuan:** Menguji apakah model CNN (yang sudah di-fine-tune) mampu 
# mengenali item pakaian yang SAMA dari foto dengan angle berbeda.
#
# **Metode:** 
# 1. Ekstrak fitur dari gambar `query` dan `gallery` menggunakan model `.h5`
# 2. Untuk setiap query, hitung cosine similarity terhadap semua gallery
# 3. Cek apakah gallery dengan `item_id` yang sama muncul di Top-K
# 4. Hitung Recall@K dan mAP
# 5. Visualisasikan hasil retrieval secara kualitatif
#
# **Environment:** WSL2 + TensorFlow GPU (RTX 4060 Ti)

# %% [markdown]
# ## Cell 1: Import & GPU Setup

# %%
import os, gc, time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPU available: {len(gpus) > 0}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")

# %% [markdown]
# ## Cell 2: Konfigurasi Path & Model

# %%
# ── Path (Cross-Platform) ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE_DIR, 'features')
FULL_CSV = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')
IMG_DIR  = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')

# ── Pilih eksperimen mana yang dievaluasi ──────────────────────
# Ubah ke 'exp2' jika ingin mengevaluasi Lightweight Fine-Tuning
EXPERIMENT = 'exp3'

# ── Konfigurasi model ───────────────────────────────────────────
# Exp 2: input 224x224 (InceptionV3=299), layer name='feature_embedding'
# Exp 3: input 256x256 (InceptionV3=299), layer name='embedding'
if EXPERIMENT == 'exp3':
    EMBEDDING_LAYER_NAME = 'embedding'
    MODEL_CONFIGS = {
        'resnet50': {
            'class': ResNet50, 'input_size': (256, 256),
            'preprocess': tf.keras.applications.resnet50.preprocess_input,
            'h5_file': 'resnet50_exp3.h5', 'pooling': 'avg',
        },
        'vgg19': {
            'class': VGG19, 'input_size': (256, 256),
            'preprocess': tf.keras.applications.vgg19.preprocess_input,
            'h5_file': 'vgg19_exp3.h5', 'pooling': 'avg',
        },
        'inceptionv3': {
            'class': InceptionV3, 'input_size': (299, 299),
            'preprocess': tf.keras.applications.inception_v3.preprocess_input,
            'h5_file': 'inceptionv3_exp3.h5', 'pooling': 'avg',
        },
        'mobilenetv3': {
            'class': MobileNetV3Large, 'input_size': (256, 256),
            'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
            'h5_file': 'mobilenetv3_exp3.h5', 'pooling': 'avg',
        },
    }
else:  # exp2
    EMBEDDING_LAYER_NAME = 'feature_embedding'
    MODEL_CONFIGS = {
        'resnet50': {
            'class': ResNet50, 'input_size': (224, 224),
            'preprocess': tf.keras.applications.resnet50.preprocess_input,
            'h5_file': 'resnet50_finetuned.h5', 'pooling': None,
        },
        'vgg19': {
            'class': VGG19, 'input_size': (224, 224),
            'preprocess': tf.keras.applications.vgg19.preprocess_input,
            'h5_file': 'vgg19_finetuned.h5', 'pooling': None,
        },
        'inceptionv3': {
            'class': InceptionV3, 'input_size': (299, 299),
            'preprocess': tf.keras.applications.inception_v3.preprocess_input,
            'h5_file': 'inceptionv3_finetuned.h5', 'pooling': None,
        },
        'mobilenetv3': {
            'class': MobileNetV3Large, 'input_size': (224, 224),
            'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
            'h5_file': 'mobilenetv3_finetuned.h5', 'pooling': None,
        },
    }

BATCH_SIZE = 32
EMBED_DIM = 512

print(f"📌 Eksperimen: {EXPERIMENT.upper()}")
print(f"📁 Features dir: {FEAT_DIR}")
print(f"📁 Images dir: {IMG_DIR}")

# %% [markdown]
# ## Cell 3: Load Dataset & Statistik Query/Gallery

# %%
df_full = pd.read_csv(FULL_CSV)

# Build full path yang dinamis dan aman untuk OS apapun
df_full['full_path'] = df_full['image_name'].apply(
    lambda x: os.path.join(IMG_DIR, os.path.normpath(str(x).replace('img/', '', 1)))
)

# Pisahkan query dan gallery
df_query   = df_full[df_full['split'] == 'query'].reset_index(drop=True)
df_gallery = df_full[df_full['split'] == 'gallery'].reset_index(drop=True)

print("=" * 55)
print("  📊 STATISTIK DATASET RETRIEVAL")
print("=" * 55)
print(f"  Total gambar full_dataset  : {len(df_full):,}")
print(f"  Gambar QUERY               : {len(df_query):,}")
print(f"  Gambar GALLERY             : {len(df_gallery):,}")
print(f"  Item unik di QUERY         : {df_query['item_id'].nunique():,}")
print(f"  Item unik di GALLERY       : {df_gallery['item_id'].nunique():,}")
print("=" * 55)

# Cek overlap item_id antara query dan gallery
query_items   = set(df_query['item_id'].unique())
gallery_items = set(df_gallery['item_id'].unique())
overlap = query_items & gallery_items
print(f"\n  Item ID yang ada di KEDUA set: {len(overlap):,}")
print(f"  (Ini yang bisa dievaluasi — sisanya tidak punya pasangan)")

# Hitung berapa gambar gallery per item (bisa >1)
gallery_per_item = df_gallery[df_gallery['item_id'].isin(overlap)].groupby('item_id').size()
print(f"\n  Rata-rata gambar gallery per item: {gallery_per_item.mean():.1f}")
print(f"  Min: {gallery_per_item.min()}, Max: {gallery_per_item.max()}")

# %% [markdown]
# ## Cell 4: Fungsi Build Model & Ekstraksi Fitur

# %%
def build_extractor(cfg, n_classes):
    """
    Bangun arsitektur model yang SAMA dengan saat training,
    load weights dari .h5, lalu potong hingga embedding layer saja.
    """
    if cfg.get('pooling'):
        # Exp 3 architecture: pooling='avg' langsung di base
        inp = tf.keras.Input(shape=(*cfg['input_size'], 3))
        base = cfg['class'](weights='imagenet', include_top=False, 
                            pooling=cfg['pooling'], input_tensor=inp)
        base.trainable = False
        x = base.output
    else:
        # Exp 2 architecture: tanpa pooling di base, GAP manual
        base = cfg['class'](weights='imagenet', include_top=False, 
                            input_shape=cfg['input_size'] + (3,))
        base.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        inp = base.input

    x   = Dense(EMBED_DIM, activation='relu', name=EMBEDDING_LAYER_NAME)(x)
    x   = Dropout(0.4 if EXPERIMENT == 'exp3' else 0.3)(x)
    out = Dense(n_classes, activation='softmax')(x)

    full_model = Model(inp, out)

    # Load trained weights dari folder hasil restrukturisasi
    h5_path = os.path.join(FEAT_DIR, f'exp3_partial_unfreeze', cfg['h5_file']) if EXPERIMENT == 'exp3' else os.path.join(FEAT_DIR, f'exp2_finetuned', cfg['h5_file'])
    if os.path.exists(h5_path):
        full_model.load_weights(h5_path)
        print(f"  ✅ Weights loaded: {cfg['h5_file']}")
    else:
        print(f"  ⚠️ WARNING: {h5_path} tidak ditemukan!")

    # Potong: hanya sampai embedding layer
    extractor = Model(full_model.input, 
                      full_model.get_layer(EMBEDDING_LAYER_NAME).output)
    return extractor


def extract_features(extractor, image_paths, input_size, preprocess_fn, desc='Extracting'):
    """Ekstrak fitur dari list image paths, return (features, valid_indices)."""
    features = []
    valid_idx = []

    for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=desc):
        batch_paths = image_paths[start:start + BATCH_SIZE]
        batch_imgs = []
        batch_indices = []

        for i, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB').resize(input_size)
                arr = np.array(img, dtype=np.float32)
                arr = np.expand_dims(arr, axis=0)
                arr = preprocess_fn(arr)
                batch_imgs.append(arr)
                batch_indices.append(start + i)
            except Exception:
                continue

        if not batch_imgs:
            continue

        batch_array = np.vstack(batch_imgs)
        batch_feats = extractor.predict(batch_array, verbose=0)
        features.append(batch_feats)
        valid_idx.extend(batch_indices)

    if not features:
        return np.array([]), []

    feat_matrix = np.vstack(features)

    # L2 Normalize
    norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    feat_matrix = feat_matrix / norms

    return feat_matrix, valid_idx

# %% [markdown]
# ## Cell 5: Tentukan Jumlah Kelas (Harus Sama dengan Saat Training)

# %%
# Kita perlu mengetahui n_classes yang sama dengan saat training
# agar arsitektur model persis sama untuk load weights

df_train = df_full[df_full['split'] == 'train'].copy()

if EXPERIMENT == 'exp2':
    # Exp 2: filter kategori >= 20 items
    cat_counts = df_train['category'].value_counts()
    valid_cats = cat_counts[cat_counts >= 20].index.tolist()
    n_classes = len(valid_cats)
else:
    # Exp 3: menggunakan semua kategori unik di train set
    n_classes = df_train['category'].nunique()

print(f"📊 Jumlah kelas untuk arsitektur model: {n_classes}")

# %% [markdown]
# ## Cell 6: Ekstraksi Fitur Query & Gallery (Semua Model)
# 
# Cell ini akan mengekstrak fitur dari gambar query dan gallery.
# Hasil disimpan ke `.npy` agar tidak perlu diulang.

# %%
RETRIEVAL_FEAT_DIR = os.path.join(FEAT_DIR, 'exp4_retrieval')
os.makedirs(RETRIEVAL_FEAT_DIR, exist_ok=True)

query_paths   = df_query['full_path'].tolist()
gallery_paths = df_gallery['full_path'].tolist()

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*55}")
    print(f"  🧠 {model_name.upper()} — Ekstraksi Query + Gallery")
    print(f"{'='*55}")

    q_feat_path = os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_features.npy')
    g_feat_path = os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_features.npy')
    q_idx_path  = os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_valid_idx.npy')
    g_idx_path  = os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_valid_idx.npy')

    # Skip jika sudah ada
    if os.path.exists(q_feat_path) and os.path.exists(g_feat_path):
        print(f"  ⏩ Fitur sudah ada, skip!")
        continue

    # Build extractor
    extractor = build_extractor(cfg, n_classes)

    # Ekstrak query features
    print(f"\n  📌 Extracting QUERY features ({len(query_paths):,} images)...")
    q_feats, q_valid = extract_features(
        extractor, query_paths, cfg['input_size'], cfg['preprocess'],
        desc=f"Query [{model_name}]"
    )

    # Ekstrak gallery features
    print(f"  📌 Extracting GALLERY features ({len(gallery_paths):,} images)...")
    g_feats, g_valid = extract_features(
        extractor, gallery_paths, cfg['input_size'], cfg['preprocess'],
        desc=f"Gallery [{model_name}]"
    )

    # Simpan
    np.save(q_feat_path, q_feats)
    np.save(g_feat_path, g_feats)
    np.save(q_idx_path, np.array(q_valid))
    np.save(g_idx_path, np.array(g_valid))

    print(f"  ✅ Query features shape  : {q_feats.shape}")
    print(f"  ✅ Gallery features shape: {g_feats.shape}")
    print(f"  💾 Tersimpan di: {RETRIEVAL_FEAT_DIR}")

    # Cleanup
    del extractor, q_feats, g_feats
    gc.collect()
    tf.keras.backend.clear_session()

print("\n🎉 Ekstraksi fitur selesai untuk semua model!")

# %% [markdown]
# ## Cell 7: Fungsi Evaluasi Retrieval (Recall@K, mAP)

# %%
def evaluate_retrieval(query_features, gallery_features, 
                       query_item_ids, gallery_item_ids, 
                       k_values=[1, 5, 10, 20]):
    """
    Evaluasi retrieval: untuk setiap query, cari gallery terdekat
    dan cek apakah item_id yang sama ada di top-K.
    
    Returns: dict dengan Recall@K dan mAP
    """
    n_queries = len(query_features)

    # Cosine similarity (karena sudah L2-normalized, dot product = cosine sim)
    similarity_matrix = query_features @ gallery_features.T  # (n_query, n_gallery)

    results = {}
    recall_at_k = {k: 0 for k in k_values}
    average_precisions = []

    for i in range(n_queries):
        query_id = query_item_ids[i]

        # Ground truth: gallery indices yang punya item_id sama
        gt_mask = (gallery_item_ids == query_id)
        n_relevant = gt_mask.sum()

        if n_relevant == 0:
            continue  # Skip query tanpa pasangan di gallery

        # Rank gallery berdasarkan similarity (descending)
        sorted_indices = np.argsort(-similarity_matrix[i])

        # Recall@K
        for k in k_values:
            top_k_indices = sorted_indices[:k]
            top_k_ids = gallery_item_ids[top_k_indices]
            if query_id in top_k_ids:
                recall_at_k[k] += 1

        # Average Precision (untuk mAP)
        ap = 0.0
        n_correct = 0
        for rank, idx in enumerate(sorted_indices):
            if gt_mask[idx]:
                n_correct += 1
                ap += n_correct / (rank + 1)
        ap /= n_relevant
        average_precisions.append(ap)

    n_evaluated = len(average_precisions)

    for k in k_values:
        results[f'Recall@{k}'] = recall_at_k[k] / n_evaluated if n_evaluated > 0 else 0

    results['mAP'] = np.mean(average_precisions) if average_precisions else 0
    results['n_queries_evaluated'] = n_evaluated

    return results

# %% [markdown]
# ## Cell 8: Jalankan Evaluasi untuk Semua Model

# %%
print("=" * 70)
print(f"  📊 EVALUASI RETRIEVAL — {EXPERIMENT.upper()}")
print("=" * 70)

all_results = {}

for model_name in MODEL_CONFIGS.keys():
    print(f"\n🧠 Evaluating: {model_name.upper()}")

    # Load features
    q_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_features.npy'))
    g_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_features.npy'))
    q_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_valid_idx.npy'))
    g_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_valid_idx.npy'))

    # Ambil item_id yang sesuai dengan valid indices
    q_item_ids = df_query.iloc[q_valid]['item_id'].values
    g_item_ids = df_gallery.iloc[g_valid]['item_id'].values

    # Evaluasi
    results = evaluate_retrieval(
        q_feats, g_feats, q_item_ids, g_item_ids,
        k_values=[1, 5, 10, 20]
    )

    all_results[model_name] = results
    print(f"  Recall@1:  {results['Recall@1']:.4f}")
    print(f"  Recall@5:  {results['Recall@5']:.4f}")
    print(f"  Recall@10: {results['Recall@10']:.4f}")
    print(f"  Recall@20: {results['Recall@20']:.4f}")
    print(f"  mAP:       {results['mAP']:.4f}")
    print(f"  Queries evaluated: {results['n_queries_evaluated']}")

# %% [markdown]
# ## Cell 9: Tabel Perbandingan Hasil

# %%
print("\n" + "=" * 80)
print(f"  📊 TABEL PERBANDINGAN RETRIEVAL — {EXPERIMENT.upper()}")
print("=" * 80)
print(f"{'Model':<15} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>11} {'Recall@20':>11} {'mAP':>10}")
print("-" * 80)

for model_name, results in all_results.items():
    print(f"{model_name:<15} "
          f"{results['Recall@1']:>10.4f} "
          f"{results['Recall@5']:>10.4f} "
          f"{results['Recall@10']:>11.4f} "
          f"{results['Recall@20']:>11.4f} "
          f"{results['mAP']:>10.4f}")

print("-" * 80)

# Buat DataFrame untuk visualisasi yang lebih rapi
df_results = pd.DataFrame(all_results).T
df_results = df_results.drop(columns=['n_queries_evaluated'])
df_results = df_results.rename(columns={'Recall@1': 'Hit@1', 'Recall@5': 'Hit@5', 'Recall@10': 'Hit@10', 'Recall@20': 'Hit@20'})
print("\n📋 DataFrame Hasil Hit Rate (Keseluruhan Dataset):")
print(df_results.to_string())

# SIMPAN KE CSV
csv_out = os.path.join(RETRIEVAL_FEAT_DIR, f'cnn_hit_miss_results_{EXPERIMENT}.csv')
df_results.to_csv(csv_out)
print(f"💾 Hasil Hit/Miss secara keseluruhan berhasil disimpan ke:\n   {csv_out}")
# %% [markdown]
# ## Cell 10: Bar Chart Perbandingan Recall@K

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Recall@K
models = list(all_results.keys())
x = np.arange(len(models))
width = 0.2

for i, k in enumerate([1, 5, 10, 20]):
    values = [all_results[m][f'Recall@{k}'] for m in models]
    axes[0].bar(x + i * width, values, width, label=f'Recall@{k}')

axes[0].set_xlabel('Model CNN')
axes[0].set_ylabel('Recall')
axes[0].set_title(f'Recall@K per Model — {EXPERIMENT.upper()}', fontweight='bold')
axes[0].set_xticks(x + 1.5 * width)
axes[0].set_xticklabels([m.upper() for m in models])
axes[0].legend()
axes[0].set_ylim(0, 1.0)
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: mAP
map_values = [all_results[m]['mAP'] for m in models]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = axes[1].bar([m.upper() for m in models], map_values, color=colors)
axes[1].set_xlabel('Model CNN')
axes[1].set_ylabel('mAP')
axes[1].set_title(f'Mean Average Precision — {EXPERIMENT.upper()}', fontweight='bold')
axes[1].set_ylim(0, max(map_values) * 1.3 if map_values else 1.0)
axes[1].grid(axis='y', alpha=0.3)

# Tambahkan value labels
for bar, val in zip(bars, map_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RETRIEVAL_FEAT_DIR, f'retrieval_comparison_{EXPERIMENT}.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart tersimpan!")

# %% [markdown]
# ## Cell 11: Visualisasi Kualitatif — Query vs Top-K Gallery
#
# Ini bagian paling penting: melihat dengan mata sendiri apakah 
# model benar-benar me-retrieve item yang tepat.

# %%
def visualize_retrieval(model_name, df_query, df_gallery, 
                        query_features, gallery_features,
                        query_valid_idx, gallery_valid_idx,
                        n_samples=5, top_k=5, seed=42):
    """
    Tampilkan n_samples query secara acak beserta top_k gallery results.
    Bingkai HIJAU = item_id sama (HIT), bingkai MERAH = item_id beda (MISS)
    """
    np.random.seed(seed)

    # Ambil item IDs
    q_item_ids = df_query.iloc[query_valid_idx]['item_id'].values
    g_item_ids = df_gallery.iloc[gallery_valid_idx]['item_id'].values

    # Cosine similarity
    sim_matrix = query_features @ gallery_features.T

    # Pilih random query indices
    sample_indices = np.random.choice(len(query_features), 
                                       size=min(n_samples, len(query_features)), 
                                       replace=False)

    fig, axes = plt.subplots(n_samples, top_k + 1, 
                              figsize=(3 * (top_k + 1), 3.5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Retrieval Results — {model_name.upper()} ({EXPERIMENT.upper()})\n'
                 f'🟢 HIT (item sama)  🔴 MISS (item beda)', 
                 fontsize=14, fontweight='bold', y=1.02)

    for row, q_idx in enumerate(sample_indices):
        query_id = q_item_ids[q_idx]
        query_df_idx = query_valid_idx[q_idx]
        query_row = df_query.iloc[query_df_idx]

        # Tampilkan query image
        ax = axes[row, 0]
        try:
            img = Image.open(query_row['full_path']).convert('RGB')
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        
        ax.set_title(f"QUERY\n{query_id}\n{query_row.get('category', '?')}", 
                     fontsize=8, fontweight='bold', color='blue')
        ax.axis('off')

        # Bingkai biru untuk query
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                  linewidth=4, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

        # Top-K gallery results
        sorted_gallery = np.argsort(-sim_matrix[q_idx])

        for col, g_rank_idx in enumerate(sorted_gallery[:top_k]):
            ax = axes[row, col + 1]
            gallery_df_idx = gallery_valid_idx[g_rank_idx]
            gallery_row = df_gallery.iloc[gallery_df_idx]
            gallery_id = g_item_ids[g_rank_idx]
            sim_score = sim_matrix[q_idx, g_rank_idx]

            is_hit = (gallery_id == query_id)

            try:
                img = Image.open(gallery_row['full_path']).convert('RGB')
                ax.imshow(img)
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')

            # Label
            status = "✅ HIT" if is_hit else "❌ MISS"
            color = 'green' if is_hit else 'red'
            ax.set_title(f"#{col+1} {status}\n{gallery_id}\nsim={sim_score:.3f}", 
                         fontsize=7, color=color, fontweight='bold')
            ax.axis('off')

            # Bingkai hijau/merah
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                      linewidth=4, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(os.path.join(RETRIEVAL_FEAT_DIR, 
                f'retrieval_visual_{model_name}_{EXPERIMENT}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## Cell 12: Visualisasi untuk Setiap Model
#
# Jalankan cell di bawah ini untuk melihat hasil retrieval visual
# dari setiap model. Ubah `N_SAMPLES` dan `TOP_K` sesuai kebutuhan.

# %%
N_SAMPLES = 5   # Jumlah query yang ditampilkan
TOP_K = 5        # Jumlah gallery results per query
SEED = 42       # Random seed (ubah untuk melihat sampel berbeda)

for model_name in MODEL_CONFIGS.keys():
    print(f"\n{'='*55}")
    print(f"  🖼️  Visualisasi: {model_name.upper()}")
    print(f"{'='*55}")

    # Load features
    q_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_features.npy'))
    g_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_features.npy'))
    q_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_valid_idx.npy'))
    g_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_valid_idx.npy'))

    visualize_retrieval(
        model_name=model_name,
        df_query=df_query,
        df_gallery=df_gallery,
        query_features=q_feats,
        gallery_features=g_feats,
        query_valid_idx=q_valid,
        gallery_valid_idx=g_valid,
        n_samples=N_SAMPLES,
        top_k=TOP_K,
        seed=SEED
    )

# %% [markdown]
# ## Cell 13: Analisis Per Kategori (Opsional)
# 
# Melihat di kategori mana model paling kuat/lemah

# %%
def per_category_recall(model_name, k=5):
    """Hitung Recall@K per kategori untuk analisis mendalam."""
    q_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_features.npy'))
    g_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_features.npy'))
    q_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_valid_idx.npy'))
    g_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_valid_idx.npy'))

    q_info = df_query.iloc[q_valid][['item_id', 'category']].values
    g_item_ids = df_gallery.iloc[g_valid]['item_id'].values

    sim_matrix = q_feats @ g_feats.T

    cat_hits = {}
    cat_total = {}

    for i in range(len(q_feats)):
        query_id = q_info[i, 0]
        query_cat = q_info[i, 1]

        # Ada pasangan di gallery?
        if query_id not in g_item_ids:
            continue

        if query_cat not in cat_total:
            cat_total[query_cat] = 0
            cat_hits[query_cat] = 0

        cat_total[query_cat] += 1

        top_k_indices = np.argsort(-sim_matrix[i])[:k]
        top_k_ids = g_item_ids[top_k_indices]
        if query_id in top_k_ids:
            cat_hits[query_cat] += 1

    results = []
    for cat in sorted(cat_total.keys()):
        recall = cat_hits[cat] / cat_total[cat] if cat_total[cat] > 0 else 0
        results.append({'category': cat, 'recall': recall, 
                        'hits': cat_hits[cat], 'total': cat_total[cat]})

    return pd.DataFrame(results).sort_values('recall', ascending=False)

# Jalankan untuk SEMUA model
for model_name in MODEL_CONFIGS.keys():
    print(f"\n📊 Recall@5 per Kategori — {model_name.upper()}")
    df_cat = per_category_recall(model_name, k=5)
    print(df_cat.to_string(index=False))
    
    # Simpan ke CSV
    csv_cat_out = os.path.join(RETRIEVAL_FEAT_DIR, f'recall_per_category_{model_name}_{EXPERIMENT}.csv')
    df_cat.to_csv(csv_cat_out, index=False)
    
    # Visualisasi
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlGn(df_cat['recall'].values)
    bars = ax.barh(df_cat['category'], df_cat['recall'], color=colors)
    ax.set_xlabel('Recall@5')
    ax.set_title(f'Recall@5 per Kategori — {model_name.upper()} ({EXPERIMENT.upper()})',
                 fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, df_cat['recall'].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2%}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RETRIEVAL_FEAT_DIR, 
                f'recall_per_category_{model_name}_{EXPERIMENT}.png'),
                dpi=150, bbox_inches='tight')
    # plt.show() # Tidak perlu ditampilkan interaktif saat looping
    plt.close()

print("\n✅ Analisis per kategori selesai untuk semua model!")

# %% [markdown]
# ## Selesai! 🎉
# 
# ### Output yang dihasilkan:
# - **Tabel metrik**: Recall@1, Recall@5, Recall@10, Recall@20, mAP untuk 4 model
# - **Bar chart**: Perbandingan visual antar model
# - **Visualisasi retrieval**: Query image + Top-K gallery results (hijau=HIT, merah=MISS)
# - **Analisis per kategori**: Kategori mana yang paling mudah/sulit di-retrieve
#
# ### Langkah selanjutnya:
# - Jika hasil retrieval memuaskan → model CNN tervalidasi → lanjut ke evaluasi reksis
# - Jika tidak → pertimbangkan fine-tuning lebih lanjut atau arsitektur berbeda
