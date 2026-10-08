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

# ── Path (Cross-Platform) ──────────────────────────────────────────────
GRID_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(GRID_DIR)

MODELS_DIR  = os.path.join(GRID_DIR, 'models', 'best')
RESULTS_DIR = os.path.join(GRID_DIR, 'results')
FEAT_DIR    = os.path.join(RESULTS_DIR, 'retrieval_features')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)

FULL_CSV = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')
IMG_DIR  = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')

# ── Konfigurasi model ───────────────────────────────────────────
EMBEDDING_LAYER_NAME = 'embedding'
MODEL_CONFIGS = {
    'resnet50': {
        'class': ResNet50, 'input_size': (256, 256),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
        'h5_file': 'best_resnet50.h5', 'pooling': 'avg',
    },
    'vgg19': {
        'class': VGG19, 'input_size': (256, 256),
        'preprocess': tf.keras.applications.vgg19.preprocess_input,
        'h5_file': 'best_vgg19.h5', 'pooling': 'avg',
    },
    'inceptionv3': {
        'class': InceptionV3, 'input_size': (299, 299),
        'preprocess': tf.keras.applications.inception_v3.preprocess_input,
        'h5_file': 'best_inceptionv3.h5', 'pooling': 'avg',
    },
    'mobilenetv3': {
        'class': MobileNetV3Large, 'input_size': (256, 256),
        'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
        'h5_file': 'best_mobilenetv3.h5', 'pooling': 'avg',
    },
}

BATCH_SIZE = 32
EMBED_DIM = 512
DROPOUT = 0.4

print(f"📌 Eksperimen: GRID SEARCH MULTI-MODEL")
print(f"📁 Models dir: {MODELS_DIR}")
print(f"📁 Results dir: {RESULTS_DIR}")
print(f"📁 Images dir: {IMG_DIR}")

# ── Load Dataset & Statistik Query/Gallery ──────────────────────────────
df_full = pd.read_csv(FULL_CSV)

# Build full path yang dinamis dan aman untuk OS apapun
df_full['full_path'] = df_full['image_name'].apply(
    lambda x: os.path.join(IMG_DIR, os.path.normpath(str(x).replace('img/', '', 1)))
)

df_query   = df_full[df_full['split'] == 'query'].reset_index(drop=True)
df_gallery = df_full[df_full['split'] == 'gallery'].reset_index(drop=True)

print("=" * 55)
print("  📊 STATISTIK DATASET RETRIEVAL")
print("=" * 55)
print(f"  Total gambar full_dataset  : {len(df_full):,}")
print(f"  Gambar QUERY               : {len(df_query):,}")
print(f"  Gambar GALLERY             : {len(df_gallery):,}")
print("=" * 55)

# ── Tentukan Jumlah Kelas ───────────────────────────────────────────────
df_train = df_full[df_full['split'] == 'train'].copy()
n_classes = df_train['category'].nunique()
print(f"📊 Jumlah kelas untuk arsitektur model: {n_classes}")

# ── Fungsi Build Model & Ekstraksi Fitur ────────────────────────────────
def build_extractor(cfg, n_classes):
    """
    Bangun arsitektur model yang SAMA dengan saat grid search,
    load weights dari models/best, lalu potong hingga embedding layer.
    """
    inp = tf.keras.Input(shape=(*cfg['input_size'], 3))
    base = cfg['class'](weights='imagenet', include_top=False, 
                        pooling=cfg['pooling'], input_tensor=inp)
    base.trainable = False
    x = base.output
    x = Dense(EMBED_DIM, activation='relu', name=EMBEDDING_LAYER_NAME)(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(n_classes, activation='softmax')(x)

    full_model = Model(inp, out)

    h5_path = os.path.join(MODELS_DIR, cfg['h5_file'])
    if os.path.exists(h5_path):
        full_model.load_weights(h5_path)
        print(f"  ✅ Weights loaded: {cfg['h5_file']}")
    else:
        print(f"  ⚠️ WARNING: {h5_path} tidak ditemukan!")

    extractor = Model(full_model.input, 
                      full_model.get_layer(EMBEDDING_LAYER_NAME).output)
    return extractor

def extract_features(extractor, image_paths, input_size, preprocess_fn, desc='Extracting'):
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


# ── Ekstraksi Fitur Query & Gallery (Semua Model) ───────────────────────
query_paths   = df_query['full_path'].tolist()
gallery_paths = df_gallery['full_path'].tolist()

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*55}")
    print(f"  🧠 {model_name.upper()} — Ekstraksi Query + Gallery")
    print(f"{'='*55}")

    q_feat_path = os.path.join(FEAT_DIR, f'{model_name}_query_features.npy')
    g_feat_path = os.path.join(FEAT_DIR, f'{model_name}_gallery_features.npy')
    q_idx_path  = os.path.join(FEAT_DIR, f'{model_name}_query_valid_idx.npy')
    g_idx_path  = os.path.join(FEAT_DIR, f'{model_name}_gallery_valid_idx.npy')

    if os.path.exists(q_feat_path) and os.path.exists(g_feat_path):
        print(f"  ⏩ Fitur sudah ada, skip!")
        continue

    extractor = build_extractor(cfg, n_classes)

    print(f"\n  📌 Extracting QUERY features ({len(query_paths):,} images)...")
    q_feats, q_valid = extract_features(
        extractor, query_paths, cfg['input_size'], cfg['preprocess'],
        desc=f"Query [{model_name}]"
    )

    print(f"  📌 Extracting GALLERY features ({len(gallery_paths):,} images)...")
    g_feats, g_valid = extract_features(
        extractor, gallery_paths, cfg['input_size'], cfg['preprocess'],
        desc=f"Gallery [{model_name}]"
    )

    np.save(q_feat_path, q_feats)
    np.save(g_feat_path, g_feats)
    np.save(q_idx_path, np.array(q_valid))
    np.save(g_idx_path, np.array(g_valid))

    print(f"  ✅ Query features shape  : {q_feats.shape}")
    print(f"  ✅ Gallery features shape: {g_feats.shape}")
    print(f"  💾 Tersimpan di: {FEAT_DIR}")

    del extractor, q_feats, g_feats
    gc.collect()
    tf.keras.backend.clear_session()

print("\n🎉 Ekstraksi fitur selesai untuk semua model!")

# ── Fungsi Evaluasi Retrieval ───────────────────────────────────────────
def evaluate_retrieval(query_features, gallery_features, 
                       query_item_ids, gallery_item_ids, 
                       k_values=[1, 5, 10, 20]):
    n_queries = len(query_features)
    similarity_matrix = query_features @ gallery_features.T

    results = {}
    recall_at_k = {k: 0 for k in k_values}
    average_precisions = []

    for i in range(n_queries):
        query_id = query_item_ids[i]
        gt_mask = (gallery_item_ids == query_id)
        n_relevant = gt_mask.sum()

        if n_relevant == 0:
            continue

        sorted_indices = np.argsort(-similarity_matrix[i])

        for k in k_values:
            top_k_indices = sorted_indices[:k]
            top_k_ids = gallery_item_ids[top_k_indices]
            if query_id in top_k_ids:
                recall_at_k[k] += 1

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
        results[f'HitRate@{k}'] = recall_at_k[k] / n_evaluated if n_evaluated > 0 else 0

    results['mAP'] = np.mean(average_precisions) if average_precisions else 0
    results['n_queries_evaluated'] = n_evaluated

    return results

# ── Jalankan Evaluasi untuk Semua Model ─────────────────────────────────
print("\n" + "=" * 70)
print(f"  📊 EVALUASI RETRIEVAL — GRID SEARCH")
print("=" * 70)

all_results = {}

for model_name in MODEL_CONFIGS.keys():
    print(f"\n🧠 Evaluating: {model_name.upper()}")

    q_feats = np.load(os.path.join(FEAT_DIR, f'{model_name}_query_features.npy'))
    g_feats = np.load(os.path.join(FEAT_DIR, f'{model_name}_gallery_features.npy'))
    q_valid = np.load(os.path.join(FEAT_DIR, f'{model_name}_query_valid_idx.npy'))
    g_valid = np.load(os.path.join(FEAT_DIR, f'{model_name}_gallery_valid_idx.npy'))

    q_item_ids = df_query.iloc[q_valid]['item_id'].values
    g_item_ids = df_gallery.iloc[g_valid]['item_id'].values

    results = evaluate_retrieval(
        q_feats, g_feats, q_item_ids, g_item_ids,
        k_values=[1, 5, 10, 20]
    )

    all_results[model_name] = results
    print(f"  HitRate@1:  {results['HitRate@1']:.4f}")
    print(f"  HitRate@5:  {results['HitRate@5']:.4f}")
    print(f"  HitRate@10: {results['HitRate@10']:.4f}")
    print(f"  HitRate@20: {results['HitRate@20']:.4f}")
    print(f"  mAP:        {results['mAP']:.4f}")

# ── Tabel Perbandingan & CSV ────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"  📊 TABEL PERBANDINGAN RETRIEVAL — GRID SEARCH")
print("=" * 80)
print(f"{'Model':<15} {'HitRate@1':>10} {'HitRate@5':>10} {'HitRate@10':>11} {'HitRate@20':>11} {'mAP':>10}")
print("-" * 80)

for model_name, results in all_results.items():
    print(f"{model_name:<15} "
          f"{results['HitRate@1']:>10.4f} "
          f"{results['HitRate@5']:>10.4f} "
          f"{results['HitRate@10']:>11.4f} "
          f"{results['HitRate@20']:>11.4f} "
          f"{results['mAP']:>10.4f}")

print("-" * 80)

df_results = pd.DataFrame(all_results).T
df_results = df_results.drop(columns=['n_queries_evaluated'])
csv_out = os.path.join(RESULTS_DIR, 'grid_search_hit_miss_results.csv')
df_results.to_csv(csv_out)
print(f"💾 Hasil Hit/Miss berhasil disimpan ke: {csv_out}")

# ── Bar Chart Perbandingan ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

models = list(all_results.keys())
x = np.arange(len(models))
width = 0.2

for i, k in enumerate([1, 5, 10, 20]):
    values = [all_results[m][f'HitRate@{k}'] for m in models]
    axes[0].bar(x + i * width, values, width, label=f'HitRate@{k}')

axes[0].set_xlabel('Model CNN')
axes[0].set_ylabel('Hit Rate')
axes[0].set_title('HitRate@K per Model — Grid Search Best', fontweight='bold')
axes[0].set_xticks(x + 1.5 * width)
axes[0].set_xticklabels([m.upper() for m in models])
axes[0].legend()
axes[0].set_ylim(0, 1.0)
axes[0].grid(axis='y', alpha=0.3)

map_values = [all_results[m]['mAP'] for m in models]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = axes[1].bar([m.upper() for m in models], map_values, color=colors)
axes[1].set_xlabel('Model CNN')
axes[1].set_ylabel('mAP')
axes[1].set_title('Mean Average Precision — Grid Search Best', fontweight='bold')
axes[1].set_ylim(0, max(map_values) * 1.3 if map_values else 1.0)
axes[1].grid(axis='y', alpha=0.3)

for bar, val in zip(bars, map_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'grid_search_retrieval_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart tersimpan di results/ !")

# ── Visualisasi Kualitatif ──────────────────────────────────────────────
def visualize_retrieval(model_name, df_query, df_gallery, 
                        query_features, gallery_features,
                        query_valid_idx, gallery_valid_idx,
                        n_samples=5, top_k=5, seed=42):
    np.random.seed(seed)
    q_item_ids = df_query.iloc[query_valid_idx]['item_id'].values
    g_item_ids = df_gallery.iloc[gallery_valid_idx]['item_id'].values
    sim_matrix = query_features @ gallery_features.T
    
    sample_indices = np.random.choice(len(query_features), 
                                       size=min(n_samples, len(query_features)), 
                                       replace=False)

    fig, axes = plt.subplots(n_samples, top_k + 1, 
                              figsize=(3 * (top_k + 1), 3.5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Retrieval Results — {model_name.upper()}\n'
                 f'🟢 HIT (item sama)  🔴 MISS (item beda)', 
                 fontsize=14, fontweight='bold', y=1.02)

    for row, q_idx in enumerate(sample_indices):
        query_id = q_item_ids[q_idx]
        query_df_idx = query_valid_idx[q_idx]
        query_row = df_query.iloc[query_df_idx]

        ax = axes[row, 0]
        try:
            img = Image.open(query_row['full_path']).convert('RGB')
            ax.imshow(img)
        except:
            pass
        ax.set_title(f"QUERY\n{query_id}\n{query_row.get('category', '?')}", 
                     fontsize=8, fontweight='bold', color='blue')
        ax.axis('off')
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                  linewidth=4, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

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
                pass

            status = "✅ HIT" if is_hit else "❌ MISS"
            color = 'green' if is_hit else 'red'
            ax.set_title(f"#{col+1} {status}\n{gallery_id}\nsim={sim_score:.3f}", 
                         fontsize=7, color=color, fontweight='bold')
            ax.axis('off')
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                      linewidth=4, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'retrieval_visual_{model_name}_grid_search.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

N_SAMPLES = 5
TOP_K = 5
print("\n" + "=" * 70)
print(f"  🖼️  MEMBUAT VISUALISASI RETRIEVAL")
print("=" * 70)
for model_name in MODEL_CONFIGS.keys():
    print(f"  Generating visual for {model_name.upper()}...")
    q_feats = np.load(os.path.join(FEAT_DIR, f'{model_name}_query_features.npy'))
    g_feats = np.load(os.path.join(FEAT_DIR, f'{model_name}_gallery_features.npy'))
    q_valid = np.load(os.path.join(FEAT_DIR, f'{model_name}_query_valid_idx.npy'))
    g_valid = np.load(os.path.join(FEAT_DIR, f'{model_name}_gallery_valid_idx.npy'))
    
    visualize_retrieval(model_name, df_query, df_gallery, q_feats, g_feats, q_valid, g_valid, 
                        n_samples=N_SAMPLES, top_k=TOP_K, seed=42)

print(f"\n✅ Proses evaluasi dan visualisasi selesai! Semua output tersimpan di {RESULTS_DIR}")
