# %% [markdown]
# # 🔍 Evaluasi Retrieval — Eksperimen 3e (tf.data + Full Augmentation)
#
# **Tujuan:** Menguji Hit Rate@K model ResNet50 yang dilatih di exp3e
# menggunakan protokol query/gallery DeepFashion.
#
# **Lingkungan:** WSL2 + TensorFlow GPU (RTX 4060 Ti)

import os, gc, time, platform
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ── Path ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE_DIR, 'features')
FULL_CSV = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')

if platform.system() == 'Linux' and os.path.exists('/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'):
    IMG_DIR = '/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'
else:
    IMG_DIR = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')

# ── Konfigurasi ──────────────────────
EXPERIMENT = 'exp3e'
EMBEDDING_LAYER_NAME = 'embedding'
BATCH_SIZE = 24
EMBED_DIM = 512

MODEL_CONFIGS = {
    'resnet50': {
        'class': ResNet50, 'input_size': (256, 256),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
        'h5_file': 'resnet50_exp3e.h5', 'pooling': 'avg',
    },
}

print(f"📌 Eksperimen: {EXPERIMENT.upper()}")
print(f"📁 Images dir: {IMG_DIR}")

# ── Dataset Loading ──────────────────────────────────────────────
df_full = pd.read_csv(FULL_CSV)

df_full['full_path'] = df_full['image_name'].apply(
    lambda x: os.path.join(IMG_DIR, os.path.normpath(str(x).replace('img/', '', 1)))
)

df_query   = df_full[df_full['split'] == 'query'].reset_index(drop=True)
df_gallery = df_full[df_full['split'] == 'gallery'].reset_index(drop=True)

print("=" * 55)
print("  📊 STATISTIK DATASET RETRIEVAL")
print("=" * 55)
print(f"  Gambar QUERY   : {len(df_query):,}")
print(f"  Gambar GALLERY : {len(df_gallery):,}")
print(f"  Item unik QUERY   : {df_query['item_id'].nunique():,}")
print(f"  Item unik GALLERY : {df_gallery['item_id'].nunique():,}")

query_items   = set(df_query['item_id'].unique())
gallery_items = set(df_gallery['item_id'].unique())
overlap = query_items & gallery_items
print(f"  Item ID overlap   : {len(overlap):,}")

# ── Build Extractor ──────────────────────────────────────────────
def build_extractor(cfg, n_classes):
    inp = tf.keras.Input(shape=(*cfg['input_size'], 3))
    base = cfg['class'](weights='imagenet', include_top=False,
                        pooling=cfg['pooling'], input_tensor=inp)
    base.trainable = False
    x = base.output
    x = Dense(EMBED_DIM, activation='relu', name=EMBEDDING_LAYER_NAME)(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)

    full_model = Model(inp, out)

    h5_path = os.path.join(FEAT_DIR, EXPERIMENT, cfg['h5_file'])
    if os.path.exists(h5_path):
        full_model.load_weights(h5_path)
        print(f"  ✅ Weights loaded: {cfg['h5_file']}")
    else:
        print(f"  ⚠️ WARNING: {h5_path} tidak ditemukan!")

    extractor = Model(full_model.input,
                      full_model.get_layer(EMBEDDING_LAYER_NAME).output)
    return extractor


def extract_features(extractor, image_paths, input_size, preprocess_fn, desc='Extracting'):
    """Ekstrak fitur menggunakan tf.data.Dataset (C++) — sangat cepat."""
    def parse_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, input_size)
        img = preprocess_fn(img)
        return img

    valid_paths = []
    valid_idx = []
    for i, path in enumerate(image_paths):
        if os.path.exists(path):
            valid_paths.append(path)
            valid_idx.append(i)

    if not valid_paths:
        return np.array([]), []

    ds = tf.data.Dataset.from_tensor_slices(valid_paths)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    features = extractor.predict(ds, verbose=1)

    # L2 Normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    features = features / norms

    return features, valid_idx


# ── Tentukan n_classes (harus sama dengan saat training) ─────────
df_train = df_full[df_full['split'] == 'train'].copy()
n_classes = df_train['category'].nunique()
print(f"\n📊 Jumlah kelas untuk arsitektur model: {n_classes}")

# ── Ekstraksi Fitur Query & Gallery ──────────────────────────────
RETRIEVAL_FEAT_DIR = os.path.join(FEAT_DIR, f'retrieval_{EXPERIMENT}')
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

    if os.path.exists(q_feat_path) and os.path.exists(g_feat_path):
        print(f"  ⏩ Fitur sudah ada, langsung ke evaluasi!")
    else:
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
        print(f"  💾 Tersimpan di: {RETRIEVAL_FEAT_DIR}")

        del extractor, q_feats, g_feats
        gc.collect()
        tf.keras.backend.clear_session()

print("\n🎉 Ekstraksi fitur selesai!")

# ── Evaluasi Retrieval ──────────────────────────────────────────
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
            top_k_ids = gallery_item_ids[sorted_indices[:k]]
            if query_id in top_k_ids:
                recall_at_k[k] += 1

        ap, n_correct = 0.0, 0
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


# ── Jalankan Evaluasi ──────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  📊 EVALUASI RETRIEVAL — {EXPERIMENT.upper()}")
print("=" * 70)

all_results = {}
for model_name in MODEL_CONFIGS.keys():
    print(f"\n🧠 Evaluating: {model_name.upper()}")

    q_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_features.npy'))
    g_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_features.npy'))
    q_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_valid_idx.npy'))
    g_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_valid_idx.npy'))

    q_item_ids = df_query.iloc[q_valid]['item_id'].values
    g_item_ids = df_gallery.iloc[g_valid]['item_id'].values

    results = evaluate_retrieval(q_feats, g_feats, q_item_ids, g_item_ids)
    all_results[model_name] = results

    print(f"\n  {'='*40}")
    print(f"  📊 HASIL {model_name.upper()} ({EXPERIMENT.upper()})")
    print(f"  {'='*40}")
    print(f"  Hit Rate @ 1 : {results['Recall@1']*100:.2f}%")
    print(f"  Hit Rate @ 5 : {results['Recall@5']*100:.2f}%")
    print(f"  Hit Rate @ 10: {results['Recall@10']*100:.2f}%")
    print(f"  Hit Rate @ 20: {results['Recall@20']*100:.2f}%")
    print(f"  mAP          : {results['mAP']*100:.2f}%")
    print(f"  Queries eval : {results['n_queries_evaluated']}")

# ── Tabel Perbandingan ──────────────────────────────────────────
print("\n" + "=" * 80)
print(f"  📊 TABEL HASIL — {EXPERIMENT.upper()}")
print("=" * 80)
print(f"{'Model':<15} {'Hit@1':>10} {'Hit@5':>10} {'Hit@10':>11} {'Hit@20':>11} {'mAP':>10}")
print("-" * 80)

for model_name, results in all_results.items():
    print(f"{model_name:<15} "
          f"{results['Recall@1']*100:>9.2f}% "
          f"{results['Recall@5']*100:>9.2f}% "
          f"{results['Recall@10']*100:>10.2f}% "
          f"{results['Recall@20']*100:>10.2f}% "
          f"{results['mAP']*100:>9.2f}%")

print("-" * 80)

# ── Simpan ke CSV ──────────────────────────────────────────
df_results = pd.DataFrame(all_results).T
df_results = df_results.drop(columns=['n_queries_evaluated'])
df_results = df_results.rename(columns={
    'Recall@1': 'Hit@1', 'Recall@5': 'Hit@5',
    'Recall@10': 'Hit@10', 'Recall@20': 'Hit@20'
})

csv_out = os.path.join(RETRIEVAL_FEAT_DIR, f'cnn_hit_miss_results_{EXPERIMENT}.csv')
df_results.to_csv(csv_out)
print(f"\n💾 Hasil disimpan ke: {csv_out}")

# ── Visualisasi Kualitatif — Query vs Top-K Gallery ──────────────
def visualize_retrieval(model_name, df_query, df_gallery,
                        query_features, gallery_features,
                        query_valid_idx, gallery_valid_idx,
                        n_samples=5, top_k=5, seed=42):
    """
    Tampilkan n_samples query secara acak beserta top_k gallery results.
    Bingkai HIJAU = item_id sama (HIT), bingkai MERAH = item_id beda (MISS)
    """
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

    fig.suptitle(f'Retrieval Results — {model_name.upper()} ({EXPERIMENT.upper()})\n'
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
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')

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
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')

            status = "✅ HIT" if is_hit else "❌ MISS"
            color = 'green' if is_hit else 'red'
            ax.set_title(f"#{col+1} {status}\n{gallery_id}\nsim={sim_score:.3f}",
                         fontsize=7, color=color, fontweight='bold')
            ax.axis('off')

            rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                      linewidth=4, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(os.path.join(RETRIEVAL_FEAT_DIR,
                f'retrieval_visual_{model_name}_{EXPERIMENT}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  🖼️  Gambar retrieval visual tersimpan!")


# ── Visualisasi untuk Setiap Model ───────────────────────────────
N_SAMPLES = 5
TOP_K = 5
SEED = 42

for model_name in MODEL_CONFIGS.keys():
    print(f"\n{'='*55}")
    print(f"  🖼️  Visualisasi: {model_name.upper()}")
    print(f"{'='*55}")

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


# ── Analisis Per Kategori ────────────────────────────────────────
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


for model_name in MODEL_CONFIGS.keys():
    print(f"\n📊 Recall@5 per Kategori — {model_name.upper()}")
    df_cat = per_category_recall(model_name, k=5)
    print(df_cat.to_string(index=False))

    csv_cat_out = os.path.join(RETRIEVAL_FEAT_DIR, f'recall_per_category_{model_name}_{EXPERIMENT}.csv')
    df_cat.to_csv(csv_cat_out, index=False)

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
    plt.close()

print("\n✅ Analisis per kategori selesai!")

print("\n" + "=" * 60)
print("  🎉 EVALUASI RETRIEVAL EXP3e SELESAI!")
print("=" * 60)
