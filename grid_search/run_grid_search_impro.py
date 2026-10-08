"""
=============================================================
  VISUAL-BASED FASHION RECOMMENDER SYSTEM
  Grid Search — Hyperparameter Optimization (Overnight)
  Environment: Local WSL + RTX 4060 Ti
=============================================================
  Terisolasi dari workspace utama.
  Semua output (CSV, .h5, .npy) tersimpan di folder grid_search/

  Parameter yang di-search:
    - OPTIMIZER:       [Adam, AdamW(wd=1e-4)]
    - BATCH_SIZE:      [24, 32]
    - PHASE2_LR:       [5e-6, 1e-5]
    - UNFREEZE_BLOCKS: [30, 50]
  Total: 16 kombinasi (~8-9 jam)

  Fitur keamanan:
    - CSV disimpan setelah SETIAP run (crash-safe)
    - Skip otomatis jika fitur sudah ada (resume-safe)
    - Evaluasi retrieval dijalankan ulang jika fitur sudah ada
    - Model terbaik auto-saved
    - Memory cleanup setiap run
    - Master dataset features diekstrak untuk model terbaik

  Output akhir:
    - grid_search_results.csv
    - grid_search_summary_barchart.png
    - hitrate_at_k_comparison.png
    - hitrate_per_category_{run_name}.csv + .png  (per run)
    - retrieval_visual_{run_name}.png              (per run)
=============================================================
"""

import os, time, gc, platform, itertools, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU: {gpus[0].name}")

# ── Path ──────────────────────────────────────────────
if platform.system() == 'Linux' and os.path.exists('/mnt/d'):
    BASE_DIR = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
else:
    BASE_DIR = r'D:\Antigravity\Visual Based Rekomender Sistem'

GRID_BASE    = os.path.join(BASE_DIR, 'grid_search')
RESULTS_DIR  = os.path.join(GRID_BASE, 'results')
MODELS_DIR   = os.path.join(GRID_BASE, 'models')
FEATURES_DIR = os.path.join(GRID_BASE, 'features')
FULL_CSV     = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')
MASTER_CSV   = os.path.join(BASE_DIR, 'dataset', 'master_dataset.csv')

if platform.system() == 'Linux' and os.path.exists('/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'):
    IMG_DIR = '/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'
else:
    IMG_DIR = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')

for d in [RESULTS_DIR, MODELS_DIR, FEATURES_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# HYPERPARAMETER GRID
# ─────────────────────────────────────────────
GRID = {
    'optimizer':       ['adam', 'adamw'],
    'batch_size':      [24, 32],
    'phase2_lr':       [5e-6, 1e-5],
    'unfreeze_blocks': [30, 50],
}
ADAMW_WEIGHT_DECAY = 1e-4

# Parameter TETAP
PHASE1_EPOCHS = 5
PHASE1_LR     = 1e-4
PHASE2_EPOCHS = 10
PATIENCE_P1   = 3
PATIENCE_P2   = 4
EMBED_DIM     = 512
DROPOUT       = 0.4

# ── Model Configs ─────────────────────────────────────────────────
MODEL_CONFIGS = {
    'resnet50': {
        'class'     : ResNet50,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
    },
    'vgg19': {
        'class'     : VGG19,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.vgg19.preprocess_input,
    },
    'inceptionv3': {
        'class'     : InceptionV3,
        'input_size': (299, 299),
        'preprocess': tf.keras.applications.inception_v3.preprocess_input,
    },
    'mobilenetv3': {
        'class'     : MobileNetV3Large,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
    },
}

keys   = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
runs_per_model = len(combos)
total_runs     = runs_per_model * len(MODEL_CONFIGS)

print(f"\n{'='*70}")
print(f"  🔍 GRID SEARCH CONFIGURATION")
print(f"{'='*70}")
for k, v in GRID.items():
    print(f"  {k:20s}: {v}")
print(f"  {'─'*50}")
print(f"  Models             : {list(MODEL_CONFIGS.keys())}")
print(f"  AdamW weight_decay : {ADAMW_WEIGHT_DECAY}")
print(f"  Kombinasi per model: {runs_per_model}")
print(f"  Total runs (semua) : {total_runs}")
print(f"  Estimasi waktu     : ~{total_runs * 35} menit ({total_runs * 35 / 60:.1f} jam)")
print(f"  Output dir         : {GRID_BASE}")


# ─────────────────────────────────────────────
# LOAD DATASET (Sekali saja — shared antar run)
# ─────────────────────────────────────────────
print("\n📦 Memuat dataset...")
df_full = pd.read_csv(FULL_CSV)

def make_path(p):
    p_clean = str(p).replace('\\', '/').replace('img/', '', 1)
    return os.path.join(IMG_DIR, p_clean)

df_full['full_path'] = df_full['image_name'].apply(
    lambda x: os.path.join(IMG_DIR, os.path.normpath(str(x).replace('img/', '', 1)))
)

# ── Split dataset ─────────────────────────────────────────────────
df_train_all = df_full[df_full['split'] == 'train'].copy()
df_train_all['full_path'] = df_train_all['image_name'].apply(make_path)
mask = df_train_all['full_path'].apply(os.path.exists)
df_train_all = df_train_all[mask].reset_index(drop=True)

n_classes  = df_train_all['category'].nunique()
cat_to_idx = {c: i for i, c in enumerate(sorted(df_train_all['category'].unique()))}
df_train_all['label'] = df_train_all['category'].map(cat_to_idx)

df_t, df_v = train_test_split(
    df_train_all, test_size=0.2, random_state=42, stratify=df_train_all['label']
)
df_t = df_t.reset_index(drop=True)
df_v = df_v.reset_index(drop=True)

df_query   = df_full[df_full['split'] == 'query'].reset_index(drop=True)
df_gallery = df_full[df_full['split'] == 'gallery'].reset_index(drop=True)

# Front = master_dataset (data utama untuk Streamlit app)
if os.path.exists(MASTER_CSV):
    df_front = pd.read_csv(MASTER_CSV)
    df_front['full_path'] = df_front['image_name'].apply(make_path)
else:
    df_front = pd.DataFrame()

query_paths   = df_query['full_path'].tolist()
gallery_paths = df_gallery['full_path'].tolist()
front_paths   = df_front['full_path'].tolist() if len(df_front) > 0 else []

print(f"📊 Train: {len(df_t):,} | Val: {len(df_v):,} | Kelas: {n_classes}")
print(f"📊 Query: {len(df_query):,} | Gallery: {len(df_gallery):,} | Front (master): {len(df_front):,}")
if len(df_front) == 0:
    print("  ⚠️  master_dataset.csv tidak ditemukan — ekstraksi front dilewati.")


# ─────────────────────────────────────────────
# AUGMENTASI LAYER
# ─────────────────────────────────────────────
augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(20/360),
    tf.keras.layers.RandomZoom((-0.25, 0.0)),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(factor=0.2),
], name='augmentation')

# INPUT_SIZE dan preprocess_fn di-set di dalam model loop
INPUT_SIZE    = None
preprocess_fn = None
MODEL_CFG     = None


# ─────────────────────────────────────────────
# FUNGSI-FUNGSI UTAMA
# ─────────────────────────────────────────────
def make_datasets(batch_size):
    def parse_aug(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, INPUT_SIZE)
        img = augmentation_layer(img, training=True)
        img = preprocess_fn(img)
        return img, label

    def parse_val(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, INPUT_SIZE)
        img = preprocess_fn(img)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices(
        (df_t['full_path'].values, df_t['label'].values)
    )
    train_ds = train_ds.shuffle(10000).map(parse_aug, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (df_v['full_path'].values, df_v['label'].values)
    )
    val_ds = val_ds.map(parse_val, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model_and_train(optimizer_name, batch_size, phase2_lr, unfreeze_blocks):
    """Build, train (Phase 1 + Phase 2), return model + base + training info."""
    train_ds, val_ds = make_datasets(batch_size)

    inp  = tf.keras.Input(shape=(*INPUT_SIZE, 3))
    base = MODEL_CFG['class'](weights='imagenet', include_top=False, pooling='avg', input_tensor=inp)
    base.trainable = False
    x   = base.output
    x   = Dense(EMBED_DIM, activation='relu', name='embedding')(x)
    x   = Dropout(DROPOUT)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out)

    # ── Phase 1: hanya head ──────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE1_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cb1 = [EarlyStopping(monitor='val_loss', patience=PATIENCE_P1,
                         restore_best_weights=True, verbose=0)]
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=PHASE1_EPOCHS, callbacks=cb1, verbose=1)
    p1_best_acc    = max(h1.history.get('val_accuracy', [0]))
    p1_stopped_ep  = len(h1.history['loss'])

    # ── Phase 2: unfreeze + fine-tune ────────
    for layer in base.layers[-unfreeze_blocks:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    if optimizer_name == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=phase2_lr)
    elif optimizer_name == 'adamw':
        opt = tf.keras.optimizers.AdamW(
            learning_rate=phase2_lr,
            weight_decay=ADAMW_WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb2 = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE_P2,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=0),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=PHASE2_EPOCHS, callbacks=cb2, verbose=1)
    p2_best_acc   = max(h2.history.get('val_accuracy', [0]))
    p2_best_ep    = h2.history['val_loss'].index(min(h2.history['val_loss'])) + 1
    p2_stopped_ep = len(h2.history['loss'])

    info = {
        'p1_val_acc'       : round(p1_best_acc, 4),
        'p1_stopped_epoch' : p1_stopped_ep,
        'p2_val_acc'       : round(p2_best_acc, 4),
        'p2_best_epoch'    : p2_best_ep,
        'p2_stopped_epoch' : p2_stopped_ep,
    }
    return model, base, info


def extract_features_fast(extractor, image_paths, batch_size):
    """Ekstrak fitur, normalisasi L2, return (features, valid_indices)."""
    valid_paths, valid_idx = [], []
    for i, p in enumerate(image_paths):
        if os.path.exists(p):
            valid_paths.append(p)
            valid_idx.append(i)
    if not valid_paths:
        return np.array([]), np.array([], dtype=int)

    ds = tf.data.Dataset.from_tensor_slices(valid_paths)
    ds = ds.map(
        lambda p: preprocess_fn(
            tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(p), channels=3), INPUT_SIZE)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    feats = extractor.predict(ds, verbose=0)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.where(norms == 0, 1e-10, norms)
    return feats, np.array(valid_idx, dtype=int)


def evaluate_retrieval(q_feats, g_feats, q_ids, g_ids, k_values=[1, 5, 10, 20]):
    """Hitung HitRate@K dan mAP."""
    sim   = q_feats @ g_feats.T
    hits  = {k: 0 for k in k_values}
    aps   = []

    for i in range(len(q_feats)):
        gt = (g_ids == q_ids[i])
        if gt.sum() == 0:
            continue
        ranked = np.argsort(-sim[i])
        for k in k_values:
            if q_ids[i] in g_ids[ranked[:k]]:
                hits[k] += 1
        ap, nc = 0.0, 0
        for r, idx in enumerate(ranked):
            if gt[idx]:
                nc += 1
                ap += nc / (r + 1)
        aps.append(ap / gt.sum())

    n = len(aps)
    return (
        {f'HitRate@{k}': hits[k] / n if n > 0 else 0 for k in k_values}
        | {'mAP': float(np.mean(aps)) if aps else 0}
    )


# ─────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────
def visualize_retrieval(run_name, q_feats, g_feats, q_valid, g_valid, out_dir,
                        n_samples=5, top_k=5):
    """Visualisasi hasil retrieval: query → top-K gallery."""
    np.random.seed(42)
    q_ids      = df_query.iloc[q_valid]['item_id'].values
    g_ids      = df_gallery.iloc[g_valid]['item_id'].values
    sim_matrix = q_feats @ g_feats.T

    sample_indices = np.random.choice(
        len(q_feats), size=min(n_samples, len(q_feats)), replace=False
    )
    fig, axes = plt.subplots(n_samples, top_k + 1, figsize=(3 * (top_k + 1), 3.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'Retrieval Results — {run_name}\n🟢 HIT (item sama)  🔴 MISS (item beda)',
        fontsize=14, fontweight='bold', y=1.02
    )

    for row, q_idx in enumerate(sample_indices):
        query_id  = q_ids[q_idx]
        query_row = df_query.iloc[q_valid[q_idx]]

        ax = axes[row, 0]
        try:
            ax.imshow(Image.open(query_row['full_path']).convert('RGB'))
        except Exception:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.set_title(
            f"QUERY\n{query_id}\n{query_row.get('category', '?')}",
            fontsize=8, fontweight='bold', color='blue'
        )
        ax.axis('off')
        ax.add_patch(patches.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            linewidth=4, edgecolor='blue', facecolor='none'
        ))

        ranked = np.argsort(-sim_matrix[q_idx])
        for col, g_rank_idx in enumerate(ranked[:top_k]):
            ax          = axes[row, col + 1]
            gallery_row = df_gallery.iloc[g_valid[g_rank_idx]]
            gallery_id  = g_ids[g_rank_idx]
            sim_score   = sim_matrix[q_idx, g_rank_idx]
            is_hit      = (gallery_id == query_id)

            try:
                ax.imshow(Image.open(gallery_row['full_path']).convert('RGB'))
            except Exception:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')

            color  = 'green' if is_hit else 'red'
            status = "✅ HIT" if is_hit else "❌ MISS"
            ax.set_title(
                f"#{col+1} {status}\n{gallery_id}\nsim={sim_score:.3f}",
                fontsize=7, color=color, fontweight='bold'
            )
            ax.axis('off')
            ax.add_patch(patches.Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                linewidth=4, edgecolor=color, facecolor='none'
            ))

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f'retrieval_visual_{run_name}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def per_category_hitrate(run_name, q_feats, g_feats, q_valid, g_valid, out_dir, k=5):
    """
    Hitung HitRate@K per kategori, simpan CSV + bar chart.
    Terminologi: HitRate@K (bukan recall).
    """
    q_info     = df_query.iloc[q_valid][['item_id', 'category']].values
    g_ids      = df_gallery.iloc[g_valid]['item_id'].values
    sim_matrix = q_feats @ g_feats.T

    cat_total = {}
    cat_hits  = {}

    for i in range(len(q_feats)):
        qid, qcat = q_info[i, 0], q_info[i, 1]
        if qid not in g_ids:
            continue
        cat_total.setdefault(qcat, 0)
        cat_hits.setdefault(qcat, 0)
        cat_total[qcat] += 1

        ranked_ids = g_ids[np.argsort(-sim_matrix[i])[:k]]
        if qid in ranked_ids:
            cat_hits[qcat] += 1

    results = [
        {
            'category'     : cat,
            f'HitRate@{k}' : cat_hits[cat] / cat_total[cat] if cat_total[cat] > 0 else 0,
            'hits'         : cat_hits[cat],
            'total'        : cat_total[cat],
        }
        for cat in sorted(cat_total.keys())
    ]
    df_cat = pd.DataFrame(results).sort_values(f'HitRate@{k}', ascending=False)

    # ── CSV ──
    df_cat.to_csv(
        os.path.join(out_dir, f'hitrate_per_category_{run_name}.csv'),
        index=False
    )

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlGn(df_cat[f'HitRate@{k}'].values)
    bars   = ax.barh(df_cat['category'], df_cat[f'HitRate@{k}'], color=colors)
    ax.set_xlabel(f'HitRate@{k}', fontsize=12)
    ax.set_title(f'HitRate@{k} per Kategori — {run_name}', fontweight='bold', fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, df_cat[f'HitRate@{k}'].values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2%}', va='center', fontsize=8
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f'hitrate_per_category_{run_name}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()


# ─────────────────────────────────────────────
# FUNGSI OUTPUT AKHIR
# ─────────────────────────────────────────────
def plot_summary_barchart(df_results, out_dir):
    """
    Bar chart grouped: perbandingan semua run pada HitRate@1/5/10/20 + mAP.
    Diurutkan berdasarkan HitRate@5.
    """
    metric_cols = ['HitRate@1', 'HitRate@5', 'HitRate@10', 'HitRate@20', 'mAP']
    colors      = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    df_sorted   = df_results.sort_values('HitRate@5', ascending=False).reset_index(drop=True)
    n_runs      = len(df_sorted)
    x           = np.arange(n_runs)
    width       = 0.15

    fig, ax = plt.subplots(figsize=(max(16, n_runs * 1.8), 7))
    for i, (col, color) in enumerate(zip(metric_cols, colors)):
        ax.bar(
            x + i * width,
            df_sorted[col].values * 100,
            width,
            label=col,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5,
        )

    ax.set_xlabel('Konfigurasi Run', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(
        'Perbandingan Semua Run — Grid Search\n(Diurutkan berdasarkan HitRate@5)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_sorted['run_name'].values, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    # Annotate top bar per group
    for i_run, row in df_sorted.iterrows():
        best_val = max(row[col] for col in metric_cols) * 100
        ax.text(
            i_run + width * 2, best_val + 1.5,
            f"{best_val:.1f}%", ha='center', fontsize=7, color='black', alpha=0.7
        )

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'grid_search_summary_barchart.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Bar chart summary → {out_path}")


def plot_hitrate_at_k(df_results, out_dir, top_n=8):
    """
    Line chart HitRate@K (K = 1, 5, 10, 20) untuk top-N run.
    Setiap run = satu garis.
    """
    k_values    = [1, 5, 10, 20]
    metric_cols = [f'HitRate@{k}' for k in k_values]
    df_sorted   = df_results.sort_values('HitRate@5', ascending=False).reset_index(drop=True)
    n_show      = min(top_n, len(df_sorted))
    cmap        = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(df_sorted.head(n_show).iterrows()):
        scores = [row[col] * 100 for col in metric_cols]
        color  = cmap(i / n_show)
        ax.plot(k_values, scores, marker='o', linewidth=2, markersize=7,
                label=row['run_name'], color=color)
        ax.annotate(
            f"{scores[-1]:.1f}%",
            xy=(20, scores[-1]),
            xytext=(6, 0), textcoords='offset points',
            va='center', fontsize=7, color=color
        )

    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('HitRate@K (%)', fontsize=12)
    ax.set_title(
        f'HitRate@K — Top {n_show} Run (Grid Search)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xticks(k_values)
    ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'hitrate_at_k_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 HitRate@K chart → {out_path}")


# ─────────────────────────────────────────────
# LOAD EXISTING CSV (resume-safe)
# ─────────────────────────────────────────────
existing_csv_path = os.path.join(RESULTS_DIR, 'grid_search_results.csv')
all_results       = []
completed_runs    = set()

if os.path.exists(existing_csv_path):
    df_existing = pd.read_csv(existing_csv_path)
    all_results     = df_existing.to_dict('records')
    completed_runs  = set(df_existing['run_name'].tolist())
    print(f"\n♻️  Menemukan CSV lama — {len(completed_runs)} run sudah selesai, dilanjutkan...")
else:
    print("\n🆕 Tidak ada CSV sebelumnya — mulai dari awal.")


# ─────────────────────────────────────────────
# GRID SEARCH LOOP
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🔍 MEMULAI GRID SEARCH — JANGAN SENTUH KOMPUTER INI!")
print(f"  ⏰ Dimulai: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

global_run_idx = 0

for model_name, model_cfg in MODEL_CONFIGS.items():
    # ── Update global variables untuk model ini ──
    MODEL_CFG     = model_cfg
    INPUT_SIZE    = model_cfg['input_size']
    preprocess_fn = model_cfg['preprocess']

    print(f"\n{'#'*70}")
    print(f"  🏗️  MODEL: {model_name.upper()} (input={INPUT_SIZE})")
    print(f"{'#'*70}")

    for combo_idx, combo in enumerate(combos):
        params   = dict(zip(keys, combo))
        opt_name = params['optimizer']
        bs       = params['batch_size']
        p2lr     = params['phase2_lr']
        uf       = params['unfreeze_blocks']
        run_name = f"{model_name}_{opt_name}_bs{bs}_lr{p2lr}_uf{uf}"
        global_run_idx += 1

        print(f"\n{'='*70}")
        print(f"  🧪 RUN {global_run_idx}/{total_runs}: {run_name}")
        print(f"     model={model_name}, optimizer={opt_name}, batch={bs}, lr={p2lr}, unfreeze={uf}")
        if opt_name == 'adamw':
            print(f"     weight_decay={ADAMW_WEIGHT_DECAY}")
        print(f"{'='*70}")

        # ── Skip jika sudah ada di CSV ───────────────────
        if run_name in completed_runs:
            print(f"  ⏭️  Run sudah ada di CSV — skip sepenuhnya.")
            continue

        t_run        = time.time()
        run_feat_dir = os.path.join(FEATURES_DIR, run_name)
        os.makedirs(run_feat_dir, exist_ok=True)

        q_feat_path  = os.path.join(run_feat_dir, 'query_features.npy')
        g_feat_path  = os.path.join(run_feat_dir, 'gallery_features.npy')
        f_feat_path  = os.path.join(run_feat_dir, 'front_features.npy')
        features_exist = (
            os.path.exists(q_feat_path)
            and os.path.exists(g_feat_path)
            and (len(front_paths) == 0 or os.path.exists(f_feat_path))
        )

        # ── Jika fitur sudah ada → skip training, langsung evaluasi ──
        if features_exist:
            print(f"  ♻️  Fitur sudah ada di disk — skip training, langsung evaluasi...")
            q_feats = np.load(q_feat_path)
            g_feats = np.load(g_feat_path)
            q_valid = np.load(os.path.join(run_feat_dir, 'query_valid_idx.npy'))
            g_valid = np.load(os.path.join(run_feat_dir, 'gallery_valid_idx.npy'))
            f_feats = np.load(f_feat_path) if os.path.exists(f_feat_path) else np.array([])
            f_valid = (
                np.load(os.path.join(run_feat_dir, 'front_valid_idx.npy'))
                if os.path.exists(os.path.join(run_feat_dir, 'front_valid_idx.npy'))
                else np.array([], dtype=int)
            )
            train_info = {
                'p1_val_acc': None, 'p1_stopped_epoch': None,
                'p2_val_acc': None, 'p2_best_epoch': None, 'p2_stopped_epoch': None,
            }

        # ── Training penuh + ekstraksi fitur ──────────────────────────
        else:
            # ── Train ────────────────────────────
            model, base, train_info = build_model_and_train(opt_name, bs, p2lr, uf)

            # ── Simpan model weights ──────────────
            h5_path = os.path.join(MODELS_DIR, f'{run_name}.h5')
            model.save_weights(h5_path)
            print(f"  💾 Model weights → {h5_path}")

            # ── Ekstraksi fitur ───────────────────
            extractor = Model(model.input, model.get_layer('embedding').output)

            print(f"  🔍 Mengekstrak fitur query...")
            q_feats, q_valid = extract_features_fast(extractor, query_paths, bs)

            print(f"  🔍 Mengekstrak fitur gallery...")
            g_feats, g_valid = extract_features_fast(extractor, gallery_paths, bs)

            print(f"  🔍 Mengekstrak fitur front...")
            if front_paths:
                f_feats, f_valid = extract_features_fast(extractor, front_paths, bs)
            else:
                f_feats, f_valid = np.array([]), np.array([], dtype=int)

            # ── Simpan fitur ke subfolder per run ─
            np.save(q_feat_path, q_feats)
            np.save(g_feat_path, g_feats)
            np.save(os.path.join(run_feat_dir, 'query_valid_idx.npy'), q_valid)
            np.save(os.path.join(run_feat_dir, 'gallery_valid_idx.npy'), g_valid)
            if len(f_feats) > 0:
                np.save(f_feat_path, f_feats)
                np.save(os.path.join(run_feat_dir, 'front_valid_idx.npy'), f_valid)

            print(f"  💾 Fitur query/gallery/front → {run_feat_dir}")

            del model, base, extractor
            gc.collect()
            tf.keras.backend.clear_session()

        # ── Evaluasi retrieval ────────────────────────────
        q_ids = df_query.iloc[q_valid]['item_id'].values
        g_ids = df_gallery.iloc[g_valid]['item_id'].values
        metrics  = evaluate_retrieval(q_feats, g_feats, q_ids, g_ids)
        elapsed  = (time.time() - t_run) / 60

        # ── Visualisasi per run ───────────────────────────
        print(f"  🖼️  Membuat visualisasi retrieval untuk {run_name}...")
        visualize_retrieval(run_name, q_feats, g_feats, q_valid, g_valid, run_feat_dir)
        per_category_hitrate(run_name, q_feats, g_feats, q_valid, g_valid, run_feat_dir)

        # ── Susun hasil ───────────────────────────────────
        result = {
            'run_name'       : run_name,
            'model'          : model_name,
            'optimizer'      : opt_name,
            'weight_decay'   : ADAMW_WEIGHT_DECAY if opt_name == 'adamw' else 0,
            'batch_size'     : bs,
            'phase2_lr'      : p2lr,
            'unfreeze_blocks': uf,
            **{k: v for k, v in (train_info or {}).items()},
            **{k: round(v, 4) for k, v in metrics.items()},
            'time_min'       : round(elapsed, 1),
        }
        all_results.append(result)

        print(f"\n  📊 HASIL RUN {global_run_idx}/{total_runs}:")
        print(
            f"     HitRate@1={metrics['HitRate@1']*100:.2f}%  "
            f"HitRate@5={metrics['HitRate@5']*100:.2f}%  "
            f"HitRate@10={metrics['HitRate@10']*100:.2f}%  "
            f"HitRate@20={metrics['HitRate@20']*100:.2f}%"
        )
        print(f"     mAP={metrics['mAP']*100:.2f}%  |  val_acc={train_info.get('p2_val_acc', 'N/A')}  |  ⏱️ {elapsed:.1f} min")

        # ── Simpan CSV setiap run (crash-safe) ────────────
        df_grid = pd.DataFrame(all_results)
        df_grid.to_csv(existing_csv_path, index=False)

        # ── Track best model per arsitektur ────────────────
        model_results = [r for r in all_results if r.get('model') == model_name]
        best_so_far = max(model_results, key=lambda x: x.get('HitRate@5', 0))
        if result.get('HitRate@5', 0) >= best_so_far.get('HitRate@5', 0):
            inp_  = tf.keras.Input(shape=(*INPUT_SIZE, 3))
            base_ = MODEL_CFG['class'](weights='imagenet', include_top=False, pooling='avg', input_tensor=inp_)
            x_    = Dense(EMBED_DIM, activation='relu', name='embedding')(base_.output)
            x_    = Dropout(DROPOUT)(x_)
            out_  = Dense(n_classes, activation='softmax')(x_)
            tmp_model = Model(inp_, out_)
            h5_path   = os.path.join(MODELS_DIR, f'{run_name}.h5')
            if os.path.exists(h5_path):
                tmp_model.load_weights(h5_path)
                tmp_model.save_weights(os.path.join(MODELS_DIR, f'best_{model_name}.h5'))
                del tmp_model
                gc.collect()
                tf.keras.backend.clear_session()

            with open(os.path.join(RESULTS_DIR, f'best_params_{model_name}.json'), 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  🏆 BEST {model_name.upper()} sejauh ini! HitRate@5={result['HitRate@5']*100:.2f}%")

        # Cleanup
        del q_feats, g_feats, f_feats
        gc.collect()


# ─────────────────────────────────────────────
# EKSTRAKSI FITUR MASTER DATASET — UNTUK SETIAP MODEL TERBAIK
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  📦 Mengekstrak fitur MASTER DATASET untuk setiap model terbaik...")
print("=" * 70)

df_master = pd.read_csv(MASTER_CSV) if os.path.exists(MASTER_CSV) else pd.DataFrame()
if len(df_master) > 0:
    df_master['full_path'] = df_master['image_name'].apply(make_path)
    master_paths = df_master['full_path'].tolist()

    for m_name, m_cfg in MODEL_CONFIGS.items():
        bp_path = os.path.join(RESULTS_DIR, f'best_params_{m_name}.json')
        bm_path = os.path.join(MODELS_DIR, f'best_{m_name}.h5')

        if not os.path.exists(bp_path) or not os.path.exists(bm_path):
            print(f"  ⚠️  {m_name}: best model/params tidak ditemukan, skip.")
            continue

        with open(bp_path, 'r') as f:
            bp = json.load(f)
        print(f"\n  🏆 {m_name.upper()}: {bp['run_name']} (HitRate@5={float(bp['HitRate@5'])*100:.2f}%)")

        m_input = m_cfg['input_size']
        m_preprocess = m_cfg['preprocess']

        inp_  = tf.keras.Input(shape=(*m_input, 3))
        base_ = m_cfg['class'](weights='imagenet', include_top=False, pooling='avg', input_tensor=inp_)
        base_.trainable = False
        x_    = Dense(EMBED_DIM, activation='relu', name='embedding')(base_.output)
        x_    = Dropout(DROPOUT)(x_)
        out_  = Dense(n_classes, activation='softmax')(x_)
        best_model = Model(inp_, out_)
        best_model.load_weights(bm_path)

        extractor = Model(best_model.input, best_model.get_layer('embedding').output)

        # Perlu set global untuk extract_features_fast
        INPUT_SIZE    = m_input
        preprocess_fn = m_preprocess

        m_feats, m_valid = extract_features_fast(extractor, master_paths, 64)

        np.save(os.path.join(FEATURES_DIR, f'best_{m_name}_master_features.npy'), m_feats)
        np.save(os.path.join(FEATURES_DIR, f'best_{m_name}_master_valid_idx.npy'), m_valid)
        print(f"  ✅ {m_name} master features: {m_feats.shape}")

        del best_model, extractor, m_feats
        gc.collect()
        tf.keras.backend.clear_session()
else:
    print("  ⚠️  master_dataset.csv tidak ditemukan, skip ekstraksi master.")


# ─────────────────────────────────────────────
# TABEL RINGKASAN + OUTPUT AKHIR
# ─────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"  📊 GRID SEARCH SELESAI — TABEL RINGKASAN")
print(f"  ⏰ Selesai: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 110)

df_final = pd.DataFrame(all_results).sort_values('HitRate@5', ascending=False)

display_cols = [
    'run_name', 'model', 'optimizer', 'weight_decay', 'batch_size', 'phase2_lr',
    'unfreeze_blocks', 'HitRate@1', 'HitRate@5', 'HitRate@10', 'HitRate@20', 'mAP',
    'p2_val_acc', 'p2_best_epoch', 'time_min',
]
print(df_final[[c for c in display_cols if c in df_final.columns]].to_string(index=False))

# Tampilkan best per model
print(f"\n{'='*70}")
print("  🏆 BEST PER MODEL:")
for m_name in MODEL_CONFIGS.keys():
    m_rows = df_final[df_final.get('model', df_final['run_name'].str.split('_').str[0]) == m_name]
    if len(m_rows) > 0:
        best = m_rows.iloc[0]
        print(f"     {m_name.upper():15s} → HitRate@5={best['HitRate@5']*100:.2f}%  mAP={best['mAP']*100:.2f}%  ({best['run_name']})")

total_time = sum(r.get('time_min', 0) or 0 for r in all_results)
print(f"\n  ⏱️  Total waktu : {total_time:.0f} menit ({total_time/60:.1f} jam)")
print(f"  💾 Hasil CSV   : {existing_csv_path}")

# ── Buat visualisasi global ───────────────────
print(f"\n  🖼️  Membuat visualisasi summary...")
plot_summary_barchart(df_final, RESULTS_DIR)
plot_hitrate_at_k(df_final, RESULTS_DIR)

print("\n" + "=" * 70)
print("  🎉 GRID SEARCH SELESAI! Selamat pagi! ☀️")
print("=" * 70)
