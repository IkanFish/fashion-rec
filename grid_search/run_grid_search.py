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
    - Model terbaik auto-saved
    - Memory cleanup setiap run
    - Master dataset features diekstrak untuk model terbaik
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

# Semua output grid search terisolasi di sini
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
    'optimizer':       ['adam', 'adamw'],      # AdamW = Adam + weight_decay
    'batch_size':      [24, 32],
    'phase2_lr':       [5e-6, 1e-5],
    'unfreeze_blocks': [30, 50],
}
# AdamW menggunakan weight_decay = 1e-4 (standar dari paper)
ADAMW_WEIGHT_DECAY = 1e-4

# Parameter TETAP (tidak di-search — identik Exp3 asli)
PHASE1_EPOCHS = 5
PHASE1_LR     = 1e-4
PHASE2_EPOCHS = 10
PATIENCE_P1   = 3
PATIENCE_P2   = 4
EMBED_DIM     = 512
DROPOUT       = 0.4

# ── Model Configs ─────────────────────────────────────────────────
# Grid search malam ini: ResNet50 saja.
# Setelah menemukan hyperparameter terbaik, uncomment model lain
# untuk menerapkan konfigurasi yang sama ke semua arsitektur.
MODEL_CONFIGS = {
    'resnet50': {
        'class'     : ResNet50,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
    },
    # 'vgg19': {
    #     'class'     : VGG19,
    #     'input_size': (256, 256),
    #     'preprocess': tf.keras.applications.vgg19.preprocess_input,
    # },
    # 'inceptionv3': {
    #     'class'     : InceptionV3,
    #     'input_size': (299, 299),
    #     'preprocess': tf.keras.applications.inception_v3.preprocess_input,
    # },
    # 'mobilenetv3': {
    #     'class'     : MobileNetV3Large,
    #     'input_size': (256, 256),
    #     'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
    # },
}

# Model yang akan di-grid search malam ini
ACTIVE_MODEL = 'resnet50'
MODEL_CFG    = MODEL_CONFIGS[ACTIVE_MODEL]

# Generate semua kombinasi
keys = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
total_runs = len(combos)

print(f"\n{'='*70}")
print(f"  🔍 GRID SEARCH CONFIGURATION")
print(f"{'='*70}")
for k, v in GRID.items():
    print(f"  {k:20s}: {v}")
print(f"  {'─'*50}")
print(f"  AdamW weight_decay : {ADAMW_WEIGHT_DECAY}")
print(f"  Total kombinasi    : {total_runs}")
print(f"  Estimasi waktu     : ~{total_runs * 35} menit ({total_runs * 35 / 60:.1f} jam)")
print(f"  Output dir         : {GRID_BASE}")


# ─────────────────────────────────────────────
# LOAD DATASET (Sekali saja — shared antar run)
# ─────────────────────────────────────────────
print("\n📦 Memuat dataset...")
df_full  = pd.read_csv(FULL_CSV)
df_train = df_full[df_full['split'] == 'train'].copy()

def make_path(p):
    p_clean = str(p).replace('\\', '/').replace('img/', '', 1)
    return os.path.join(IMG_DIR, p_clean)

df_train['full_path'] = df_train['image_name'].apply(make_path)
mask = df_train['full_path'].apply(os.path.exists)
df_train = df_train[mask].reset_index(drop=True)

n_classes = df_train['category'].nunique()
cat_to_idx = {c: i for i, c in enumerate(sorted(df_train['category'].unique()))}
df_train['label'] = df_train['category'].map(cat_to_idx)

df_t, df_v = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])
df_t = df_t.reset_index(drop=True)
df_v = df_v.reset_index(drop=True)

df_full['full_path'] = df_full['image_name'].apply(
    lambda x: os.path.join(IMG_DIR, os.path.normpath(str(x).replace('img/', '', 1)))
)
df_query   = df_full[df_full['split'] == 'query'].reset_index(drop=True)
df_gallery = df_full[df_full['split'] == 'gallery'].reset_index(drop=True)

print(f"📊 Train: {len(df_t):,} | Val: {len(df_v):,}")
print(f"📊 Query: {len(df_query):,} | Gallery: {len(df_gallery):,} | Kelas: {n_classes}")


# ─────────────────────────────────────────────
# AUGMENTASI LAYER (Shared antar run)
# ─────────────────────────────────────────────
augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(20/360),
    tf.keras.layers.RandomZoom((-0.25, 0.0)),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(factor=0.2),
], name='augmentation')

INPUT_SIZE    = MODEL_CFG['input_size']
preprocess_fn = MODEL_CFG['preprocess']


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

    train_ds = tf.data.Dataset.from_tensor_slices((df_t['full_path'].values, df_t['label'].values))
    train_ds = train_ds.shuffle(10000).map(parse_aug, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((df_v['full_path'].values, df_v['label'].values))
    val_ds = val_ds.map(parse_val, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model_and_train(optimizer_name, batch_size, phase2_lr, unfreeze_blocks):
    """Build, train (Phase 1 + Phase 2), return model + base + training history."""

    train_ds, val_ds = make_datasets(batch_size)

    # Build model — menggunakan arsitektur dari MODEL_CFG
    inp  = tf.keras.Input(shape=(*INPUT_SIZE, 3))
    base = MODEL_CFG['class'](weights='imagenet', include_top=False, pooling='avg', input_tensor=inp)
    base.trainable = False
    x = base.output
    x = Dense(EMBED_DIM, activation='relu', name='embedding')(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out)

    # Phase 1 optimizer (selalu Adam — hanya head)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE1_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Phase 1
    cb1 = [EarlyStopping(monitor='val_loss', patience=PATIENCE_P1, restore_best_weights=True, verbose=0)]
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS, callbacks=cb1, verbose=1)
    p1_best_acc = max(h1.history.get('val_accuracy', [0]))
    p1_stopped_epoch = len(h1.history['loss'])

    # Unfreeze
    for layer in base.layers[-unfreeze_blocks:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # Phase 2 optimizer — BERBEDA tergantung pilihan
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
        EarlyStopping(monitor='val_loss', patience=PATIENCE_P2, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=0),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS, callbacks=cb2, verbose=1)
    p2_best_acc = max(h2.history.get('val_accuracy', [0]))
    p2_best_epoch = h2.history['val_loss'].index(min(h2.history['val_loss'])) + 1
    p2_stopped_epoch = len(h2.history['loss'])

    info = {
        'p1_val_acc': round(p1_best_acc, 4),
        'p1_stopped_epoch': p1_stopped_epoch,
        'p2_val_acc': round(p2_best_acc, 4),
        'p2_best_epoch': p2_best_epoch,
        'p2_stopped_epoch': p2_stopped_epoch,
    }

    return model, base, info


def extract_features_fast(extractor, image_paths, batch_size):
    valid_paths, valid_idx = [], []
    for i, p in enumerate(image_paths):
        if os.path.exists(p):
            valid_paths.append(p)
            valid_idx.append(i)
    if not valid_paths:
        return np.array([]), np.array([])

    ds = tf.data.Dataset.from_tensor_slices(valid_paths)
    ds = ds.map(
        lambda p: preprocess_fn(tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(p), channels=3), INPUT_SIZE)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    feats = extractor.predict(ds, verbose=0)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.where(norms == 0, 1e-10, norms)
    return feats, np.array(valid_idx)


def evaluate_retrieval(q_feats, g_feats, q_ids, g_ids, k_values=[1, 5, 10, 20]):
    sim = q_feats @ g_feats.T
    recall = {k: 0 for k in k_values}
    aps = []

    for i in range(len(q_feats)):
        gt = (g_ids == q_ids[i])
        if gt.sum() == 0:
            continue
        ranked = np.argsort(-sim[i])
        for k in k_values:
            if q_ids[i] in g_ids[ranked[:k]]:
                recall[k] += 1
        ap, nc = 0.0, 0
        for r, idx in enumerate(ranked):
            if gt[idx]:
                nc += 1
                ap += nc / (r + 1)
        aps.append(ap / gt.sum())

    n = len(aps)
    return {f'Hit Rate@{k}': recall[k]/n if n > 0 else 0 for k in k_values} | {'mAP': np.mean(aps) if aps else 0}


# ─────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────
def visualize_retrieval(run_name, q_feats, g_feats, q_valid, g_valid, out_dir, n_samples=5, top_k=5):
    np.random.seed(42)
    q_ids = df_query.iloc[q_valid]['item_id'].values
    g_ids = df_gallery.iloc[g_valid]['item_id'].values
    sim_matrix = q_feats @ g_feats.T
    
    sample_indices = np.random.choice(len(q_feats), size=min(n_samples, len(q_feats)), replace=False)
    fig, axes = plt.subplots(n_samples, top_k + 1, figsize=(3 * (top_k + 1), 3.5 * n_samples))
    if n_samples == 1: axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Retrieval Results — {run_name}\n🟢 HIT (item sama)  🔴 MISS (item beda)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    for row, q_idx in enumerate(sample_indices):
        query_id = q_ids[q_idx]
        query_row = df_query.iloc[q_valid[q_idx]]
        
        ax = axes[row, 0]
        try:
            ax.imshow(Image.open(query_row['full_path']).convert('RGB'))
        except:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.set_title(f"QUERY\n{query_id}\n{query_row.get('category', '?')}", fontsize=8, fontweight='bold', color='blue')
        ax.axis('off')
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=4, edgecolor='blue', facecolor='none'))
        
        ranked = np.argsort(-sim_matrix[q_idx])
        for col, g_rank_idx in enumerate(ranked[:top_k]):
            ax = axes[row, col + 1]
            gallery_row = df_gallery.iloc[g_valid[g_rank_idx]]
            gallery_id = g_ids[g_rank_idx]
            sim_score = sim_matrix[q_idx, g_rank_idx]
            is_hit = (gallery_id == query_id)
            
            try:
                ax.imshow(Image.open(gallery_row['full_path']).convert('RGB'))
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                
            color = 'green' if is_hit else 'red'
            status = "✅ HIT" if is_hit else "❌ MISS"
            ax.set_title(f"#{col+1} {status}\n{gallery_id}\nsim={sim_score:.3f}", fontsize=7, color=color, fontweight='bold')
            ax.axis('off')
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=4, edgecolor=color, facecolor='none'))
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'retrieval_visual_{run_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def per_category_recall(run_name, q_feats, g_feats, q_valid, g_valid, out_dir, k=5):
    q_info = df_query.iloc[q_valid][['item_id', 'category']].values
    g_ids = df_gallery.iloc[g_valid]['item_id'].values
    sim_matrix = q_feats @ g_feats.T
    
    cat_hits = {}
    cat_total = {}
    
    for i in range(len(q_feats)):
        qid, qcat = q_info[i, 0], q_info[i, 1]
        if qid not in g_ids: continue
        
        if qcat not in cat_total:
            cat_total[qcat] = 0
            cat_hits[qcat] = 0
        cat_total[qcat] += 1
        
        ranked_ids = g_ids[np.argsort(-sim_matrix[i])[:k]]
        if qid in ranked_ids:
            cat_hits[qcat] += 1
            
    results = [{'category': cat, 'recall': cat_hits[cat] / cat_total[cat] if cat_total[cat] > 0 else 0,
                'hits': cat_hits[cat], 'total': cat_total[cat]} for cat in sorted(cat_total.keys())]
    df_cat = pd.DataFrame(results).sort_values('recall', ascending=False)
    df_cat.to_csv(os.path.join(out_dir, f'recall_per_category_{run_name}.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlGn(df_cat['recall'].values)
    bars = ax.barh(df_cat['category'], df_cat['recall'], color=colors)
    ax.set_xlabel(f'Hit Rate@{k}')
    ax.set_title(f'Hit Rate@{k} per Kategori — {run_name}', fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, df_cat['recall'].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2%}', va='center', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'recall_per_category_{run_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# GRID SEARCH LOOP
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  🔍 MEMULAI GRID SEARCH — JANGAN SENTUH KOMPUTER INI!")
print(f"  ⏰ Dimulai: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

all_results = []
query_paths   = df_query['full_path'].tolist()
gallery_paths = df_gallery['full_path'].tolist()

for run_idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    opt_name = params['optimizer']
    bs       = params['batch_size']
    p2lr     = params['phase2_lr']
    uf       = params['unfreeze_blocks']
    run_name = f"{ACTIVE_MODEL}_{opt_name}_bs{bs}_lr{p2lr}_uf{uf}"

    print(f"\n{'='*70}")
    print(f"  🧪 RUN {run_idx+1}/{total_runs}: {run_name}")
    print(f"     optimizer={opt_name}, batch={bs}, lr={p2lr}, unfreeze={uf}")
    if opt_name == 'adamw':
        print(f"     weight_decay={ADAMW_WEIGHT_DECAY}")
    print(f"{'='*70}")

    t_run = time.time()

    # ── Train ────────────────────────────
    model, base, train_info = build_model_and_train(opt_name, bs, p2lr, uf)

    # ── Simpan model ────────────────────────────
    h5_path = os.path.join(MODELS_DIR, f'{run_name}.h5')
    model.save_weights(h5_path)

    # ── Ekstraksi fitur query & gallery ────────────────────────────
    print(f"  🔍 Extracting query + gallery features...")
    extractor = Model(model.input, model.get_layer('embedding').output)

    q_feats, q_valid = extract_features_fast(extractor, query_paths, bs)
    g_feats, g_valid = extract_features_fast(extractor, gallery_paths, bs)
    q_ids = df_query.iloc[q_valid]['item_id'].values
    g_ids = df_gallery.iloc[g_valid]['item_id'].values

    # Simpan fitur ke subfolder per run
    run_feat_dir = os.path.join(FEATURES_DIR, run_name)
    os.makedirs(run_feat_dir, exist_ok=True)
    np.save(os.path.join(run_feat_dir, 'query_features.npy'), q_feats)
    np.save(os.path.join(run_feat_dir, 'gallery_features.npy'), g_feats)
    np.save(os.path.join(run_feat_dir, 'query_valid_idx.npy'), q_valid)
    np.save(os.path.join(run_feat_dir, 'gallery_valid_idx.npy'), g_valid)

    # ── Evaluasi ────────────────────────────
    metrics = evaluate_retrieval(q_feats, g_feats, q_ids, g_ids)
    elapsed = (time.time() - t_run) / 60

    print(f"  🖼️ Generating visualizations for {run_name}...")
    visualize_retrieval(run_name, q_feats, g_feats, q_valid, g_valid, run_feat_dir)
    per_category_recall(run_name, q_feats, g_feats, q_valid, g_valid, run_feat_dir)

    result = {
        'run_name': run_name,
        'optimizer': opt_name,
        'weight_decay': ADAMW_WEIGHT_DECAY if opt_name == 'adamw' else 0,
        'batch_size': bs,
        'phase2_lr': p2lr,
        'unfreeze_blocks': uf,
        **train_info,
        **{k: round(v, 4) for k, v in metrics.items()},
        'time_min': round(elapsed, 1),
    }
    all_results.append(result)

    print(f"\n  📊 HASIL RUN {run_idx+1}/{total_runs}:")
    print(f"     Hit Rate@1={metrics['Hit Rate@1']*100:.2f}%  Hit Rate@5={metrics['Hit Rate@5']*100:.2f}%  "
          f"Hit Rate@10={metrics['Hit Rate@10']*100:.2f}%  Hit Rate@20={metrics['Hit Rate@20']*100:.2f}%")
    print(f"     mAP={metrics['mAP']*100:.2f}%  |  val_acc={train_info['p2_val_acc']:.4f}  |  ⏱️ {elapsed:.1f} min")

    # Simpan CSV setiap run (crash-safe)
    df_grid = pd.DataFrame(all_results)
    df_grid.to_csv(os.path.join(RESULTS_DIR, 'grid_search_results.csv'), index=False)

    # Track best model (berdasarkan Hit Rate@5)
    best_so_far = max(all_results, key=lambda x: x['Hit Rate@5'])
    if result['Hit Rate@5'] == best_so_far['Hit Rate@5']:
        model.save_weights(os.path.join(MODELS_DIR, 'best_model.h5'))
        with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  🏆 MODEL TERBAIK sejauh ini! Hit Rate@5={result['Hit Rate@5']*100:.2f}%")

    # Cleanup
    del model, base, extractor, q_feats, g_feats
    gc.collect()
    tf.keras.backend.clear_session()


# ─────────────────────────────────────────────
# EKSTRAKSI FITUR MASTER DATASET — UNTUK MODEL TERBAIK
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  📦 Mengekstrak fitur MASTER DATASET untuk model terbaik...")
print("=" * 70)

with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'r') as f:
    best_params = json.load(f)

print(f"  🏆 Best: {best_params['run_name']} (Hit Rate@5={best_params['Hit Rate@5']*100:.2f}%)")

# Rebuild model dan load best weights
inp  = tf.keras.Input(shape=(*INPUT_SIZE, 3))
base = MODEL_CFG['class'](weights='imagenet', include_top=False, pooling='avg', input_tensor=inp)
base.trainable = False
x = base.output
x = Dense(EMBED_DIM, activation='relu', name='embedding')(x)
x = Dropout(DROPOUT)(x)
out = Dense(n_classes, activation='softmax')(x)
best_model = Model(inp, out)
best_model.load_weights(os.path.join(MODELS_DIR, 'best_model.h5'))

extractor = Model(best_model.input, best_model.get_layer('embedding').output)

df_master = pd.read_csv(MASTER_CSV)
df_master['full_path'] = df_master['image_name'].apply(make_path)
master_paths = df_master['full_path'].tolist()

m_feats, m_valid = extract_features_fast(extractor, master_paths, 64)

np.save(os.path.join(FEATURES_DIR, 'best_master_features.npy'), m_feats)
np.save(os.path.join(FEATURES_DIR, 'best_master_valid_idx.npy'), m_valid)

print(f"  ✅ Master features shape: {m_feats.shape}")
print(f"  💾 Tersimpan di: {FEATURES_DIR}")

del best_model, extractor
gc.collect()
tf.keras.backend.clear_session()


# ─────────────────────────────────────────────
# TABEL RINGKASAN AKHIR
# ─────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"  📊 GRID SEARCH SELESAI — TABEL RINGKASAN")
print(f"  ⏰ Selesai: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 110)

df_final = pd.DataFrame(all_results).sort_values('Hit Rate@5', ascending=False)
display_cols = ['run_name', 'optimizer', 'weight_decay', 'batch_size', 'phase2_lr',
                'unfreeze_blocks', 'Hit Rate@1', 'Hit Rate@5', 'Hit Rate@10', 'Hit Rate@20', 'mAP',
                'p2_val_acc', 'p2_best_epoch', 'time_min']
print(df_final[display_cols].to_string(index=False))

best = df_final.iloc[0]
print(f"\n{'='*70}")
print(f"  🏆 KOMBINASI TERBAIK: {best['run_name']}")
print(f"     optimizer={best['optimizer']}, weight_decay={best['weight_decay']}")
print(f"     batch_size={int(best['batch_size'])}, phase2_lr={best['phase2_lr']}, unfreeze={int(best['unfreeze_blocks'])}")
print(f"     Hit Rate@1={best['Hit Rate@1']*100:.2f}%  Hit Rate@5={best['Hit Rate@5']*100:.2f}%")
print(f"     Hit Rate@10={best['Hit Rate@10']*100:.2f}%  Hit Rate@20={best['Hit Rate@20']*100:.2f}%  mAP={best['mAP']*100:.2f}%")

total_time = sum(r['time_min'] for r in all_results)
print(f"\n  ⏱️  Total waktu: {total_time:.0f} menit ({total_time/60:.1f} jam)")
print(f"  💾 Hasil CSV   : {os.path.join(RESULTS_DIR, 'grid_search_results.csv')}")
print(f"  💾 Best model  : {os.path.join(MODELS_DIR, 'best_model.h5')}")
print(f"  💾 Best params : {os.path.join(RESULTS_DIR, 'best_params.json')}")

print("\n" + "=" * 70)
print("  🎉 GRID SEARCH SELESAI! Selamat pagi! ☀️")
print("=" * 70)
