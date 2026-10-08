"""
=============================================================
  VISUAL-BASED FASHION RECOMMENDER SYSTEM
  Notebook 04c — Eksperimen 3c: tf.data Tuning (Max GPU)
  Environment: Local WSL + RTX 4060 Ti
=============================================================
Tujuan:
  Menguji performa maksimal GPU menggunakan tf.data.Dataset.
  Semua parameter sama dengan Eksperimen 3b, hanya pipeline
  pembacaan data yang diganti agar I/O tidak menjadi bottleneck.

Output:
  - features/exp3c/{model}_features_exp3c.npy  → Feature matrix 512-D
  - features/exp3c/{model}_valid_idx_exp3c.npy → Index gambar valid
  - features/exp3c/{model}_exp3c.h5            → Bobot model terlatih

Cara menjalankan:
  1. Jalankan script ini dari WSL (Ubuntu terminal):
       python notebooks/04c_experiment3c_tfdata_tuning.py
=============================================================
"""

import os, time, gc
import numpy as np
import pandas as pd
import platform
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.model_selection import train_test_split

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPU available: {len(gpus) > 0}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")
else:
    print("⚠️  PERINGATAN: GPU tidak terdeteksi! Training akan berjalan di CPU (sangat lambat).")
    print("   Pastikan script dijalankan dari WSL, bukan dari Windows Python.")


# ─────────────────────────────────────────────
# KONFIGURASI PATH (Auto-detect WSL vs Windows)
# ─────────────────────────────────────────────
if platform.system() == 'Linux' and os.path.exists('/mnt/d'):
    # Running inside WSL (Ubuntu on Windows)
    BASE_DIR = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
else:
    # Running on Windows (Miniconda / native Python) — CPU only
    BASE_DIR = r'D:\Antigravity\Visual Based Rekomender Sistem'

FEAT_DIR   = os.path.join(BASE_DIR, 'features', 'exp3c')
MASTER_CSV = os.path.join(BASE_DIR, 'dataset', 'master_dataset.csv')
FULL_CSV   = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')

# Cek jika folder fashion_project (Native WSL) ada, gunakan itu agar sangat ngebut
if platform.system() == 'Linux' and os.path.exists('/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'):
    IMG_DIR = '/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'
    print("🚀 Menggunakan dataset native WSL untuk I/O super cepat!")
else:
    IMG_DIR = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')
    if not os.path.exists(IMG_DIR):
        IMG_DIR = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'img')

os.makedirs(FEAT_DIR, exist_ok=True)
print(f"🖥️  Running on: {platform.system()} | BASE_DIR: {BASE_DIR}")


# ─────────────────────────────────────────────
# HYPERPARAMETER — UBAH DI SINI UNTUK EKSPERIMEN
# ─────────────────────────────────────────────
# Phase 1: Latih Custom Head saja (base masih frozen)
PHASE1_EPOCHS   = 15      # [Exp3: 5]  Lebih banyak agar head konvergen dulu
PHASE1_LR       = 1e-4    # [Exp3: 1e-4] Sama

# Phase 2: Unfreeze sebagian layer base + fine-tune bersama
PHASE2_EPOCHS   = 30      # [Exp3: 10] EarlyStopping akan stop jika stagnan
PHASE2_LR       = 5e-6    # [Exp3: 1e-5] Lebih kecil untuk unfreeze lebih dalam
UNFREEZE_BLOCKS = 50      # [Exp3: 30] Buka lebih banyak layer

# Shared
PATIENCE        = 7       # [Exp3: 3-4] Lebih sabar sebelum stop
BATCH_SIZE      = 64      # Diturunkan ke 32 untuk menghemat RAM / VRAM
EMBED_DIM       = 512     # Tetap 512-D untuk kompatibilitas downstream

# ── Pilih model yang ingin dilatih ──────────────────────────────────────────
MODEL_CONFIGS = {
    'resnet50': {
        'class'     : ResNet50,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
    },
}

print(f"\n📋 Konfigurasi Eksperimen 3c (tf.data):")
print(f"   Phase 1 : {PHASE1_EPOCHS} epochs @ LR={PHASE1_LR}")
print(f"   Phase 2 : {PHASE2_EPOCHS} epochs @ LR={PHASE2_LR}")
print(f"   Patience: {PATIENCE} | Unfreeze: {UNFREEZE_BLOCKS} layers")
print(f"   Models  : {list(MODEL_CONFIGS.keys())}")
print(f"   Output  : {FEAT_DIR}")


# ─────────────────────────────────────────────
# LOAD DATASET TRAINING
# ─────────────────────────────────────────────
print("\n📦 Memuat dataset training...")
df_full  = pd.read_csv(FULL_CSV)
df_train = df_full[df_full['split'] == 'train'].copy()

def make_path(p):
    # Strips 'img/' from 'img/WOMEN/...' and joins with IMG_DIR
    p_clean = str(p).replace('\\', '/').replace('img/', '', 1)
    return os.path.join(IMG_DIR, p_clean)

df_train['full_path'] = df_train['image_name'].apply(make_path)
mask = df_train['full_path'].apply(os.path.exists)
df_train = df_train[mask].reset_index(drop=True)

print(f"📊 Total gambar training : {len(df_train):,}")
print(f"📊 Jumlah kelas kategori : {df_train['category'].nunique()}")
n_classes = df_train['category'].nunique()

cat_to_idx = {c: i for i, c in enumerate(sorted(df_train['category'].unique()))}
df_train['label'] = df_train['category'].map(cat_to_idx)

df_t, df_v = train_test_split(
    df_train, test_size=0.2, random_state=42, stratify=df_train['label']
)
df_t = df_t.reset_index(drop=True)
df_v = df_v.reset_index(drop=True)
print(f"📊 Train: {len(df_t):,} | Val: {len(df_v):,}")


# ─────────────────────────────────────────────
# IMAGE DATA GENERATORS (tf.data.Dataset)
# ─────────────────────────────────────────────
def make_generators(input_size, preprocess_fn):
    # Menggunakan tf.data.Dataset untuk utilisasi GPU 90-100%
    def parse_image(img_path, label, augment=False):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, input_size)
        
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            
        img = preprocess_fn(img)
        return img, label

    def create_dataset(df, augment=False, shuffle=False):
        paths = df['full_path'].values
        labels = df['label'].values
        
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)
            
        # AUTOTUNE menyuruh CPU mengerahkan semua core C++ secara dinamis
        ds = ds.map(lambda p, l: parse_image(p, l, augment), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = create_dataset(df_t, augment=True, shuffle=True)
    val_ds = create_dataset(df_v, augment=False, shuffle=False)
    
    return train_ds, val_ds


# ─────────────────────────────────────────────
# BUILD MODEL (2-Phase Architecture)
# ─────────────────────────────────────────────
def build_model(base_class, input_size, n_classes, lr):
    inp  = tf.keras.Input(shape=(*input_size, 3))
    base = base_class(weights='imagenet', include_top=False, pooling='avg', input_tensor=inp)
    base.trainable = False  # Phase 1: semua frozen
    x   = base.output
    x   = Dense(EMBED_DIM, activation='relu', name='embedding')(x)
    x   = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model, base


def unfreeze_top_layers(base, n_unfreeze):
    for layer in base.layers[-n_unfreeze:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    total = sum(1 for l in base.layers if l.trainable)
    print(f"   🔓 {total} / {len(base.layers)} layers base sekarang trainable")


def make_cosine_scheduler(initial_lr, total_epochs):
    import math
    def schedule(epoch, lr):
        cosine = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        return float(initial_lr * cosine)
    return LearningRateScheduler(schedule, verbose=0)


# ─────────────────────────────────────────────
# LOOP TRAINING UTAMA
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  🚀 MEMULAI EKSPERIMEN 3c — TF.DATA TUNING")
print("="*60)

for model_name, cfg in MODEL_CONFIGS.items():
    feat_out = os.path.join(FEAT_DIR, f'{model_name}_features_exp3c.npy')
    if os.path.exists(feat_out):
        print(f"\n⏭️  {model_name.upper()} sudah ada (exp3c), skip.")
        continue

    print(f"\n{'─'*60}")
    print(f"🧠 Model: {model_name.upper()}")
    t_start = time.time()

    train_flow, val_flow = make_generators(cfg['input_size'], cfg['preprocess'])

    # ── Phase 1: Latih Custom Head ───────────────────────────────
    print(f"\n  📌 PHASE 1: Custom Head ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
    model, base = build_model(cfg['class'], cfg['input_size'], n_classes, PHASE1_LR)

    cb_p1 = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    hist_p1 = model.fit(
        train_flow, validation_data=val_flow,
        epochs=PHASE1_EPOCHS, callbacks=cb_p1, verbose=1
    )
    print(f"  ✅ Phase 1 selesai. Best val_acc: {max(hist_p1.history.get('val_accuracy', [0])):.4f}")

    # ── Phase 2: Unfreeze + Fine-Tune ───────────────────────────
    print(f"\n  📌 PHASE 2: Partial Unfreeze ({UNFREEZE_BLOCKS} layers, LR={PHASE2_LR})")
    unfreeze_top_layers(base, UNFREEZE_BLOCKS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE2_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    cb_p2 = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        make_cosine_scheduler(PHASE2_LR, PHASE2_EPOCHS),
    ]

    hist_p2 = model.fit(
        train_flow, validation_data=val_flow,
        epochs=PHASE2_EPOCHS, callbacks=cb_p2, verbose=1
    )
    print(f"  ✅ Phase 2 selesai. Best val_acc: {max(hist_p2.history.get('val_accuracy', [0])):.4f}")

    # Simpan bobot model
    model_save_path = os.path.join(FEAT_DIR, f'{model_name}_exp3c.h5')
    model.save_weights(model_save_path)
    print(f"  💾 Bobot disimpan: {model_save_path}")

    # ── Ekstraksi Feature 512-D ──────────────────────────────────
    print(f"\n  🔍 Mengekstrak fitur 512-D dari embedding layer...")
    extractor  = Model(model.input, model.get_layer('embedding').output)
    df_master  = pd.read_csv(MASTER_CSV)
    df_master['full_path'] = df_master['image_name'].apply(make_path)

    features_list  = []
    valid_idx_list = []

    for idx, row in tqdm(df_master.iterrows(), total=len(df_master), desc=f"Extracting {model_name}"):
        img_path = row['full_path']
        if not os.path.exists(img_path):
            continue
        try:
            img  = Image.open(img_path).convert('RGB').resize(cfg['input_size'])
            arr  = np.array(img, dtype=np.float32)[np.newaxis]
            arr  = cfg['preprocess'](arr)
            feat = extractor.predict(arr, verbose=0)[0]
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            features_list.append(feat)
            valid_idx_list.append(idx)
        except Exception:
            pass

    feat_matrix   = np.vstack(features_list).astype(np.float32)
    valid_idx_arr = np.array(valid_idx_list)

    np.save(feat_out, feat_matrix)
    np.save(os.path.join(FEAT_DIR, f'{model_name}_valid_idx_exp3c.npy'), valid_idx_arr)

    elapsed = (time.time() - t_start) / 60
    print(f"  ✅ {model_name} selesai! Shape: {feat_matrix.shape}")
    print(f"  ⏱️  Total waktu: {elapsed:.1f} menit")

    del model, base, extractor, feat_matrix
    gc.collect()
    tf.keras.backend.clear_session()


print("\n" + "="*60)
print("  🎉 EKSPERIMEN 3c SELESAI!")
print("="*60)
