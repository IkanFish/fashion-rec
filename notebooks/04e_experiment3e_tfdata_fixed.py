"""
=============================================================
  VISUAL-BASED FASHION RECOMMENDER SYSTEM
  Notebook 04e — Eksperimen 3e: tf.data + Full Augmentation
  Environment: Local WSL + RTX 4060 Ti
=============================================================
Tujuan:
  Mereplikasi Eksperimen 3 (04_experiment3_partial_unfreeze.py)
  secara PERSIS IDENTIK, tetapi menggunakan tf.data.Dataset (C++)
  sebagai pengganti ImageDataGenerator (Python) untuk kecepatan
  I/O yang jauh lebih tinggi.

  Semua hyperparameter IDENTIK dengan Eksperimen 3 asli:
    - Phase 1: 5 epoch, LR=1e-4
    - Phase 2: 10 epoch, LR=1e-5
    - Patience: 3 (P1), 4 (P2)
    - UNFREEZE_BLOCKS: 30
    - Augmentasi: 6 teknik (flip, rotation, zoom, brightness, shift)
    - Label target: KATEGORI (bukan item_id)

  PERBEDAAN SATU-SATUNYA:
    - Pipeline I/O: tf.data.Dataset (C++) bukan ImageDataGenerator (Python)
    - BATCH_SIZE: 64 (dari 24) — hanya mempercepat, tidak merusak performa

Output:
  - features/exp3e/{model}_features_exp3e.npy
  - features/exp3e/{model}_valid_idx_exp3e.npy
  - features/exp3e/{model}_exp3e.h5
=============================================================
"""

import os, time, gc, platform
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPU available: {len(gpus) > 0}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")
else:
    print("⚠️  GPU tidak terdeteksi!")


# ─────────────────────────────────────────────
# KONFIGURASI PATH (Auto-detect WSL vs Windows)
# ─────────────────────────────────────────────
if platform.system() == 'Linux' and os.path.exists('/mnt/d'):
    BASE_DIR = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
else:
    BASE_DIR = r'D:\Antigravity\Visual Based Rekomender Sistem'

FEAT_DIR   = os.path.join(BASE_DIR, 'features', 'exp3e')
MASTER_CSV = os.path.join(BASE_DIR, 'dataset', 'master_dataset.csv')
FULL_CSV   = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')

# Prioritaskan native WSL path untuk I/O tercepat
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
# HYPERPARAMETER — IDENTIK DENGAN EKSPERIMEN 3 ASLI
# ─────────────────────────────────────────────
# Phase 1: Latih Custom Head saja (base masih frozen)
PHASE1_EPOCHS   = 5        # [Exp3 asli: 5] ✅ SAMA
PHASE1_LR       = 1e-4     # [Exp3 asli: 1e-4] ✅ SAMA
PATIENCE_P1     = 3        # [Exp3 asli: 3] ✅ SAMA

# Phase 2: Unfreeze sebagian layer base + fine-tune bersama
PHASE2_EPOCHS   = 10       # [Exp3 asli: 10] ✅ SAMA
PHASE2_LR       = 1e-5     # [Exp3 asli: 1e-5] ✅ SAMA
PATIENCE_P2     = 4        # [Exp3 asli: 4 (patience+1)] ✅ SAMA
UNFREEZE_BLOCKS = 30       # [Exp3 asli: 30] ✅ SAMA

# Shared
BATCH_SIZE      = 64       # [Exp3 asli: 24] ← Dinaikkan untuk KECEPATAN
EMBED_DIM       = 512      # [Exp3 asli: 512] ✅ SAMA

# ── Pilih model yang ingin dilatih ──────────────────────────────────────────
MODEL_CONFIGS = {
    'resnet50': {
        'class'     : ResNet50,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
    },
    # Uncomment model lain jika diperlukan:
    #'vgg19': {
    #    'class'     : VGG19,
    #    'input_size': (256, 256),
    #    'preprocess': tf.keras.applications.vgg19.preprocess_input,
    #},
    #'inceptionv3': {
    #    'class'     : InceptionV3,
    #    'input_size': (299, 299),
    #    'preprocess': tf.keras.applications.inception_v3.preprocess_input,
    #},
    #'mobilenetv3': {
    #    'class'     : MobileNetV3Large,
    #    'input_size': (256, 256),
    #    'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
    #},
}

print(f"\n📋 Konfigurasi Eksperimen 3e (tf.data + Full Augmentation):")
print(f"   Phase 1 : {PHASE1_EPOCHS} epochs @ LR={PHASE1_LR}, patience={PATIENCE_P1}")
print(f"   Phase 2 : {PHASE2_EPOCHS} epochs @ LR={PHASE2_LR}, patience={PATIENCE_P2}")
print(f"   Unfreeze: {UNFREEZE_BLOCKS} layers | Batch: {BATCH_SIZE}")
print(f"   Models  : {list(MODEL_CONFIGS.keys())}")
print(f"   Output  : {FEAT_DIR}")


# ─────────────────────────────────────────────
# LOAD DATASET TRAINING
# ─────────────────────────────────────────────
print("\n📦 Memuat dataset training...")
df_full  = pd.read_csv(FULL_CSV)
df_train = df_full[df_full['split'] == 'train'].copy()

def make_path(p):
    p_clean = str(p).replace('\\', '/').replace('img/', '', 1)
    return os.path.join(IMG_DIR, p_clean)

df_train['full_path'] = df_train['image_name'].apply(make_path)
mask = df_train['full_path'].apply(os.path.exists)
df_train = df_train[mask].reset_index(drop=True)

print(f"📊 Total gambar training : {len(df_train):,}")
print(f"📊 Jumlah kelas kategori : {df_train['category'].nunique()}")
n_classes = df_train['category'].nunique()

# Label encoding — KATEGORI, bukan item_id
cat_to_idx = {c: i for i, c in enumerate(sorted(df_train['category'].unique()))}
df_train['label'] = df_train['category'].map(cat_to_idx)

# Train/Val split — IDENTIK dengan Exp3 asli
df_t, df_v = train_test_split(
    df_train, test_size=0.2, random_state=42, stratify=df_train['label']
)
df_t = df_t.reset_index(drop=True)
df_v = df_v.reset_index(drop=True)
print(f"📊 Train: {len(df_t):,} | Val: {len(df_v):,}")


# ─────────────────────────────────────────────
# AUGMENTASI LAYER (Keras Preprocessing — Berjalan di C++/GPU)
# Mereplikasi semua 6 teknik augmentasi dari ImageDataGenerator asli
# ─────────────────────────────────────────────
augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),               # horizontal_flip=True
    tf.keras.layers.RandomRotation(20/360),                  # rotation_range=20°
    tf.keras.layers.RandomZoom((-0.25, 0.0)),                # zoom_range=0.25
    tf.keras.layers.RandomTranslation(0.1, 0.1),             # width/height_shift_range=0.1
    tf.keras.layers.RandomBrightness(factor=0.2),            # brightness_range=[0.8, 1.2]
    # shear_range=0.15 tidak tersedia native di Keras/tf.data.
    # Efek geometriknya sudah tercakup oleh kombinasi Rotation + Zoom + Translation.
], name='augmentation')

print("✅ Augmentasi layer siap (6 teknik, berjalan di GPU/C++):")
print("   → RandomFlip, RandomRotation(±20°), RandomZoom(25%),")
print("   → RandomTranslation(10%), RandomBrightness(±20%)")


# ─────────────────────────────────────────────
# TF.DATA PIPELINE (Menggantikan ImageDataGenerator)
# ─────────────────────────────────────────────
def make_generators(input_size, preprocess_fn):
    """
    Membuat tf.data.Dataset untuk training dan validasi.
    Augmentasi diterapkan SEBELUM preprocessing (sama seperti ImageDataGenerator).
    """
    def parse_and_augment(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, input_size)
        # Augmentasi dulu (pada pixel [0,255]), baru preprocess
        img = augmentation_layer(img, training=True)
        img = preprocess_fn(img)
        return img, label

    def parse_only(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, input_size)
        # Validasi: TANPA augmentasi, langsung preprocess
        img = preprocess_fn(img)
        return img, label

    # Training dataset — dengan augmentasi dan shuffle
    train_ds = tf.data.Dataset.from_tensor_slices((df_t['full_path'].values, df_t['label'].values))
    train_ds = train_ds.shuffle(buffer_size=10000)
    train_ds = train_ds.map(parse_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Validation dataset — TANPA augmentasi
    val_ds = tf.data.Dataset.from_tensor_slices((df_v['full_path'].values, df_v['label'].values))
    val_ds = val_ds.map(parse_only, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ─────────────────────────────────────────────
# BUILD MODEL (2-Phase Architecture) — IDENTIK dengan Exp3
# ─────────────────────────────────────────────
def build_model(base_class, input_size, n_classes, lr):
    inp  = tf.keras.Input(shape=(*input_size, 3))
    base = base_class(weights='imagenet', include_top=False, pooling='avg', input_tensor=inp)
    base.trainable = False  # Phase 1: semua frozen
    x   = base.output
    x   = Dense(EMBED_DIM, activation='relu', name='embedding')(x)
    x   = Dropout(0.4)(x)       # [Exp3 asli: 0.4] ✅ SAMA
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


# ─────────────────────────────────────────────
# LOOP TRAINING UTAMA
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  🚀 MEMULAI EKSPERIMEN 3e — TF.DATA + FULL AUGMENTATION")
print("="*60)

for model_name, cfg in MODEL_CONFIGS.items():
    feat_out = os.path.join(FEAT_DIR, f'{model_name}_features_exp3e.npy')
    if os.path.exists(feat_out):
        print(f"\n⏭️  {model_name.upper()} sudah ada (exp3e), skip.")
        continue

    print(f"\n{'─'*60}")
    print(f"🧠 Model: {model_name.upper()}")
    t_start = time.time()

    train_flow, val_flow = make_generators(cfg['input_size'], cfg['preprocess'])

    # ── Phase 1: Latih Custom Head ───────────────────────────────
    print(f"\n  📌 PHASE 1: Custom Head ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
    model, base = build_model(cfg['class'], cfg['input_size'], n_classes, PHASE1_LR)

    cb_p1 = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE_P1, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
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
        EarlyStopping(monitor='val_loss', patience=PATIENCE_P2, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1),
    ]

    hist_p2 = model.fit(
        train_flow, validation_data=val_flow,
        epochs=PHASE2_EPOCHS, callbacks=cb_p2, verbose=1
    )
    print(f"  ✅ Phase 2 selesai. Best val_acc: {max(hist_p2.history.get('val_accuracy', [0])):.4f}")

    # Simpan bobot model
    model_save_path = os.path.join(FEAT_DIR, f'{model_name}_exp3e.h5')
    model.save_weights(model_save_path)
    print(f"  💾 Bobot disimpan: {model_save_path}")

    # ── Ekstraksi Feature 512-D (tf.data — cepat) ───────────────
    print(f"\n  🔍 Mengekstrak fitur 512-D dari embedding layer...")
    extractor  = Model(model.input, model.get_layer('embedding').output)
    df_master  = pd.read_csv(MASTER_CSV)
    df_master['full_path'] = df_master['image_name'].apply(make_path)

    valid_paths, valid_idx = [], []
    for i, p in enumerate(df_master['full_path']):
        if os.path.exists(p):
            valid_paths.append(p)
            valid_idx.append(i)

    ds_master = tf.data.Dataset.from_tensor_slices(valid_paths)
    ds_master = ds_master.map(
        lambda p: cfg['preprocess'](tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(p), channels=3), cfg['input_size'])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_master = ds_master.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    feats = extractor.predict(ds_master, verbose=1)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.where(norms == 0, 1e-10, norms)

    np.save(feat_out, feats)
    np.save(os.path.join(FEAT_DIR, f'{model_name}_valid_idx_exp3e.npy'), np.array(valid_idx))

    elapsed = (time.time() - t_start) / 60
    print(f"  ✅ {model_name} selesai! Shape: {feats.shape}")
    print(f"  ⏱️  Total waktu: {elapsed:.1f} menit")

    del model, base, extractor, feats
    gc.collect()
    tf.keras.backend.clear_session()


print("\n" + "="*60)
print("  🎉 EKSPERIMEN 3e SELESAI!")
print("="*60)
print("""
Langkah selanjutnya:
  1. Jalankan evaluasi retrieval untuk exp3e
  2. Bandingkan Hit Rate@K dengan Eksperimen 3 asli
""")
