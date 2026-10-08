"""
=============================================================
  VISUAL-BASED FASHION RECOMMENDER SYSTEM
  Notebook 04d — Eksperimen 3d: Item-ID Training (The Right Way)
  Environment: Local WSL + RTX 4060 Ti
=============================================================
Tujuan:
  Sesuai dengan esensi Retrieval, kita MELATIH MODEL UNTUK MENGENALI 
  ITEM UNIK (item_id), BUKAN HANYA KATEGORI UMUM.
  
  Dengan jumlah kelas ~7900+ (sebanyak jumlah item unik), layer 
  Embedding (512-D) akan dipaksa secara matematis untuk membedakan 
  setiap potong baju secara spesifik, yang akan meroketkan 
  Hit Rate / Cosine Similarity saat evaluasi.
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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.model_selection import train_test_split

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ── Path (Cross-Platform) ──────────────────────────────────────────────
if platform.system() == 'Linux' and os.path.exists('/mnt/d'):
    BASE_DIR = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
else:
    BASE_DIR = r'D:\Antigravity\Visual Based Rekomender Sistem'

FEAT_DIR   = os.path.join(BASE_DIR, 'features', 'exp3d')
MASTER_CSV = os.path.join(BASE_DIR, 'dataset', 'master_dataset.csv')
FULL_CSV   = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')

if platform.system() == 'Linux' and os.path.exists('/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'):
    IMG_DIR = '/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'
else:
    IMG_DIR = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')

os.makedirs(FEAT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# HYPERPARAMETER
# ─────────────────────────────────────────────
PHASE1_EPOCHS   = 15
PHASE1_LR       = 1e-4

PHASE2_EPOCHS   = 30
PHASE2_LR       = 1e-5    
UNFREEZE_BLOCKS = 50

PATIENCE        = 5
BATCH_SIZE      = 64
EMBED_DIM       = 512

MODEL_CONFIGS = {
    'resnet50': {
        'class'     : ResNet50,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
    },
}

# ─────────────────────────────────────────────
# LOAD DATASET TRAINING (PERUBAHAN KRUSIAL DI SINI)
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

# 🔥 DI SINI KUNCI UTAMANYA: Kita pakai item_id sebagai target! 🔥
print(f"📊 Total gambar training : {len(df_train):,}")
print(f"📊 Jumlah KATEGORI unik  : {df_train['category'].nunique()} (Kita TINGGALKAN ini)")
print(f"📊 Jumlah ITEM_ID unik   : {df_train['item_id'].nunique()} (Ini TARGET BARU kita!)")
n_classes = df_train['item_id'].nunique()

item_to_idx = {c: i for i, c in enumerate(sorted(df_train['item_id'].unique()))}
df_train['label'] = df_train['item_id'].map(item_to_idx)

# Gunakan stratify=None karena ada item_id yang mungkin gambarnya sangat sedikit
df_t, df_v = train_test_split(df_train, test_size=0.15, random_state=42)
df_t = df_t.reset_index(drop=True)
df_v = df_v.reset_index(drop=True)

# ─────────────────────────────────────────────
# TF.DATA GENERATORS
# ─────────────────────────────────────────────
def make_generators(input_size, preprocess_fn):
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
            
        ds = ds.map(lambda p, l: parse_image(p, l, augment), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    return create_dataset(df_t, augment=True, shuffle=True), create_dataset(df_v, augment=False, shuffle=False)

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
def build_model(base_class, input_size, n_classes, lr):
    inp  = tf.keras.Input(shape=(*input_size, 3))
    base = base_class(weights='imagenet', include_top=False, pooling='avg', input_tensor=inp)
    base.trainable = False
    x   = base.output
    x   = Dense(EMBED_DIM, activation='relu', name='embedding')(x)
    x   = Dropout(0.4)(x)
    # Output layer sekarang berukuran ~7900 nodes!
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, base

def unfreeze_top_layers(base, n_unfreeze):
    for layer in base.layers[-n_unfreeze:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
for model_name, cfg in MODEL_CONFIGS.items():
    feat_out = os.path.join(FEAT_DIR, f'{model_name}_features_exp3d.npy')
    if os.path.exists(feat_out):
        continue

    train_flow, val_flow = make_generators(cfg['input_size'], cfg['preprocess'])

    print(f"\n📌 PHASE 1: Custom Head (Mengenali {n_classes} Item ID)")
    model, base = build_model(cfg['class'], cfg['input_size'], n_classes, PHASE1_LR)
    cb_p1 = [EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)]
    model.fit(train_flow, validation_data=val_flow, epochs=PHASE1_EPOCHS, callbacks=cb_p1, verbose=1)

    print(f"\n📌 PHASE 2: Partial Unfreeze")
    unfreeze_top_layers(base, UNFREEZE_BLOCKS)
    model.compile(optimizer=tf.keras.optimizers.Adam(PHASE2_LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb_p2 = [EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)]
    model.fit(train_flow, validation_data=val_flow, epochs=PHASE2_EPOCHS, callbacks=cb_p2, verbose=1)

    model_save_path = os.path.join(FEAT_DIR, f'{model_name}_exp3d.h5')
    model.save_weights(model_save_path)

    print(f"\n🔍 Ekstraksi 512-D Fitur Master Dataset...")
    extractor  = Model(model.input, model.get_layer('embedding').output)
    
    # Ekstrak secara cepat pakai tf.data
    df_master  = pd.read_csv(MASTER_CSV)
    df_master['full_path'] = df_master['image_name'].apply(make_path)
    valid_paths, valid_idx = [], []
    for i, p in enumerate(df_master['full_path']):
        if os.path.exists(p):
            valid_paths.append(p)
            valid_idx.append(i)
            
    ds_master = tf.data.Dataset.from_tensor_slices(valid_paths)
    ds_master = ds_master.map(lambda p: cfg['preprocess'](tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(p), channels=3), cfg['input_size'])), num_parallel_calls=tf.data.AUTOTUNE)
    ds_master = ds_master.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    feats = extractor.predict(ds_master, verbose=1)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.where(norms == 0, 1e-10, norms)

    np.save(feat_out, feats)
    np.save(os.path.join(FEAT_DIR, f'{model_name}_valid_idx_exp3d.npy'), np.array(valid_idx))

print("\n🎉 EKSPERIMEN 3d SELESAI!")
