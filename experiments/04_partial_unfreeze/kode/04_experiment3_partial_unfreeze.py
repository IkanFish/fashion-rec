"""
=============================================================
  VISUAL-BASED FASHION RECOMMENDER SYSTEM
  Notebook 04 — Eksperimen 3: Partial Unfreeze Fine-Tuning
  Environment: Local WSL2 + RTX 4060 Ti
=============================================================
Tujuan:
  Meningkatkan performa di atas Eksperimen 2 dengan membuka sebagian
  layer akhir dari base CNN (Partial Unfreezing). Ini memungkinkan
  model yang sebelumnya hanya melatih Custom Head-nya saja, kini
  juga mengadaptasi lapisan konvolusi dalam pada domain fashion.

Perbedaan utama vs Eksperimen 2 (02_feature_extraction.py):
  - UNFREEZE_BLOCKS layer terakhir base CNN dibuka (Trainable = True)
  - Learning rate jauh lebih kecil (1e-5 vs 1e-4) untuk layer yang dibuka
  - Input resolution lebih besar (InceptionV3 tetap 299, lainnya 256)
  - Sistem 2-fase: Phase 1 = latih head saja, Phase 2 = unfreeze + fine-tune
  - Feature vector output tetap 512-D (kompatibel dengan 03_evaluation.py)

Output:
  - features/{model}_features_exp3.npy   → Feature matrix 512-D
  - features/{model}_valid_idx_exp3.npy  → Index gambar yang valid
  - features/{model}_exp3.h5             → Bobot model terlatih
=============================================================
"""

import os, time, gc
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPU available: {len(gpus) > 0}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")


# ─────────────────────────────────────────────
# CELL 2: Konfigurasi Path & Hyperparameter
# ─────────────────────────────────────────────
BASE_DIR    = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
FEAT_DIR    = os.path.join(BASE_DIR, 'features')
MASTER_CSV  = os.path.join(BASE_DIR, 'master_dataset.csv')
FULL_CSV    = os.path.join(BASE_DIR, 'full_dataset.csv')
IMG_DIR     = os.path.join(BASE_DIR, 'In-shop Clothes Retrieval Benchmark', 'img', 'img')

if not os.path.exists(IMG_DIR):
    IMG_DIR = os.path.join(BASE_DIR, 'In-shop Clothes Retrieval Benchmark', 'img')

os.makedirs(FEAT_DIR, exist_ok=True)

# ── Hyperparameter Utama ──────────────────────────────────────────────────────
# Phase 1: Latih Custom Head saja (base masih frozen)
PHASE1_EPOCHS   = 5       # Lebih pendek karena hanya head
PHASE1_LR       = 1e-4    # Sama seperti Eksperimen 2
PATIENCE        = 3

# Phase 2: Unfreeze sebagian layer base + fine-tune bersama
PHASE2_EPOCHS   = 10      # Epoch untuk fine-tune layer yang dibuka
PHASE2_LR       = 1e-5    # SANGAT rendah agar tidak merusak bobot lama
UNFREEZE_BLOCKS = 30      # Berapa layer terakhir base CNN yang dibuka

BATCH_SIZE      = 24      # Sedikit dikurangi karena beban lebih berat
EMBED_DIM       = 512     # Output feature vector tetap 512-D

MODEL_CONFIGS = {
    'resnet50': {
        'class'     : ResNet50,
        'input_size': (256, 256),   # Ditingkatkan dari 224 → 256
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
    },
    'vgg19': {
        'class'     : VGG19,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.vgg19.preprocess_input,
    },
    'inceptionv3': {
        'class'     : InceptionV3,
        'input_size': (299, 299),   # InceptionV3 tetap 299 (minimum arsitektur)
        'preprocess': tf.keras.applications.inception_v3.preprocess_input,
    },
    'mobilenetv3': {
        'class'     : MobileNetV3Large,
        'input_size': (256, 256),
        'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
    },
}

print("✅ Config loaded!")
print(f"📁 Fitur akan disimpan di: {FEAT_DIR}")


# ─────────────────────────────────────────────
# CELL 3: Load Dataset Training
# ─────────────────────────────────────────────
print("\n📦 Memuat dataset training...")
df_full = pd.read_csv(FULL_CSV)
df_train = df_full[df_full['split'] == 'train'].copy()

# Bangun full path gambar
def make_wsl_path(p):
    """Konversi path relatif ke path absolut WSL2."""
    p = str(p).replace('\\', '/')
    return os.path.join(IMG_DIR, p)

df_train['full_path'] = df_train['image_name'].apply(make_wsl_path)

# Filter gambar yang benar-benar ada di disk
mask = df_train['full_path'].apply(os.path.exists)
df_train = df_train[mask].reset_index(drop=True)

print(f"📊 Total gambar training : {len(df_train):,}")
print(f"📊 Jumlah kelas kategori : {df_train['category'].nunique()}")
n_classes = df_train['category'].nunique()

# Label Encoding
cat_to_idx = {c: i for i, c in enumerate(sorted(df_train['category'].unique()))}
df_train['label'] = df_train['category'].map(cat_to_idx)

# Train/Val split (80/20), stratified
df_t, df_v = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])
df_t = df_t.reset_index(drop=True)
df_v = df_v.reset_index(drop=True)


# ─────────────────────────────────────────────
# CELL 4: ImageDataGenerator (Augmentasi)
# ─────────────────────────────────────────────
def make_generators(input_size, preprocess_fn):
    """Buat training dan validation generator untuk satu model."""
    # Augmentasi DITINGKATKAN (Agresif) — Kapasitas model di Eksperimen 3 
    # jauh lebih besar karena 30 layer dibuka. Tanpa augmentasi berat, 
    # model akan dengan cepat menghafal/overfit data latih.
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        horizontal_flip=True,
        rotation_range=20,      
        zoom_range=0.25,        
        shear_range=0.15,       
        brightness_range=[0.8, 1.2],  
        width_shift_range=0.1,        
        height_shift_range=0.1,       
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)


    def flow(df, gen, shuffle):
        return gen.flow_from_dataframe(
            df,
            x_col='full_path',
            y_col='label',
            target_size=input_size,
            batch_size=BATCH_SIZE,
            class_mode='raw',
            shuffle=shuffle,
        )

    return flow(df_t, train_gen, True), flow(df_v, val_gen, False)


# ─────────────────────────────────────────────
# CELL 5: Build Model (2-Phase Architecture)
# ─────────────────────────────────────────────
def build_model(base_class, input_size, n_classes, lr):
    """
    Bangun model dengan custom head.
    Pada Phase 1, base sepenuhnya frozen.
    Pada Phase 2, UNFREEZE_BLOCKS layer terakhir base akan dibuka.
    """
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


def unfreeze_top_layers(base, n_unfreeze, new_lr):
    """
    Buka n_unfreeze layer terakhir base model.
    Gunakan learning rate yang sangat kecil agar tidak merusak bobot lama.
    """
    # Buka n layer terakhir
    for layer in base.layers[-n_unfreeze:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            # BatchNorm tetap frozen — penting agar statistik populasi tidak berubah
            layer.trainable = True

    total_trainable = sum(1 for l in base.layers if l.trainable)
    print(f"   🔓 {total_trainable} layers dibuka dari {len(base.layers)} total layers base")


# ─────────────────────────────────────────────
# CELL 6: Loop Training (Phase 1 + Phase 2)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  🚀 MEMULAI EKSPERIMEN 3 — PARTIAL UNFREEZE FINE-TUNING")
print("="*60)

for model_name, cfg in MODEL_CONFIGS.items():
    feat_out = os.path.join(FEAT_DIR, f'{model_name}_features_exp3.npy')
    if os.path.exists(feat_out):
        print(f"\n⏭️  {model_name.upper()} sudah ada, skip.")
        continue

    print(f"\n{'─'*60}")
    print(f"🧠 Model: {model_name.upper()}")

    # ── Generators ───────────────────────────────────────────────
    train_flow, val_flow = make_generators(cfg['input_size'], cfg['preprocess'])

    # ── Phase 1: Latih Custom Head ───────────────────────────────
    print(f"\n  📌 PHASE 1: Melatih Custom Head ({PHASE1_EPOCHS} epochs, LR={PHASE1_LR})")
    model, base = build_model(cfg['class'], cfg['input_size'], n_classes, PHASE1_LR)

    cb_p1 = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ]

    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=PHASE1_EPOCHS,
        callbacks=cb_p1,
        verbose=1,
    )

    # ── Phase 2: Unfreeze + Fine-Tune Layer Base ─────────────────
    print(f"\n  📌 PHASE 2: Partial Unfreeze ({UNFREEZE_BLOCKS} layers, LR={PHASE2_LR})")
    unfreeze_top_layers(base, UNFREEZE_BLOCKS, PHASE2_LR)

    # Recompile dengan LR lebih rendah
    model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE2_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    cb_p2 = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE + 1, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1),
    ]

    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=PHASE2_EPOCHS,
        callbacks=cb_p2,
        verbose=1,
    )

    # Simpan bobot model setelah Phase 2
    model_save_path = os.path.join(FEAT_DIR, f'{model_name}_exp3.h5')
    model.save_weights(model_save_path)
    print(f"  💾 Bobot model disimpan: {model_save_path}")


    # ─────────────────────────────────────────────
    # CELL 7: Ekstraksi Feature 512-D (dari embedding layer)
    # ─────────────────────────────────────────────
    print(f"\n  🔍 Mengekstrak fitur 512-D dari embedding layer...")

    # Buat feature extractor: Input → embedding layer (Dense 512)
    extractor = Model(model.input, model.get_layer('embedding').output)

    df_master = pd.read_csv(MASTER_CSV)
    df_master['full_path'] = df_master['image_name'].apply(make_wsl_path)

    features_list = []
    valid_idx_list = []

    for idx, row in tqdm(df_master.iterrows(), total=len(df_master), desc=f"Extracting {model_name}"):
        img_path = row['full_path']
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert('RGB').resize(cfg['input_size'])
            arr = np.array(img, dtype=np.float32)[np.newaxis]
            arr = cfg['preprocess'](arr)
            feat = extractor.predict(arr, verbose=0)[0]
            # L2 normalize
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            features_list.append(feat)
            valid_idx_list.append(idx)
        except Exception as e:
            pass

    feat_matrix = np.vstack(features_list).astype(np.float32)
    valid_idx_arr = np.array(valid_idx_list)

    np.save(feat_out, feat_matrix)
    np.save(os.path.join(FEAT_DIR, f'{model_name}_valid_idx_exp3.npy'), valid_idx_arr)

    print(f"  ✅ {model_name} selesai! Shape: {feat_matrix.shape}")
    print(f"  💾 Tersimpan: {feat_out}")

    del model, base, extractor, feat_matrix
    gc.collect()
    tf.keras.backend.clear_session()

print("\n" + "="*60)
print("  🎉 EKSPERIMEN 3 SELESAI!")
print("  Selanjutnya: Jalankan 03_evaluation.py")
print("  Ganti nama file .npy dengan suffix _exp3 di konfigurasi")
print("="*60)
