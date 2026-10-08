"""
=============================================================
  VISUAL-BASED FASHION RECOMMENDER SYSTEM
  Notebook 02 — Lightweight Fine-Tuning & Feature Extraction
  Environment: Local WSL2 + RTX 4060 Ti
=============================================================
Tujuan:
  - Load full_dataset.csv untuk melatih custom head CNN
  - Augmentasi data training via ImageDataGenerator
  - Fine-tune 4 model (ResNet50, VGG19, InceptionV3, MobileNetV3)
  - Ekstrak feature vector 512-D dari custom head yang sudah dilatih
  - Simpan feature matrix & image index
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPU available: {len(gpus) > 0}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")


# ─────────────────────────────────────────────
# CELL 2: Konfigurasi Path WSL2
# ─────────────────────────────────────────────
BASE_DIR    = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
FEAT_DIR    = os.path.join(BASE_DIR, 'features')

# Sesuai output LS dari terminal kamu, lokasinya langsung di BASE_DIR
MASTER_CSV  = os.path.join(BASE_DIR, 'master_dataset.csv')
FULL_CSV    = os.path.join(BASE_DIR, 'full_dataset.csv')

# Path ke gambar original D:\...\img\img
IMG_DIR     = os.path.join(BASE_DIR, 'In-shop Clothes Retrieval Benchmark', 'img', 'img')

# Safety check jika folder sedikit berbeda
if not os.path.exists(IMG_DIR):
    IMG_DIR = os.path.join(BASE_DIR, 'In-shop Clothes Retrieval Benchmark', 'img')

os.makedirs(FEAT_DIR, exist_ok=True)

EPOCHS      = 10     # Bisa dinaikkan ke 20-30 jika hasilnya belum saturasi
BATCH_SIZE  = 32     # Sangat aman untuk RTX 4060 Ti 16GB
LR          = 1e-4
PATIENCE    = 3

MODEL_CONFIGS = {
    'resnet50': {
        'class'     : ResNet50,
        'input_size': (224, 224),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
    },
    'vgg19': {
        'class'     : VGG19,
        'input_size': (224, 224),
        'preprocess': tf.keras.applications.vgg19.preprocess_input,
    },
    'inceptionv3': {
        'class'     : InceptionV3,
        'input_size': (299, 299),
        'preprocess': tf.keras.applications.inception_v3.preprocess_input,
    },
    'mobilenetv3': {
        'class'     : MobileNetV3Large,
        'input_size': (224, 224),
        'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
    },
}

print("✅ Config loaded!")
print(f"📁 Fitur akan disimpan di: {FEAT_DIR}")


# ─────────────────────────────────────────────
# CELL 3: Load Data Training (Dari full_dataset)
# ─────────────────────────────────────────────
print("\n📦 Memuat dataset training...")
df_full = pd.read_csv(FULL_CSV)

# Ubah nama file gambar menjadi path fix untuk Linux, convert '\' ke '/'
df_full['full_path'] = df_full['image_name'].apply(lambda x: os.path.join(IMG_DIR, x.replace('\\', '/')))

# Ambil HANYA yang split='train' untuk proses belajar fine-tuning
df_train_all = df_full[df_full['split'] == 'train'].copy()

# Filter kategori yang kemunculannya terlalu sedikit
cat_counts = df_train_all['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 20].index.tolist()
df_train_all = df_train_all[df_train_all['category'].isin(valid_cats)].reset_index(drop=True)

# Encode kategori menjadi angka berurutan (0, 1, 2, ...) untuk TensorFlow
label_enc = LabelEncoder()
df_train_all['category_idx'] = label_enc.fit_transform(df_train_all['category'])
num_classes = len(label_enc.classes_)

print(f"📊 Total gambar training : {len(df_train_all):,}")
print(f"📊 Jumlah kelas kategori : {num_classes}")

# Split 80/20 untuk set validasi awal
train_df, val_df = train_test_split(
    df_train_all, 
    test_size=0.2, 
    stratify=df_train_all['category_idx'], 
    random_state=42
)

# Keras ImageDataGenerator butuh label berbentuk string
train_df['category_str'] = train_df['category_idx'].astype(str)
val_df['category_str']   = val_df['category_idx'].astype(str)


# ─────────────────────────────────────────────
# CELL 4: Fugsi Utility (Model & Generator)
# ─────────────────────────────────────────────
def build_finetune_model(cfg, num_classes):
    """Membangun arsitektur: Base model (frozen) + Head (Trainable Dense 512)."""
    # 1. Base model FROZEN
    base_model = cfg['class'](weights='imagenet', include_top=False, input_shape=cfg['input_size']+(3,))
    base_model.trainable = False
    
    # 2. Custom Head ("The Lightweight Part")
    x = GlobalAveragePooling2D()(base_model.output)
    
    # KUNCI UTAMA: Layer 512-D ini akan ditarik fiturnya (menjadi embeding)
    x = Dense(512, activation='relu', name='feature_embedding')(x)
    x = Dropout(0.3)(x)
    
    # 3. Layer Klasifikasi Akhir
    outputs = Dense(num_classes, activation='softmax', name='classification_head')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_data_generators(cfg, train_df, val_df):
    """Menghasilkan pipeline data dengan Augmentasi Gambar."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=cfg['preprocess'],
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.2,       # Zoom in/out random 20%
        shear_range=0.1
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=cfg['preprocess']
    )
    
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='full_path',
        y_col='category_str',
        target_size=cfg['input_size'],
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='full_path',
        y_col='category_str',
        target_size=cfg['input_size'],
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )
    return train_gen, val_gen

def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return matrix / norms

def load_and_preprocess_image(img_path, target_size, preprocess_fn):
    try:
        img = keras_image.load_img(img_path, target_size=target_size)
        x   = keras_image.img_to_array(img)
        x   = np.expand_dims(x, axis=0)
        x   = preprocess_fn(x)
        return x
    except Exception:
        return None

def extract_features_batched(model, image_paths, target_size, preprocess_fn, desc='Extracting'):
    n = len(image_paths)
    features = []
    valid_idx = []
    
    for start in tqdm(range(0, n, BATCH_SIZE), desc=desc):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch_imgs  = []
        batch_idx   = []
        
        for i, path in enumerate(batch_paths):
            x = load_and_preprocess_image(path, target_size, preprocess_fn)
            if x is not None:
                batch_imgs.append(x)
                batch_idx.append(start + i)
                
        if not batch_imgs: continue
        
        batch_array  = np.vstack(batch_imgs)
        batch_feats  = model.predict(batch_array, verbose=0)
        features.append(batch_feats)
        valid_idx.extend(batch_idx)

    feature_matrix = np.vstack(features)
    return feature_matrix, valid_idx


# ─────────────────────────────────────────────
# CELL 5: Fine-Tuning Training Loop
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  🚀 MEMULAI FINE-TUNING (TRAINING HEAD)")
print("="*55)

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n🧠 Model: {model_name.upper()}")
    
    model_path = os.path.join(FEAT_DIR, f"{model_name}_finetuned.h5")
    
    if os.path.exists(model_path):
        print(f"  ⏩ Model weight sudah ada, skipping training...")
        continue
        
    train_gen, val_gen = get_data_generators(cfg, train_df, val_df)
    model = build_finetune_model(cfg, num_classes)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    ]
    
    print(f"  ▶️ Memulai training {EPOCHS} Epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print(f"  ✅ Training selesai! Model tersimpan di {model_path}")
    
    # Cleanup memory
    del model, train_gen, val_gen
    gc.collect()
    tf.keras.backend.clear_session()


# ─────────────────────────────────────────────
# CELL 6: Ekstraksi Fitur dari Model Fine-Tuned
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  🔍 EKSTRAKSI FITUR FINAL (UNTUK REKOMENDASI)")
print("="*55)

# Load MASTER_CSV (karena Recommendation mengandalkan 1 foto katalog utama per item)
df_master = pd.read_csv(MASTER_CSV)
df_master['full_path'] = df_master['image_name'].apply(lambda x: os.path.join(IMG_DIR, x.replace('\\', '/')))
image_paths = df_master['full_path'].tolist()

extraction_log = {}

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n🧠 Feature Extraction: {model_name.upper()}")
    
    feat_path  = os.path.join(FEAT_DIR, f'{model_name}_features.npy')
    idx_path   = os.path.join(FEAT_DIR, f'{model_name}_valid_idx.npy')
    model_path = os.path.join(FEAT_DIR, f"{model_name}_finetuned.h5")

    # Ini dibuat False untuk MEMAKSA EKSTRAKSI ULANG (meng-overwrite hasil lama dari baseline Colab)
    # Jika kamu mau melanjutkan estraksi yg putus, bisa ganti jadi if os.path.exists(...)
    if False:
        pass

    full_model = build_finetune_model(cfg, num_classes)
    if os.path.exists(model_path):
        full_model.load_weights(model_path)
    else:
        print(f"  ⚠️ Warning: Weight {model_path} tidak ditemukan! Menggunakan untrained head.")

    # ❗ KUNCI UTAMA: Memotong layer dan hanya memunculkan output dari Dense(512)
    extractor = Model(inputs=full_model.input, outputs=full_model.get_layer('feature_embedding').output)
    
    t0 = time.time()
    feat_mat, valid_idx = extract_features_batched(
        model=extractor,
        image_paths=image_paths,
        target_size=cfg['input_size'],
        preprocess_fn=cfg['preprocess'],
        desc=f"Extracting [{model_name}]"
    )
    
    feat_mat = l2_normalize(feat_mat)
    elapsed = time.time() - t0
    
    np.save(feat_path, feat_mat)
    np.save(idx_path, np.array(valid_idx))
    
    print(f"  💾 Saved: {feat_path}")
    print(f"  ✅ Shape: {feat_mat.shape} (Dimensi kini harusnya 512-D!) | Time: {elapsed:.1f}s")
    
    extraction_log[model_name] = {'n_features': feat_mat.shape[0], 'feature_dim': feat_mat.shape[1], 'status': 'extracted'}
    
    del full_model, extractor, feat_mat
    gc.collect()
    tf.keras.backend.clear_session()


# ─────────────────────────────────────────────
# CELL 7: Simpan Index & Laporan
# ─────────────────────────────────────────────
df_save_path = os.path.join(FEAT_DIR, 'image_index.csv')
df_master.to_csv(df_save_path, index=False)

print("\n" + "="*55)
print("  ✅ LAPORAN AKHIR")
print("="*55)
for name, log in extraction_log.items():
    print(f"{name:<15} N:{log['n_features']:>6,} | Dim:{log['feature_dim']:>4} | Status:{log['status']:>10}")

print("\n🎉 SELURUH PROSES FINE-TUNING DAN EKSTRAKSI LOKAL SELESAI!")
print("➡️ Silakan buka Notebook 03: Evaluation untuk membandingkan Presisi-nya!")
