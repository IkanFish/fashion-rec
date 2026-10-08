# ─────────────────────────────────────────────
# CELL 1: Import Library & System Setup
# ─────────────────────────────────────────────
import os, gc, time
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

print(f"✅ TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPU available: {len(gpus) > 0}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")


# ─────────────────────────────────────────────
# CELL 2: Konfigurasi Path & Model
# ─────────────────────────────────────────────
if os.name == 'posix':
    BASE_DIR = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
else:
    BASE_DIR = r'D:\Antigravity\Visual Based Rekomender Sistem'

FULL_CSV = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')
if not os.path.exists(FULL_CSV):
    FULL_CSV = os.path.join(BASE_DIR, 'skripsi', 'persiapan_skripsi', 'data preperation', 'dataset_output', 'full_dataset.csv')

IMG_DIR   = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'lightweight')
FEAT_DIR  = os.path.join(BASE_DIR, 'features', 'exp2_lightweight')

os.makedirs(FEAT_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_CLASSES = 16 # Jumlah kelas kategori yang dilatih

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
# CELL 3: Memuat Dataset Query & Gallery
# ─────────────────────────────────────────────
print("\n📦 Memuat dataset retrieval (Query & Gallery)...")
df_full = pd.read_csv(FULL_CSV)

# Path gambar presisi
df_full['full_path'] = df_full['image_name'].apply(
    lambda x: os.path.join(IMG_DIR, os.path.normpath(str(x).replace('img/', '', 1))) if str(x).startswith('img/') else os.path.join(IMG_DIR, os.path.normpath(str(x)))
)

df_query   = df_full[df_full['split'] == 'query'].reset_index(drop=True)
df_gallery = df_full[df_full['split'] == 'gallery'].reset_index(drop=True)

print(f"📊 Total QUERY   : {len(df_query):,} gambar")
print(f"📊 Total GALLERY : {len(df_gallery):,} gambar")


# ─────────────────────────────────────────────
# CELL 4: Fungsi Utility Ekstraksi Fitur
# ─────────────────────────────────────────────
def build_extractor(cfg, num_classes, model_path):
    """Membangun ulang arsitektur fine-tuned dan mengambil layer feature_embedding (512-D)."""
    base_model = cfg['class'](weights=None, include_top=False, input_shape=cfg['input_size']+(3,))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', name='feature_embedding')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name='classification_head')(x)
    
    full_model = Model(inputs=base_model.input, outputs=outputs)
    full_model.load_weights(model_path)
    
    # Potong model dan ambil output dari layer 512-D
    extractor = Model(inputs=full_model.input, outputs=full_model.get_layer('feature_embedding').output)
    return extractor

def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return matrix / norms

def extract_features(model, image_paths, input_size, preprocess_fn, desc='Extracting'):
    features = []
    valid_idx = []
    
    for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=desc):
        batch_paths = image_paths[start : start + BATCH_SIZE]
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
        batch_feats = model.predict(batch_array, verbose=0)
        features.append(batch_feats)
        valid_idx.extend(batch_indices)
        
    if not features:
        return np.array([]), []
        
    feat_matrix = np.vstack(features)
    feat_matrix = l2_normalize(feat_matrix)
    return feat_matrix, valid_idx


# ─────────────────────────────────────────────
# CELL 5: Ekstraksi Fitur 512-D (Query & Gallery)
# ─────────────────────────────────────────────
query_paths   = df_query['full_path'].tolist()
gallery_paths = df_gallery['full_path'].tolist()

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*55}")
    print(f"  🧠 EKSTRAKSI FITUR LIGHTWEIGHT (512-D) — {model_name.upper()}")
    print(f"{'='*55}")
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}_finetuned.h5")
    if not os.path.exists(model_path):
        print(f"  ⚠️ Warning: Model {model_path} belum ada, skipping...")
        continue
        
    q_feat_path = os.path.join(FEAT_DIR, f'{model_name}_query_features.npy')
    g_feat_path = os.path.join(FEAT_DIR, f'{model_name}_gallery_features.npy')
    q_idx_path  = os.path.join(FEAT_DIR, f'{model_name}_query_valid_idx.npy')
    g_idx_path  = os.path.join(FEAT_DIR, f'{model_name}_gallery_valid_idx.npy')
    
    if os.path.exists(q_feat_path) and os.path.exists(g_feat_path):
        print(f"  ⏩ Fitur {model_name} sudah ada, skipping...")
        continue
        
    extractor = build_extractor(cfg, NUM_CLASSES, model_path)
    
    # Ekstrak Query
    print(f"  📌 Extracting QUERY ({len(query_paths):,} images)...")
    q_feats, q_valid = extract_features(
        extractor, query_paths, cfg['input_size'], cfg['preprocess'],
        desc=f"Query [{model_name}]"
    )
    
    # Ekstrak Gallery
    print(f"  📌 Extracting GALLERY ({len(gallery_paths):,} images)...")
    g_feats, g_valid = extract_features(
        extractor, gallery_paths, cfg['input_size'], cfg['preprocess'],
        desc=f"Gallery [{model_name}]"
    )
    
    # Simpan Vektor & Indeks
    np.save(q_feat_path, q_feats)
    np.save(g_feat_path, g_feats)
    np.save(q_idx_path, np.array(q_valid))
    np.save(g_idx_path, np.array(g_valid))
    
    print(f"  💾 Saved: {q_feat_path} ({q_feats.shape})")
    print(f"  💾 Saved: {g_feat_path} ({g_feats.shape})")
    
    del extractor
    gc.collect()
    tf.keras.backend.clear_session()

print("\n🎉 EKSTRAKSI FITUR LIGHTWEIGHT (512-D) SELESAI!")
