# %% [markdown]
# # 🔍 Eksperimen Evaluasi Retrieval — Sisi Visi Komputer
# 
# **Tujuan:** Menguji performa model CNN (ResNet50) yang dilatih dengan tf.data (exp3c)
# dalam me-retrieve item pakaian yang sama dari gambar query.
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

# ── Path (Cross-Platform) ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE_DIR, 'features')
FULL_CSV = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')

# Cek jika native Linux path ada
if platform.system() == 'Linux' and os.path.exists('/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'):
    IMG_DIR = '/home/ghani/fashion_project/In-shop Clothes Retrieval Benchmark/Img'
else:
    IMG_DIR  = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')

# ── Konfigurasi ──────────────────────
EXPERIMENT = 'exp3c'
EMBEDDING_LAYER_NAME = 'embedding'
BATCH_SIZE = 64
EMBED_DIM = 512

MODEL_CONFIGS = {
    'resnet50': {
        'class': ResNet50, 'input_size': (256, 256),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
        'h5_file': 'resnet50_exp3c.h5', 'pooling': 'avg',
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

# ── Ekstraktor ──────────────────────────────────────────────
def build_extractor(cfg, n_classes):
    inp = tf.keras.Input(shape=(*cfg['input_size'], 3))
    base = cfg['class'](weights='imagenet', include_top=False, 
                        pooling=cfg['pooling'], input_tensor=inp)
    base.trainable = False
    x = base.output

    x   = Dense(EMBED_DIM, activation='relu', name=EMBEDDING_LAYER_NAME)(x)
    x   = Dropout(0.4)(x)
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
    # Menggunakan tf.data.Dataset untuk ekstraksi gambar yang secepat kilat
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

# ── Proses Ekstraksi ──────────────────────────────────────────────
df_train = df_full[df_full['split'] == 'train'].copy()
n_classes = df_train['category'].nunique()

RETRIEVAL_FEAT_DIR = os.path.join(FEAT_DIR, f'retrieval_{EXPERIMENT}')
os.makedirs(RETRIEVAL_FEAT_DIR, exist_ok=True)

query_paths   = df_query['full_path'].tolist()
gallery_paths = df_gallery['full_path'].tolist()

for model_name, cfg in MODEL_CONFIGS.items():
    q_feat_path = os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_features.npy')
    g_feat_path = os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_features.npy')
    q_idx_path  = os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_valid_idx.npy')
    g_idx_path  = os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_valid_idx.npy')

    if os.path.exists(q_feat_path) and os.path.exists(g_feat_path):
        print(f"\n⏩ Fitur {model_name.upper()} sudah ada, langsung ke evaluasi!")
    else:
        extractor = build_extractor(cfg, n_classes)

        print(f"\n📌 Extracting QUERY features ({len(query_paths):,} images)...")
        q_feats, q_valid = extract_features(extractor, query_paths, cfg['input_size'], cfg['preprocess'])

        print(f"📌 Extracting GALLERY features ({len(gallery_paths):,} images)...")
        g_feats, g_valid = extract_features(extractor, gallery_paths, cfg['input_size'], cfg['preprocess'])

        np.save(q_feat_path, q_feats)
        np.save(g_feat_path, g_feats)
        np.save(q_idx_path, np.array(q_valid))
        np.save(g_idx_path, np.array(g_valid))

        del extractor, q_feats, g_feats
        gc.collect()
        tf.keras.backend.clear_session()

# ── Evaluasi ──────────────────────────────────────────────
def evaluate_retrieval(query_features, gallery_features, query_item_ids, gallery_item_ids, k_values=[1, 5, 10, 20]):
    n_queries = len(query_features)
    similarity_matrix = query_features @ gallery_features.T
    results = {}
    recall_at_k = {k: 0 for k in k_values}
    average_precisions = []

    for i in range(n_queries):
        query_id = query_item_ids[i]
        gt_mask = (gallery_item_ids == query_id)
        n_relevant = gt_mask.sum()

        if n_relevant == 0: continue

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
    return results

all_results = {}
for model_name in MODEL_CONFIGS.keys():
    q_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_features.npy'))
    g_feats = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_features.npy'))
    q_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_query_valid_idx.npy'))
    g_valid = np.load(os.path.join(RETRIEVAL_FEAT_DIR, f'{model_name}_gallery_valid_idx.npy'))

    q_item_ids = df_query.iloc[q_valid]['item_id'].values
    g_item_ids = df_gallery.iloc[g_valid]['item_id'].values

    results = evaluate_retrieval(q_feats, g_feats, q_item_ids, g_item_ids)
    all_results[model_name] = results
    
    print(f"\n{'='*40}")
    print(f"📊 EVALUASI {model_name.upper()} (EXP3c)")
    print(f"{'='*40}")
    print(f"  Hit Rate @ 1 : {results['Recall@1']*100:.2f}%")
    print(f"  Hit Rate @ 5 : {results['Recall@5']*100:.2f}%")
    print(f"  Hit Rate @ 10: {results['Recall@10']*100:.2f}%")
    print(f"  Hit Rate @ 20: {results['Recall@20']*100:.2f}%")
    print(f"  mAP          : {results['mAP']*100:.2f}%")

df_results = pd.DataFrame(all_results).T
df_results = df_results.rename(columns={'Recall@1': 'Hit@1', 'Recall@5': 'Hit@5', 'Recall@10': 'Hit@10', 'Recall@20': 'Hit@20'})

csv_out = os.path.join(RETRIEVAL_FEAT_DIR, f'cnn_hit_miss_results_{EXPERIMENT}.csv')
df_results.to_csv(csv_out)
print(f"\n💾 Disimpan ke: {csv_out}")
