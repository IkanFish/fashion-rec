import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_1 = """# Eksperimen 1: Baseline Feature Extraction
Notebook ini bertujuan untuk mengekstrak fitur visual secara murni menggunakan model pre-trained (ImageNet) **tanpa fine-tuning**, sesuai dengan desain Eksperimen 1.

Fitur akan diekstrak langsung dari layer `GlobalAveragePooling2D`."""

code_1 = """import os, gc, time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image as keras_image

print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")
"""

code_2 = """# Konfigurasi Path
BASE_DIR = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'

FEAT_DIR   = os.path.join(BASE_DIR, 'features')
DATASET_DIR= os.path.join(BASE_DIR, 'dataset')
MASTER_CSV = os.path.join(DATASET_DIR, 'master_dataset.csv')
# image_name di CSV sudah include prefix 'img/...', jadi IMG_BASE = folder In-shop
IMG_BASE   = os.path.join(DATASET_DIR, 'In-shop Clothes Retrieval Benchmark')

os.makedirs(FEAT_DIR, exist_ok=True)
BATCH_SIZE = 32

MODEL_CONFIGS = {
    'resnet50': {
        'class': ResNet50,
        'input_size': (224, 224),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
    },
    'vgg19': {
        'class': VGG19,
        'input_size': (224, 224),
        'preprocess': tf.keras.applications.vgg19.preprocess_input,
    },
    'inceptionv3': {
        'class': InceptionV3,
        'input_size': (299, 299),
        'preprocess': tf.keras.applications.inception_v3.preprocess_input,
    },
    'mobilenetv3': {
        'class': MobileNetV3Large,
        'input_size': (224, 224),
        'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
    },
}
print("Config loaded!")
"""

code_3 = """# Load Data
df_master = pd.read_csv(MASTER_CSV)
# image_name sudah include prefix 'img/WOMEN/...' → join ke IMG_BASE
df_master['full_path'] = df_master['image_name'].apply(
    lambda x: os.path.join(IMG_BASE, x.replace('/', os.sep))
)
image_paths = df_master['full_path'].tolist()
# Verifikasi path pertama
print(f"Sample path: {image_paths[0]}")
print(f"Exists: {os.path.exists(image_paths[0])}")
print(f"Total gambar untuk diekstrak: {len(image_paths)}")
"""

code_4 = """# Fungsi Bantuan Ekstraksi
def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return matrix / norms

def load_and_preprocess_image(img_path, target_size, preprocess_fn):
    try:
        img = keras_image.load_img(img_path, target_size=target_size)
        x   = keras_image.img_to_array(img)
        x   = np.expand_dims(x, axis=0)
        return preprocess_fn(x)
    except Exception:
        return None

def extract_features_batched(model, paths, target_size, preprocess_fn, desc):
    features, valid_idx = [], []
    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc=desc):
        batch_paths = paths[start : start + BATCH_SIZE]
        batch_imgs, batch_idx = [], []
        
        for i, path in enumerate(batch_paths):
            x = load_and_preprocess_image(path, target_size, preprocess_fn)
            if x is not None:
                batch_imgs.append(x)
                batch_idx.append(start + i)
                
        if batch_imgs:
            batch_array = np.vstack(batch_imgs)
            batch_feats = model.predict(batch_array, verbose=0)
            features.append(batch_feats)
            valid_idx.extend(batch_idx)
            
    return np.vstack(features), valid_idx
"""

code_5 = """# Proses Ekstraksi Baseline
extraction_log = {}

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\\n🧠 Baseline Extraction: {model_name.upper()}")
    feat_path = os.path.join(FEAT_DIR, f"{model_name}_features_baseline.npy")
    idx_path  = os.path.join(FEAT_DIR, f"{model_name}_valid_idx_baseline.npy")
    
    if os.path.exists(feat_path):
        print(f"  ⏩ Fitur {model_name} sudah ada, skip...")
        continue
        
    # Build model (FROZEN ImageNet weights, include_top=False)
    base_model = cfg['class'](weights='imagenet', include_top=False, input_shape=cfg['input_size']+(3,))
    
    # Tambahkan layer GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    
    t0 = time.time()
    feat_mat, valid_idx = extract_features_batched(
        model=model,
        paths=image_paths,
        target_size=cfg['input_size'],
        preprocess_fn=cfg['preprocess'],
        desc=f"[{model_name}]"
    )
    
    feat_mat = l2_normalize(feat_mat)
    np.save(feat_path, feat_mat)
    np.save(idx_path, np.array(valid_idx))
    
    print(f"  ✅ Selesai | Waktu: {time.time() - t0:.1f}s | Dimensi: {feat_mat.shape}")
    
    # Cleanup memory
    del model, base_model, feat_mat
    gc.collect()
    tf.keras.backend.clear_session()
    
print("\\n🎉 SELURUH PROSES EKSTRAKSI BASELINE SELESAI!")
"""

text_2 = """### Langkah Selanjutnya:
Setelah _cell_ di atas selesai dieksekusi, fitur akan tersimpan di dalam folder `features/` dengan nama berakhiran `_features_baseline.npy` (misal: `resnet50_features_baseline.npy`).

Kamu bisa memodifikasi dan menjalankan script `03_evaluation.py` (dengan mengubah baris load feature agar memuat nama file ini) untuk mendapatkan skor **Diversity**, Precision, dll, pada Eksperimen 1 Baseline.
"""

nb.cells = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(text_2)
]

output_path = r'd:\\Antigravity\\Visual Based Rekomender Sistem\\notebooks\\baseline_extraction.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
