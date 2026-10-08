import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load Dataset & Path Setup
if os.name == 'posix':
    BASE_DIR = '/mnt/d/Antigravity/Visual Based Rekomender Sistem'
else:
    BASE_DIR = r'D:\Antigravity\Visual Based Rekomender Sistem'

FULL_CSV = os.path.join(BASE_DIR, 'dataset', 'full_dataset.csv')
if not os.path.exists(FULL_CSV):
    FULL_CSV = os.path.join(BASE_DIR, 'skripsi', 'persiapan_skripsi', 'data preperation', 'dataset_output', 'full_dataset.csv')

IMG_DIR = os.path.join(BASE_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')

MODEL_DIR = os.path.join(BASE_DIR, 'models', 'lightweight')
EVAL_DIR  = os.path.join(BASE_DIR, 'skripsi', 'persiapan_skripsi', 'lightweight', 'hasil_evaluasi')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# 1. Reproduksi Data Validasi (Harus sama persis dengan saat training)
print("Memuat dataset untuk evaluasi klasifikasi...")
df_full = pd.read_csv(FULL_CSV)
df_full['full_path'] = df_full['image_name'].apply(
    lambda x: os.path.join(IMG_DIR, os.path.normpath(str(x).replace('img/', '', 1))) if str(x).startswith('img/') else os.path.join(IMG_DIR, os.path.normpath(str(x)))
)

df_train_all = df_full[df_full['split'] == 'train'].copy()
cat_counts = df_train_all['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 20].index.tolist()
df_train_all = df_train_all[df_train_all['category'].isin(valid_cats)].reset_index(drop=True)

label_enc = LabelEncoder()
df_train_all['category_idx'] = label_enc.fit_transform(df_train_all['category'])
num_classes = len(label_enc.classes_)
class_names = label_enc.classes_

_, val_df = train_test_split(
    df_train_all, test_size=0.2, 
    stratify=df_train_all['category_idx'], random_state=42
)
val_df['category_str'] = val_df['category_idx'].astype(str)

print(f"Total gambar validasi untuk pengujian : {len(val_df)}")

# 2. Konfigurasi Model
BATCH_SIZE = 32
MODEL_CONFIGS = {
    'resnet50': {'class': ResNet50, 'input_size': (224, 224), 'preprocess': tf.keras.applications.resnet50.preprocess_input},
    'vgg19': {'class': VGG19, 'input_size': (224, 224), 'preprocess': tf.keras.applications.vgg19.preprocess_input},
    'inceptionv3': {'class': InceptionV3, 'input_size': (299, 299), 'preprocess': tf.keras.applications.inception_v3.preprocess_input},
    'mobilenetv3': {'class': MobileNetV3Large, 'input_size': (224, 224), 'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input},
}

def build_finetune_model(cfg, num_classes):
    base_model = cfg['class'](weights=None, include_top=False, input_shape=cfg['input_size']+(3,))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', name='feature_embedding')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name='classification_head')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# 3. Proses Evaluasi
for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*50}\nEVALUASI KLASIFIKASI: {model_name.upper()}\n{'='*50}")
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}_finetuned.h5")
    if not os.path.exists(model_path):
        print(f"⚠️ Model {model_path} tidak ditemukan! Harap jalankan training terlebih dahulu.")
        continue
        
    val_datagen = ImageDataGenerator(preprocessing_function=cfg['preprocess'])
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df, x_col='full_path', y_col='category_str', 
        target_size=cfg['input_size'], batch_size=BATCH_SIZE, 
        class_mode='sparse', shuffle=False
    )
    
    model = build_finetune_model(cfg, num_classes)
    model.load_weights(model_path)
    
    print(f"Memprediksi {len(val_df)} gambar validasi...")
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes
    
    # Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    report_path = os.path.join(EVAL_DIR, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report for {model_name.upper()}\n\n")
        f.write(report)
        
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(EVAL_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()
    plt.close()
    
    print(f"✅ Report disimpan di: {report_path}")
    print(f"✅ Confusion Matrix disimpan di: {cm_path}")
    
    del model, val_gen
    tf.keras.backend.clear_session()

print("\nEvaluasi klasifikasi selesai!")
