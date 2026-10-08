import json
import os

fpath = r'd:\Antigravity\Visual Based Rekomender Sistem\notebooks\08_category_nn_visualization.ipynb'

with open(fpath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

correct_source = """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
BASE_DIR = os.getcwd()
if os.path.basename(BASE_DIR) == 'notebooks':
    ROOT_DIR = os.path.dirname(BASE_DIR) # Naik 1 tingkat jika di dalam folder notebooks
else:
    ROOT_DIR = BASE_DIR                  # Tetap di sini jika sudah di folder utama
    
ANNO_DIR   = os.path.join(ROOT_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Anno')
EVAL_DIR   = os.path.join(ROOT_DIR, 'evaluation', 'vector_space_comparison')
IMG_ROOT   = os.path.join(ROOT_DIR, 'dataset', 'In-shop Clothes Retrieval Benchmark', 'Img')
NN_VIZ_DIR = os.path.join(EVAL_DIR, 'nn_viz')
os.makedirs(NN_VIZ_DIR, exist_ok=True)

TARGET_CATEGORIES = [
    'Denim',
    'Dresses'
]
"""
nb['cells'][1]['source'] = [line + '\n' for line in correct_source.split('\n')[:-1]]

with open(fpath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
