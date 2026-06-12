# ─────────────────────────────────────────────────────────────
#   config.py — User Study App Configuration
#   Fashion Recommendation A/B Evaluation
# ─────────────────────────────────────────────────────────────
import os

# ── Root Paths ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
TEXT_EVAL_DIR = os.path.join(BASE_DIR, 'baseline_text_cbf', 'evaluation')

# ── Dataset ───────────────────────────────────────────────────
MASTER_CSV = os.path.join(DATASET_DIR, 'master_dataset.csv')
IMG_ROOT   = os.path.join(DATASET_DIR, 'In-shop Clothes Retrieval Benchmark', 'Img')

# ── Feature Files (auto-detect local dev vs cloud deployment) ──
# CNN: VGG19 Experiment 3 (Partial Unfreeze) — best performing model
_cnn_deploy = os.path.join(FEATURES_DIR, 'vgg19_features_exp3.npy')
_cnn_dev    = os.path.join(FEATURES_DIR, 'exp3_partial_unfreeze', 'vgg19_features_exp3.npy')
CNN_FEATURE_PATH = _cnn_deploy if os.path.exists(_cnn_deploy) else _cnn_dev

# Text: One-Hot Filtered — leaky attributes removed
_text_deploy = os.path.join(FEATURES_DIR, 'onehot_filtered_matrix.npy')
_text_dev    = os.path.join(TEXT_EVAL_DIR, 'onehot_filtered_matrix.npy')
TEXT_FEATURE_PATH = _text_deploy if os.path.exists(_text_deploy) else _text_dev

# ── DataFrame Columns ─────────────────────────────────────────
COL_PATH     = 'image_name'   # relative image path
COL_ITEM_ID  = 'item_id'
COL_CATEGORY = 'category'

# ── Recommendation Parameters ─────────────────────────────────
COLD_START_K   = 8    # K-Means clusters → 8 cold start items (5 women + 3 men)
REC_TOP_N      = 8    # items per recommendation set
RANDOM_STATE   = 42

# ── Image Display ─────────────────────────────────────────────
IMG_DISPLAY_SIZE = (200, 200)

# ── Google Sheets ─────────────────────────────────────────────
# Set to your Google Sheet name. Credentials via Streamlit secrets.
SHEET_NAME = 'fashion_user_study'
