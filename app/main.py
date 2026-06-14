# ─────────────────────────────────────────────────────────────
#   main.py — User Study App (A/B Evaluation)
#   Fashion Recommendation: CNN vs Text-based
#   Run: streamlit run app/main.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import uuid
import random
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    IMG_ROOT, COL_PATH, COL_CATEGORY,
    COLD_START_K, REC_TOP_N, IMG_DISPLAY_SIZE, SHEET_NAME,
    IS_DEV_ENV
)
from engine import DualRecommenderSystem
from sheets import submit_to_sheets, build_submission_row
from data_loader import ensure_data_ready

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Evaluasi Sistem Rekomendasi Fashion",
    page_icon="👗",
    layout="centered",          # mobile-friendly
    initial_sidebar_state="collapsed",
)

# Invisible marker at the absolute top of the app
st.markdown("<div id='top-marker'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  DATA CHECK (downloads from GitHub Releases on first cloud run)
# ══════════════════════════════════════════════
ensure_data_ready()

# ══════════════════════════════════════════════
#  PREVENT DOUBLE SUBMISSION (Cookie / LocalStorage)
# ══════════════════════════════════════════════
if 'cookie_checked' not in st.session_state:
    completed_status = streamlit_js_eval(js_expressions="window.localStorage.getItem('fashion_study_completed') || 'false'", key="check_completed")
    
    if completed_status is None:
        st.markdown("<h3 style='text-align: center; margin-top: 100px; font-weight: normal;'>Memuat sesi... ⏳</h3>", unsafe_allow_html=True)
        st.stop()
    else:
        st.session_state.cookie_checked = True
        st.session_state.has_completed = (completed_status == 'true')
        st.rerun()

if st.session_state.get('has_completed', False):
    st.error("⚠️ Anda sudah pernah mengisi survei ini sebelumnya. Terima kasih banyak atas partisipasi Anda!")
    st.stop()

# ══════════════════════════════════════════════
#  CSS — Mobile-first, clean questionnaire style
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, header, footer { visibility: hidden; }

/* Hide sidebar completely for user study */
section[data-testid="stSidebar"] { display: none; }

/* Progress indicator */
.step-indicator {
    display: flex; justify-content: center; gap: 8px;
    margin-bottom: 1.5rem;
}
.step-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #e0e0e0;
}
.step-dot.active { background: #2575fc; }
.step-dot.done { background: #00c853; }

/* Card style for recommendation columns */
.rec-column {
    background: #f8f9fa; border: 2px solid #e9ecef;
    border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
}
.rec-column-title {
    font-size: 1.1rem; font-weight: 700; text-align: center;
    margin-bottom: 0.8rem; color: #333;
}

/* Item card */
.item-card {
    text-align: center; margin-bottom: 0.5rem;
    background: white; border-radius: 8px; padding: 4px;
    border: 1px solid #eee;
}
.item-label {
    font-size: 0.7rem; color: #666; margin-top: 2px;
}

/* Preference buttons */
div[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Info/warning boxes */
.info-box {
    background: #e3f2fd; border-left: 4px solid #2575fc;
    border-radius: 8px; padding: 0.8rem 1rem;
    color: #1a237e; font-size: 0.88rem; margin: 0.8rem 0;
}

/* Checkbox custom styling */
div[data-testid="stCheckbox"] {
    border-radius: 8px !important;
    padding: 4px 8px !important;
    margin-top: 2px !important;
}
/* Responsive item grid: 4 cols on PC, 2 cols on mobile */
.item-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 0.5rem;
}
@media (max-width: 768px) {
    .item-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 6px;
    }
}

/* Grid cell styling */
.grid-cell {
    text-align: center;
    background: white;
    border-radius: 8px;
    border: 1px solid #eee;
    padding: 4px;
    overflow: hidden;
}
.grid-cell img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
}

/* Mobile overrides */
@media (max-width: 768px) {
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
    }

    /* Prevent Streamlit from stacking columns vertically, but allow wrap for 3+ columns */
    div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 2% !important; /* Use percentage gap for safer fluid layout */
    }
    
    /* Force immediate children of horizontal block to split evenly */
    div[data-testid="stHorizontalBlock"] > div {
        width: 49% !important;
        flex: 0 0 49% !important;
        min-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  IMAGE LOADER
# ══════════════════════════════════════════════
def resolve_img_path(relative_path: str) -> str:
    """Convert relative path from CSV to absolute path.
    Tries original extension first, then .jpg fallback (for deployment)."""
    p = str(relative_path).replace('\\', '/')
    if p.startswith('img/'):
        p = p[4:]
    full = os.path.join(IMG_ROOT, p)
    # Deployment: all images are resized .jpg — try that if original not found
    if not os.path.exists(full):
        jpg_path = os.path.splitext(full)[0] + '.jpg'
        if os.path.exists(jpg_path):
            return jpg_path
    return full


@st.cache_data(show_spinner=False)
def load_image_cached(path: str, size_w: int = 200, size_h: int = 200) -> Image.Image:
    """Load and cache image. Returns placeholder on failure."""
    size = (size_w, size_h)
    try:
        full_path = resolve_img_path(path)
        if os.path.exists(full_path):
            img = Image.open(full_path).convert('RGB')
            img.thumbnail(size, Image.LANCZOS)
            square = Image.new('RGB', size, (245, 245, 245))
            offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
            square.paste(img, offset)
            return square
    except Exception:
        pass
    return Image.new('RGB', size, (230, 230, 230))


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 JPEG string."""
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _build_cell_html(img_path: str, category: str, number: int = 0) -> str:
    """Build HTML for a single grid cell (image + category label)."""
    img = load_image_cached(img_path, *IMG_DISPLAY_SIZE)
    b64 = pil_to_base64(img)
    cat_display = category.replace('_', ' ').title() if category else ''
    badge = (
        f'<div style="position:absolute;top:6px;left:6px;background:rgba(0,0,0,0.55);'
        f'color:#fff;border-radius:50%;width:24px;height:24px;display:flex;'
        f'align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;">'
        f'{number}</div>'
    ) if number > 0 else ''
    return (
        f'<div style="position:relative;text-align:center;background:#fff;'
        f'border-radius:10px;border:1px solid #eee;padding:6px;overflow:hidden;">'
        f'{badge}'
        f'<img src="data:image/jpeg;base64,{b64}" '
        f'style="width:100%;height:auto;border-radius:6px;display:block;">'
        f'<div style="font-size:0.72rem;color:#666;margin-top:3px;">{cat_display}</div>'
        f'</div>'
    )


def render_single_html_card(img_path: str, category: str, number: int = 0):
    """Render exactly 1 image card with HTML (for borders/badges)."""
    html = _build_cell_html(img_path, category, number)
    # wrap in a div with bottom margin so it doesn't hug the checkbox too tightly
    st.markdown(f'<div style="margin-bottom: 0.2rem;">{html}</div>', unsafe_allow_html=True)


def render_item(image_path: str, category: str = '', show_checkbox: bool = False,
                cb_key: str = '', cb_label: str = 'Pilih'):
    """Render a single item card with optional checkbox (legacy, for non-grid use)."""
    img = load_image_cached(image_path, *IMG_DISPLAY_SIZE)
    st.image(img, use_container_width=True)
    if category:
        cat_display = category.replace('_', ' ').title()
        st.markdown(f"<div class='item-label'>{cat_display}</div>", unsafe_allow_html=True)
    if show_checkbox:
        return st.checkbox(cb_label, key=cb_key)
    return None


def render_step_indicator(current_step: int, total_steps: int = 5):
    """Render dot progress indicator."""
    dots = ''.join(
        f'<div class="step-dot {"active" if i == current_step else "done" if i < current_step else ""}"></div>'
        for i in range(1, total_steps + 1)
    )
    st.markdown(f'<div class="step-indicator">{dots}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
def init_state():
    defaults = {
        'phase'              : 'questionnaire',
        'previous_phase'     : 'questionnaire',
        'session_id'         : str(uuid.uuid4())[:8],
        # Phase 1: demographics
        'demo_age'           : '',
        'demo_gender'        : '',
        'demo_frequency'     : '',
        'demo_platforms'     : [],
        'demo_difficulty'    : '',
        # Phase 2: cold start
        'cold_start_df'      : None,
        'cold_start_indices' : [],   # row indices of cold start items
        'liked_cs'           : [],   # indices user liked
        # Phase 3: recommendations
        'recs_a_df'          : None,
        'recs_b_df'          : None,
        'recs_a_indices'     : [],
        'recs_b_indices'     : [],
        'ab_mapping'         : {},   # {'A': 'cnn', 'B': 'text'} or reversed
        # Phase 4: evaluation
        'preference'         : '',
        'liked_a'            : [],
        'liked_b'            : [],
        # Phase 5: submitted
        'submitted'          : False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def scroll_to_top():
    """Injects an iframe that accesses the parent DOM to scroll to the top marker."""
    js = '''
    <script>
        function forceScroll() {
            var p = window.parent;
            if (p) {
                // Target 1: The explicit marker
                var marker = p.document.getElementById('top-marker');
                if (marker) marker.scrollIntoView();
                
                // Target 2: The standard Streamlit view container
                var container = p.document.querySelector('[data-testid="stAppViewContainer"]');
                if (container) container.scrollTop = 0;
                
                // Target 3: Fallback window scroll
                p.scrollTo(0, 0);
            }
        }
        forceScroll();
        setTimeout(forceScroll, 100);
        setTimeout(forceScroll, 300);
        setTimeout(forceScroll, 800);
    </script>
    '''
    import streamlit.components.v1 as components
    components.html(js, height=0)

# Check if phase just changed to trigger scroll to top
if st.session_state.phase != st.session_state.previous_phase:
    st.session_state.previous_phase = st.session_state.phase
    scroll_to_top()


# ══════════════════════════════════════════════
#  LOAD RECOMMENDER (cached singleton)
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner="Memuat sistem rekomendasi...")
def get_recommender() -> DualRecommenderSystem:
    return DualRecommenderSystem()


# ══════════════════════════════════════════════
#  PHASE 1: QUESTIONNAIRE
# ══════════════════════════════════════════════
if st.session_state.phase == 'questionnaire':
    render_step_indicator(1)

    st.markdown("## 👗 Evaluasi Sistem Rekomendasi Fashion")
    st.markdown("""
    <div class='info-box'>
    Halo! Perkenalkan nama saya <strong>Ghani Naja Fadhilah</strong>, mahasiswa tingkat akhir yang sedang menyusun tugas akhir. Di sini Anda dapat membantu saya mengevaluasi sistem rekomendasi pakaian berbasis <em>machine learning</em>.<br><br>
    Proses ini hanya memakan waktu <strong>~2 menit</strong> saja! Tenang, saya hanya mengambil data yang Anda berikan pada halaman ini, dan menjamin data Anda murni hanya akan digunakan untuk kebutuhan akademis.<br><br>
    Terima kasih sudah meluangkan waktu untuk mengisi survei ini. Jawaban Anda sangat berharga!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Data Diri")

    col1, col2 = st.columns(2)
    with col1:
        age = st.selectbox(
            "Rentang Usia",
            options=['', '< 18', '18-21', '22-25', '26-30', '31-35', '> 35'],
            format_func=lambda x: '— Pilih —' if x == '' else x,
        )
    with col2:
        gender = st.radio(
            "Gender",
            options=['Pria', 'Wanita', 'Tidak menjawab'],
            index=None,
            horizontal=True,
        )

    st.markdown("---")
    st.markdown("### Kebiasaan Belanja Fashion")

    frequency = st.radio(
        "Seberapa sering Anda membeli pakaian/fashion secara online?",
        options=[
            'Tidak pernah membeli fashion online',
            '1-2 kali per bulan',
            '3-5 kali per bulan',
            'Lebih dari 5 kali per bulan',
        ],
        index=None,
    )

    platforms = st.multiselect(
        "Platform e-commerce yang pernah Anda gunakan *(opsional)*",
        options=['Shopee', 'Tokopedia', 'Lazada', 'TikTok Shop', 'Blibli', 'Zalora', 'Lainnya'],
        help="Pilih semua yang pernah Anda pakai. Boleh dikosongkan.",
    )

    difficulty = st.radio(
        "Pernah mengalami kesulitan mencari pakaian yang sesuai secara visual saat belanja online?",
        options=[
            'Sering — hampir selalu sulit menemukan yang cocok',
            'Kadang-kadang — terkadang sulit',
            'Jarang — biasanya mudah menemukan',
            'Tidak pernah — selalu mudah',
        ],
        index=None,
    )

    st.markdown("---")

    # Validation
    can_proceed = age != '' and gender != ''

    if st.button("Mulai Evaluasi →", disabled=not can_proceed, use_container_width=True):
        st.session_state.demo_age       = age
        st.session_state.demo_gender    = gender
        st.session_state.demo_frequency = frequency
        st.session_state.demo_platforms = platforms
        st.session_state.demo_difficulty = difficulty
        st.session_state.phase = 'cold_start'
        st.rerun()

    if not can_proceed:
        st.info("💡 Lengkapi rentang usia dan gender untuk melanjutkan.")

    st.stop()


# ══════════════════════════════════════════════
#  PHASE 2: COLD START
# ══════════════════════════════════════════════
if st.session_state.phase == 'cold_start':
    render_step_indicator(2)
    rec_sys = get_recommender()

    st.markdown("## 👋 Pilih Pakaian Favorit Anda")
    st.markdown("""
    <div class='info-box'>
    Pilihlah item pakaian di bawah ini yang sesuai dengan selera Anda. Sistem akan mempelajari selera Anda dari pakaian yang Anda sukai.<br>
    Anda juga dapat mengklik tombol <strong>'Acak Ulang'</strong> jika tidak ada item yang Anda sukai.<br><br>
    <strong>Pilih minimal 1 item</strong> yang Anda suka.
    </div>
    """, unsafe_allow_html=True)

    # Generate cold start items if needed
    if st.session_state.cold_start_df is None:
        cs_df = rec_sys.get_cold_start_items()
        st.session_state.cold_start_df = cs_df
        st.session_state.cold_start_indices = cs_df.index.tolist()

    cs_df = st.session_state.cold_start_df

    # Render using st.columns(2) PER PAIR. 
    # Ensures logical order (1, 2, 3, 4) even if mobile browser forces stacking.
    items_list = list(cs_df.iterrows())
    current_liked = set(st.session_state.liked_cs)

    for i in range(0, len(items_list), 2):
        pair = items_list[i:i+2]
        cols = st.columns(2 if len(pair) == 2 else 1)
        for col_idx, (col, (_, row)) in enumerate(zip(cols, pair)):
            with col:
                actual_idx = int(row.name)
                img_path = row.get(COL_PATH, '')
                cat = row.get(COL_CATEGORY, '')
                
                # number is i + col_idx + 1
                item_number = i + col_idx + 1
                render_single_html_card(img_path, cat, number=item_number)
                
                checked = st.checkbox("❤️ Suka", key=f"cs_{actual_idx}")
                if checked:
                    current_liked.add(actual_idx)
                else:
                    current_liked.discard(actual_idx)

    st.session_state.liked_cs = list(current_liked)

    st.markdown("---")
    n_liked = len(st.session_state.liked_cs)

    if n_liked > 0:
        st.success(f"✅ {n_liked} item dipilih")
    else:
        st.info("💡 Pilih minimal 1 item untuk melanjutkan.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if st.button("Lihat Rekomendasi →", disabled=(n_liked == 0),
                      use_container_width=True):
            # Build profiles for both engines
            rec_sys.build_profiles(st.session_state.liked_cs)

            # Get recommendations
            cnn_recs, text_recs = rec_sys.get_recommendations(st.session_state.liked_cs)

            # Randomize A/B assignment
            is_cnn_first = random.random() < 0.5
            if is_cnn_first:
                st.session_state.ab_mapping  = {'A': 'cnn', 'B': 'text'}
                st.session_state.recs_a_df   = cnn_recs
                st.session_state.recs_b_df   = text_recs
            else:
                st.session_state.ab_mapping  = {'A': 'text', 'B': 'cnn'}
                st.session_state.recs_a_df   = text_recs
                st.session_state.recs_b_df   = cnn_recs

            st.session_state.recs_a_indices = st.session_state.recs_a_df.index.tolist()
            st.session_state.recs_b_indices = st.session_state.recs_b_df.index.tolist()
            st.session_state.phase = 'recommendation'
            st.rerun()

    with col_b:
        if st.button("🔄 Acak Ulang", use_container_width=True):
            rec_sys.reset()
            st.session_state.cold_start_df     = None
            st.session_state.cold_start_indices = []
            st.session_state.liked_cs          = []
            st.rerun()

    st.stop()


# ══════════════════════════════════════════════
#  PHASE 3: A/B RECOMMENDATION
# ══════════════════════════════════════════════
if st.session_state.phase == 'recommendation':
    render_step_indicator(3)

    st.markdown("## 🛍️ Bandingkan Dua Rekomendasi")
    st.markdown("""
    <div class='info-box'>
    Sistem menghasilkan dua set rekomendasi berdasarkan selera visual Anda.<br>
    Lihat kedua set di bawah, lalu pilih mana yang <strong>lebih Anda sukai</strong>.
    </div>
    """, unsafe_allow_html=True)

    # ── Rekomendasi A ──────────
    st.markdown("### Rekomendasi A")
    a_df = st.session_state.recs_a_df
    if a_df is not None:
        a_items = list(a_df.iterrows())
        for i in range(0, len(a_items), 2):
            pair = a_items[i:i+2]
            cols = st.columns(2 if len(pair) == 2 else 1)
            for col, (_, row) in zip(cols, pair):
                with col:
                    render_single_html_card(row.get(COL_PATH, ''), row.get(COL_CATEGORY, ''))

    st.markdown("---")

    # ── Rekomendasi B ──────────
    st.markdown("### Rekomendasi B")
    b_df = st.session_state.recs_b_df
    if b_df is not None:
        b_items = list(b_df.iterrows())
        for i in range(0, len(b_items), 2):
            pair = b_items[i:i+2]
            cols = st.columns(2 if len(pair) == 2 else 1)
            for col, (_, row) in zip(cols, pair):
                with col:
                    render_single_html_card(row.get(COL_PATH, ''), row.get(COL_CATEGORY, ''))

    st.markdown("---")
    st.markdown("### Mana yang lebih Anda sukai?")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👈 Rekomendasi A", use_container_width=True):
            st.session_state.preference = 'A'
            st.session_state.phase = 'evaluation'
            st.rerun()
    with col2:
        if st.button("🤝 Sama saja", use_container_width=True):
            st.session_state.preference = 'equal'
            st.session_state.phase = 'evaluation'
            st.rerun()
    with col3:
        if st.button("👉 Rekomendasi B", use_container_width=True):
            st.session_state.preference = 'B'
            st.session_state.phase = 'evaluation'
            st.rerun()

    st.stop()


# ══════════════════════════════════════════════
#  PHASE 4: EVALUATION (mark liked items)
# ══════════════════════════════════════════════
if st.session_state.phase == 'evaluation':
    render_step_indicator(4)

    pref = st.session_state.preference
    pref_label = {
        'A': 'Rekomendasi A',
        'B': 'Rekomendasi B',
        'equal': 'Keduanya sama saja',
    }.get(pref, pref)

    st.markdown("## ✅ Tandai Item yang Anda Suka")
    st.markdown(f"""
    <div class='info-box'>
    Anda memilih: <strong>{pref_label}</strong>.<br>
    Sekarang tandai item yang <strong>benar-benar Anda sukai</strong>
    (yang akan Anda pakai/beli) dari kedua set. Boleh dikosongkan.
    </div>
    """, unsafe_allow_html=True)

    # ── Rekomendasi A ─
    st.markdown("### Rekomendasi A")
    a_df = st.session_state.recs_a_df
    liked_a = set(st.session_state.liked_a)

    if a_df is not None:
        a_items = list(a_df.iterrows())
        for i in range(0, len(a_items), 2):
            pair = a_items[i:i+2]
            cols = st.columns(2 if len(pair) == 2 else 1)
            for col_idx, (col, (_, row)) in enumerate(zip(cols, pair)):
                with col:
                    actual_idx = int(row.name)
                    item_number = i + col_idx + 1
                    render_single_html_card(row.get(COL_PATH, ''), row.get(COL_CATEGORY, ''), number=item_number)
                    
                    checked = st.checkbox("Suka", key=f"eval_a_{actual_idx}")
                    if checked:
                        liked_a.add(actual_idx)
                    else:
                        liked_a.discard(actual_idx)

    st.session_state.liked_a = list(liked_a)

    st.markdown("---")

    # ── Rekomendasi B ─
    st.markdown("### Rekomendasi B")
    b_df = st.session_state.recs_b_df
    liked_b = set(st.session_state.liked_b)

    if b_df is not None:
        b_items = list(b_df.iterrows())
        for i in range(0, len(b_items), 2):
            pair = b_items[i:i+2]
            cols = st.columns(2 if len(pair) == 2 else 1)
            for col_idx, (col, (_, row)) in enumerate(zip(cols, pair)):
                with col:
                    actual_idx = int(row.name)
                    item_number = i + col_idx + 1
                    render_single_html_card(row.get(COL_PATH, ''), row.get(COL_CATEGORY, ''), number=item_number)
                    
                    checked = st.checkbox("Suka", key=f"eval_b_{actual_idx}")
                    if checked:
                        liked_b.add(actual_idx)
                    else:
                        liked_b.discard(actual_idx)

    st.session_state.liked_b = list(liked_b)

    st.markdown("---")

    n_a = len(st.session_state.liked_a)
    n_b = len(st.session_state.liked_b)
    st.info(f"Rekomendasi A: {n_a} disukai | Rekomendasi B: {n_b} disukai")

    if st.button("Kirim Jawaban →", use_container_width=True):
        st.session_state.phase = 'submit'
        st.rerun()

    st.stop()


# ══════════════════════════════════════════════
#  PHASE 5: SUBMIT
# ══════════════════════════════════════════════
if st.session_state.phase == 'submit':
    render_step_indicator(5)

    st.markdown("## 📤 Mengirim Jawaban...")

    if not st.session_state.submitted:
        row = build_submission_row(
            session_id      = st.session_state.session_id,
            demographics    = {
                'age'       : st.session_state.demo_age,
                'gender'    : st.session_state.demo_gender,
                'frequency' : st.session_state.demo_frequency,
                'platforms' : ', '.join(st.session_state.demo_platforms),
                'difficulty': st.session_state.demo_difficulty,
            },
            cold_start_items = st.session_state.cold_start_indices,
            liked_cold_start = st.session_state.liked_cs,
            ab_mapping       = st.session_state.ab_mapping,
            recs_a_items     = st.session_state.recs_a_indices,
            recs_b_items     = st.session_state.recs_b_indices,
            preference       = st.session_state.preference,
            liked_a_items    = st.session_state.liked_a,
            liked_b_items    = st.session_state.liked_b,
        )

        success = submit_to_sheets(row, SHEET_NAME)
        st.session_state.submitted = True

        if success:
            st.session_state.phase = 'done'
            st.rerun()
        else:
            st.error("Gagal mengirim. Coba lagi.")
            if st.button("Coba Lagi"):
                st.session_state.submitted = False
                st.rerun()
    else:
        st.session_state.phase = 'done'
        st.rerun()

    st.stop()


# ══════════════════════════════════════════════
#  PHASE 6: DONE
# ══════════════════════════════════════════════
if st.session_state.phase == 'done':
    # Write cookie to prevent double submission
    streamlit_js_eval(js_expressions="window.localStorage.setItem('fashion_study_completed', 'true')", key="set_completed")

    st.markdown("---")
    st.markdown("")
    st.markdown("## 🎉 Terima Kasih!")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <p style='font-size: 1.1rem; color: #333;'>
            Jawaban Anda telah berhasil dikirim dan sangat membantu penelitian ini.
        </p>
        <p style='font-size: 0.9rem; color: #666;'>
            Anda dapat menutup halaman ini.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if IS_DEV_ENV:
        with st.expander("🔍 Detail Sesi (Debug)"):
            st.json({
                'session_id' : st.session_state.session_id,
                'demographics': {
                    'age': st.session_state.demo_age,
                    'gender': st.session_state.demo_gender,
                    'frequency': st.session_state.demo_frequency,
                },
                'ab_mapping' : st.session_state.ab_mapping,
                'preference' : st.session_state.preference,
                'n_liked_cs' : len(st.session_state.liked_cs),
                'n_liked_a'  : len(st.session_state.liked_a),
                'n_liked_b'  : len(st.session_state.liked_b),
            })


