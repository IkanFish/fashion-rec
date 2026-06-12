"""
Data loader for Streamlit Community Cloud deployment.

Downloads and extracts data archives from GitHub Releases at first startup.
On subsequent runs, uses cached data.
"""
import os
import zipfile
import tempfile
import requests
import streamlit as st
from pathlib import Path

# GitHub Release asset URLs — update these after creating the release
# Format: https://github.com/OWNER/REPO/releases/download/TAG/FILENAME
_RELEASE_BASE = "https://github.com/{owner}/{repo}/releases/download/{tag}/{filename}"

# Data extraction targets relative to BASE_DIR
_EXTRACT_MAP = {
    'deploy_images.zip':  '.',          # extracts to dataset/In-shop.../Img/...
    'deploy_features.zip': 'features',  # extracts to features/
    'deploy_meta.zip':     'dataset',    # extracts to dataset/master_dataset.csv
}


def _get_base_dir() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def _get_release_url(filename: str) -> str:
    """Build download URL from Streamlit secrets or environment."""
    try:
        owner = st.secrets["github"]["owner"]
        repo = st.secrets["github"]["repo"]
        tag = st.secrets["github"]["release_tag"]
    except KeyError:
        # Fallback to environment variables
        owner = os.environ.get("GITHUB_OWNER", "")
        repo = os.environ.get("GITHUB_REPO", "")
        tag = os.environ.get("GITHUB_RELEASE_TAG", "")

    if not all([owner, repo, tag]):
        return ""

    return _RELEASE_BASE.format(owner=owner, repo=repo, tag=tag, filename=filename)


def _download_and_extract(url: str, extract_to: Path, zip_name: str) -> bool:
    """Download a zip from URL and extract to target directory."""
    st.info(f"Downloading {zip_name}... (this may take a minute)")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        # Get total size for progress
        total = int(response.headers.get('content-length', 0))
        downloaded = 0

        # Save to system temp file (avoids directory creation issues)
        fd, temp_str = tempfile.mkstemp(suffix='.zip')
        temp_path = Path(temp_str)

        progress_bar = st.progress(0)

        with os.fdopen(fd, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    progress_bar.progress(min(downloaded / total, 1.0))

        progress_bar.empty()

        # Ensure extract target directory exists
        extract_to.mkdir(parents=True, exist_ok=True)

        # Extract
        st.info(f"Extracting {zip_name}...")
        with zipfile.ZipFile(temp_path, 'r') as zf:
            zf.extractall(extract_to)

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

        st.success(f"{zip_name} ready!")
        return True

    except Exception as e:
        st.error(f"Failed to download {zip_name}: {e}")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False


def ensure_data_ready():
    """
    Check if all required data files exist. If not, download from GitHub Releases.
    Call this once at app startup before loading features/images.
    Does NOT use st.rerun() — continues execution after download completes.
    """
    base = _get_base_dir()

    # Define required files and their zip sources
    required_files = [
        (base / 'dataset' / 'master_dataset.csv', 'deploy_meta.zip', base / 'dataset'),
        (base / 'features' / 'vgg19_features_exp3.npy', 'deploy_features.zip', base / 'features'),
        (base / 'features' / 'onehot_filtered_matrix.npy', 'deploy_features.zip', base / 'features'),
    ]

    # Check if we need to download anything
    missing_zips = set()
    for check_path, zip_name, extract_to in required_files:
        if not check_path.exists():
            missing_zips.add((zip_name, extract_to))

    # Check images directory
    img_dir = base / 'dataset' / 'In-shop Clothes Retrieval Benchmark' / 'Img'
    if not img_dir.exists() or not any(img_dir.rglob('*.jpg')):
        missing_zips.add(('deploy_images.zip', base))

    if not missing_zips:
        return  # All data present

    # Need to download — show progress UI
    progress_container = st.container()
    with progress_container:
        st.markdown("## ⏳ Preparing Data")
        st.markdown("First-time setup: downloading data files...")

        for zip_name, extract_to in missing_zips:
            url = _get_release_url(zip_name)
            if not url:
                st.error(
                    "GitHub Release configuration missing. "
                    "Add `[github]` section to `.streamlit/secrets.toml` "
                    "with `owner`, `repo`, and `release_tag`."
                )
                st.stop()

            success = _download_and_extract(url, extract_to, zip_name)
            if not success:
                st.stop()

        st.markdown("---")
        st.success("All data ready! Loading app...")

    # Verify files exist after extraction
    for check_path, _, _ in required_files:
        if not check_path.exists():
            st.error(f"Data verification failed: {check_path} not found after extraction")
            st.stop()

    if not img_dir.exists() or not any(img_dir.rglob('*.jpg')):
        st.error("Image verification failed: no .jpg files found after extraction")
        st.stop()

    # Clear progress UI and continue (no rerun needed)
    progress_container.empty()

