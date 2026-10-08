"""
Prepare lightweight deployment data for Streamlit Community Cloud.

Creates:
- data/deploy_images.zip  — all 7975 images resized to 200x200
- data/deploy_features.zip — CNN + text feature matrices
- data/deploy_meta.zip    — master_dataset.csv
"""
import os
import sys
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'dataset'
FEATURES_DIR = BASE_DIR / 'features'
TEXT_EVAL_DIR = BASE_DIR / 'baseline_text_cbf' / 'evaluation'
IMG_ROOT = DATASET_DIR / 'In-shop Clothes Retrieval Benchmark' / 'Img'
MASTER_CSV = DATASET_DIR / 'master_dataset.csv'

OUTPUT_DIR = BASE_DIR / 'data'
TARGET_SIZE = (200, 200)
JPEG_QUALITY = 80


def prepare_images(df: pd.DataFrame) -> Path:
    """Resize all referenced images to 200x200 and create zip."""
    zip_path = OUTPUT_DIR / 'deploy_images.zip'
    if zip_path.exists():
        print(f"[SKIP] {zip_path.name} already exists")
        return zip_path

    print(f"Resizing {len(df)} images to {TARGET_SIZE}...")
    count = 0
    errors = 0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for _, row in df.iterrows():
            img_rel = row['image_name']  # e.g., img/WOMEN/...
            # Strip 'img/' prefix and resolve from IMG_ROOT
            img_path = IMG_ROOT / img_rel.replace('img/', '', 1)

            if not img_path.exists():
                errors += 1
                continue

            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(TARGET_SIZE, Image.LANCZOS)

                # Save to zip with same relative path
                arcname = img_rel  # keep img/WOMEN/... structure

                # Write to temp buffer then add to zip
                import io
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=JPEG_QUALITY)
                buf.seek(0)

                # Use a .jpg extension in archive
                arcname_jpg = arcname.rsplit('.', 1)[0] + '.jpg'
                zf.writestr(arcname_jpg, buf.read())
                count += 1

                if count % 500 == 0:
                    print(f"  ... {count}/{len(df)} done")
            except Exception as e:
                errors += 1

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[DONE] {zip_path.name}: {count} images, {errors} errors, {size_mb:.1f} MB")
    return zip_path


def prepare_features() -> Path:
    """Bundle feature matrices into zip."""
    zip_path = OUTPUT_DIR / 'deploy_features.zip'
    if zip_path.exists():
        print(f"[SKIP] {zip_path.name} already exists")
        return zip_path

    files = {
        'vgg19_features_exp3.npy': FEATURES_DIR / 'exp3_partial_unfreeze' / 'vgg19_features_exp3.npy',
        'onehot_filtered_matrix.npy': TEXT_EVAL_DIR / 'onehot_filtered_matrix.npy',
        'onehot_item_index.csv': TEXT_EVAL_DIR / 'onehot_item_index.csv',
    }

    print(f"Bundling {len(files)} feature files...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in files.items():
            if path.exists():
                zf.write(path, arcname)
                print(f"  + {arcname} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
            else:
                print(f"  ! MISSING: {path}")

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[DONE] {zip_path.name}: {size_mb:.1f} MB")
    return zip_path


def prepare_meta(df: pd.DataFrame) -> Path:
    """Bundle metadata CSV."""
    zip_path = OUTPUT_DIR / 'deploy_meta.zip'
    if zip_path.exists():
        print(f"[SKIP] {zip_path.name} already exists")
        return zip_path

    print("Bundling metadata...")
    meta_csv = OUTPUT_DIR / 'master_dataset.csv'
    df.to_csv(meta_csv, index=False)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(meta_csv, 'master_dataset.csv')

    meta_csv.unlink()  # clean temp
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[DONE] {zip_path.name}: {size_mb:.1f} MB")
    return zip_path


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Deployment Data Preparation")
    print("=" * 60)

    df = pd.read_csv(MASTER_CSV)
    print(f"Dataset: {len(df)} items")

    prepare_meta(df)
    prepare_features()
    prepare_images(df)

    print("\n" + "=" * 60)
    print("  All archives ready in data/")
    print("  Upload these to a GitHub Release:")
    print("    - deploy_images.zip")
    print("    - deploy_features.zip")
    print("    - deploy_meta.zip")
    print("=" * 60)
