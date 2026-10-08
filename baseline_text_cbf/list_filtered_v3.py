"""
List semua atribut yang AKAN difilter di v3.
Threshold: atribut single-word yang >70% concentrated + 
           atribut multi-word yang mengandung kata tersebut.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTR_DIR = os.path.join(ROOT, 'Anno', 'attributes')

# 8 kata kunci yang >70% concentrated di 1 kategori
LEAKY_KEYWORDS = {
    'jumpsuit',   # 100.0% Rompers_Jumpsuits
    'romper',     # 97.3%  Rompers_Jumpsuits
    'sweatshirt', # 79.1%  Sweatshirts_Hoodies
    'shorts',     # 78.0%  Shorts
    'pants',      # 76.9%  Pants
    'blouse',     # 76.3%  Blouses_Shirts
    'hoodie',     # 74.8%  Sweatshirts_Hoodies
    'dress',      # 70.7%  Dresses
}

# Load ALL attribute names (full name, not just first word)
all_attrs = []
with open(os.path.join(ATTR_DIR, 'list_attr_cloth.txt')) as f:
    f.readline(); f.readline()
    for line in f:
        if line.strip():
            all_attrs.append(line.strip())

# Find all attributes to filter
print("=" * 70)
print("  ATRIBUT YANG AKAN DIFILTER DI v3")
print("  Kriteria: mengandung kata dengan >70% konsentrasi di 1 kategori")
print("=" * 70)

filtered = []
kept = []

for i, attr in enumerate(all_attrs):
    words = set(attr.lower().replace('-', ' ').replace('_', ' ').split())
    overlap = words & LEAKY_KEYWORDS
    if overlap:
        filtered.append((i+1, attr, overlap))  # i+1 = line number in file (1-indexed from line 3)
    else:
        kept.append(attr)

print(f"\n  DIFILTER ({len(filtered)} atribut):")
print(f"  {'No':<5} {'Line':<6} {'Atribut':<35} {'Kata Bocor'}")
print(f"  {'─'*75}")

for idx, (line, attr, overlap) in enumerate(filtered, 1):
    print(f"  {idx:<5} {line:<6} {attr:<35} {overlap}")

print(f"\n  TOTAL: {len(filtered)} / {len(all_attrs)} atribut difilter")
print(f"  TERSISA: {len(kept)} atribut yang digunakan")
