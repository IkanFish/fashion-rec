# Experiment Plan: Improved Attribute Filtering untuk Fashion Classification

**Konteks eksperimen:** DeepFashion / FashionAI attribute-based classification  
**Masalah yang diselesaikan:** Filtering atribut leaky yang saat ini hanya berbasis exact string match terlalu kasar dan tidak menangkap semantic leakage maupun distributional leakage  
**Target:** Menemukan strategi filtering yang lebih principled sebelum membangun representasi one-hot untuk perbandingan adil dengan model CNN  

---

## Latar Belakang

Filtering yang ada saat ini:

```python
LEAKY_KEYWORDS = {
    'jumpsuit', 'romper', 'sweatshirt', 'shorts',
    'pants', 'blouse', 'hoodie', 'dress',
}
```

Pendekatan ini hanya membuang atribut yang secara literal sama dengan nama kategori (**lexical leakage**). Dua jenis kebocoran lain tidak tertangkap:

- **Semantic leakage**: atribut yang tidak sama namanya tetapi secara makna hampir eksklusif menunjuk ke satu kategori (contoh: `zip fly` → Denim/Pants, `kangaroo pocket` → Sweatshirts_Hoodies)
- **Distributional leakage**: atribut yang secara statistik muncul jauh lebih sering di satu kategori dibanding kategori lain, meskipun secara semantik terlihat netral (contoh: `floral` bisa 80%+ muncul di Dresses di dataset tertentu)

Tanpa menangkap keduanya, representasi one-hot tetap "bocor" dan perbandingan dengan CNN tidak fair — model one-hot mendapat keuntungan shortcut informasi yang tidak dimiliki CNN.

---

## Tujuan Eksperimen

1. Mengidentifikasi atribut leaky secara **data-driven**, bukan hanya string matching
2. Menghasilkan filtered attribute set yang lebih bersih untuk representasi one-hot
3. Membandingkan kualitas representasi sebelum dan sesudah filtering baru terhadap baseline CNN
4. Memahami trade-off: filtering terlalu agresif → one-hot terlalu miskin informasi; filtering terlalu longgar → one-hot curang

---

## Fase 1 — Analisis Distribusi Atribut per Kategori

### Tujuan
Mendapatkan gambaran awal distribusi atribut sebelum memutuskan threshold filtering. Tanpa ini, kita blind terhadap skewness yang sebenarnya ada di data.

### Langkah

**1.1 Hitung conditional probability P(kategori | atribut)**

```python
import pandas as pd
import numpy as np

def compute_conditional_prob(df, attribute_cols, label_col):
    """
    df: DataFrame dengan kolom atribut (0/1) dan kolom label kategori
    Menghitung P(kategori | atribut = 1) untuk setiap pasangan atribut-kategori
    """
    results = {}
    categories = df[label_col].unique()
    
    for attr in attribute_cols:
        attr_present = df[df[attr] == 1]
        if len(attr_present) == 0:
            continue
        
        cat_dist = attr_present[label_col].value_counts(normalize=True)
        results[attr] = {
            'max_category': cat_dist.idxmax(),
            'max_prob': cat_dist.max(),
            'entropy': -sum(p * np.log2(p + 1e-9) for p in cat_dist.values),
            'n_samples': len(attr_present),
            'distribution': cat_dist.to_dict()
        }
    
    return pd.DataFrame(results).T

cond_prob_df = compute_conditional_prob(df, attribute_cols, 'category')
```

**1.2 Hitung specificity score per atribut**

```python
def specificity_score(cond_prob_df, n_categories=16):
    """
    Specificity = max P(c|a) / mean P(c|a)
    Nilai tinggi → atribut sangat spesifik ke satu kategori → kandidat leaky
    Nilai mendekati 1 → distribusi merata → atribut informatif tapi tidak leaky
    """
    uniform_prob = 1 / n_categories
    cond_prob_df['specificity'] = cond_prob_df['max_prob'] / uniform_prob
    return cond_prob_df.sort_values('specificity', ascending=False)
```

**1.3 Visualisasi heatmap distribusi**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Buat matrix (atribut × kategori)
heatmap_data = pd.DataFrame(
    {attr: data['distribution'] for attr, data in cond_prob_df['distribution'].items()}
).fillna(0).T

plt.figure(figsize=(20, 60))
sns.heatmap(heatmap_data, cmap='YlOrRd', vmin=0, vmax=1, annot=False)
plt.title('P(kategori | atribut) — semakin merah semakin leaky')
plt.savefig('attr_heatmap.png', dpi=150, bbox_inches='tight')
```

### Output yang diharapkan
- Tabel `cond_prob_df` dengan kolom: `max_category`, `max_prob`, `entropy`, `specificity`
- Heatmap visual untuk inspeksi manual
- Daftar atribut dengan `max_prob > 0.85` sebagai kandidat leaky pertama

---

## Fase 2 — Metode Filtering Berbasis Statistik

Tiga metode akan dijalankan secara **paralel** lalu hasilnya di-intersect dan di-union untuk membandingkan sensitivitasnya.

### Metode A: Threshold Conditional Probability

**Mengapa:** Metode paling langsung dan intuitif. Jika mengetahui satu atribut hampir selalu muncul bersama satu kategori, atribut itu essentially adalah proxy label.

**Cara kerja:** Atribut di-flag sebagai leaky jika `P(kategori_dominan | atribut = 1) > threshold`.

```python
def filter_by_conditional_prob(cond_prob_df, threshold=0.85):
    """
    Threshold 0.85 berarti: jika 85% item dengan atribut ini masuk ke
    satu kategori, atribut tersebut dianggap leaky.
    
    Pertimbangkan juga minimum sample size untuk menghindari false positive
    dari atribut yang sangat jarang muncul.
    """
    leaky = cond_prob_df[
        (cond_prob_df['max_prob'] > threshold) & 
        (cond_prob_df['n_samples'] >= 30)  # minimum support
    ].index.tolist()
    
    return leaky

# Eksperimen dengan beberapa threshold
for t in [0.75, 0.80, 0.85, 0.90]:
    leaky_attrs = filter_by_conditional_prob(cond_prob_df, threshold=t)
    print(f"Threshold {t}: {len(leaky_attrs)} atribut leaky")
    print(leaky_attrs[:10])
    print()
```

**Catatan penting:** Threshold 0.85 adalah titik awal yang wajar, tetapi perlu dikalibrasi. Jika dataset sangat imbalanced, atribut umum bisa terlihat leaky karena bias label, bukan karena benar-benar leaky.

---

### Metode B: Mutual Information (MI) dan Chi-Square

**Mengapa:** MI dan chi-square mengukur dependensi statistik antara kehadiran atribut dan label kategori. Berbeda dari conditional probability yang hanya melihat satu arah, MI mengukur seberapa besar informasi atribut mengurangi ketidakpastian label secara keseluruhan.

**Cara kerja:** Atribut dengan MI score tinggi adalah atribut yang paling "prediktif" terhadap label. Ini tidak selalu berarti leaky — atribut genuinely informatif juga punya MI tinggi. Oleh karena itu MI digunakan bukan sebagai filter tunggal, tetapi **dikombinasikan** dengan conditional probability.

```python
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(df['category'])
X_attrs = df[attribute_cols].values

# Mutual Information
mi_scores = mutual_info_classif(X_attrs, y_encoded, discrete_features=True, random_state=42)
mi_df = pd.Series(mi_scores, index=attribute_cols).sort_values(ascending=False)

# Chi-Square
chi2_scores, chi2_pvalues = chi2(X_attrs, y_encoded)
chi2_df = pd.Series(chi2_scores, index=attribute_cols).sort_values(ascending=False)

# Gabungkan dengan conditional probability
combined = cond_prob_df.copy()
combined['mi_score'] = mi_df
combined['chi2_score'] = chi2_df

# Flag leaky: MI tinggi DAN max_prob tinggi → genuinely leaky
# MI tinggi tapi max_prob rendah → informatif tapi tidak leaky (PERTAHANKAN)
combined['is_leaky_mi'] = (
    (combined['mi_score'] > combined['mi_score'].quantile(0.80)) &
    (combined['max_prob'] > 0.80)
)
```

**Interpretasi kombinasi MI + conditional prob:**

| MI score | max_prob | Interpretasi | Keputusan |
|---|---|---|---|
| Tinggi | Tinggi | Atribut = proxy label | **Filter (leaky)** |
| Tinggi | Rendah | Informatif, diskriminatif merata | **Pertahankan** |
| Rendah | Tinggi | Jarang muncul, kebetulan di satu kategori | Filter jika n_samples kecil |
| Rendah | Rendah | Noise / tidak informatif | Filter (irrelevant) |

---

### Metode C: Permutation Importance Post-hoc

**Mengapa:** Dua metode sebelumnya berbasis distribusi marginal atribut — mereka tidak memperhitungkan interaksi antar atribut. Permutation importance mengukur kontribusi atribut dalam konteks model yang sudah dilatih, sehingga lebih realistis.

**Cara kerja:** Train model one-hot baseline, lalu untuk setiap atribut, acak nilainya di test set dan ukur penurunan F1 per kategori. Atribut yang permulasinya menyebabkan penurunan besar pada **satu kategori saja** adalah leaky.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X_attrs, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Train model baseline (RF cukup untuk analisis, tidak perlu model akhir)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Permutation importance terhadap macro F1
perm_result = permutation_importance(
    clf, X_test, y_test,
    n_repeats=20,
    scoring='f1_macro',
    random_state=42
)

perm_df = pd.DataFrame({
    'attribute': attribute_cols,
    'importance_mean': perm_result.importances_mean,
    'importance_std': perm_result.importances_std
}).sort_values('importance_mean', ascending=False)

# ---- Versi per-kategori (lebih detail) ----
def per_category_permutation(clf, X_test, y_test, attribute_cols, n_repeats=10):
    """
    Untuk setiap atribut, ukur penurunan F1 per kategori saat atribut di-permutasi.
    Atribut leaky: penurunan besar di tepat satu kategori.
    """
    baseline_f1 = f1_score(y_test, clf.predict(X_test), average=None)
    results = {}
    
    for i, attr in enumerate(attribute_cols):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            permuted_f1 = f1_score(y_test, clf.predict(X_permuted), average=None)
            drops.append(baseline_f1 - permuted_f1)
        
        mean_drop = np.mean(drops, axis=0)
        results[attr] = {
            'max_drop': mean_drop.max(),
            'max_drop_category': le.classes_[mean_drop.argmax()],
            'drop_concentration': mean_drop.max() / (mean_drop.sum() + 1e-9)
            # drop_concentration tinggi → penurunan terpusat di 1 kategori → leaky
        }
    
    return pd.DataFrame(results).T

perm_per_cat = per_category_permutation(clf, X_test, y_test, attribute_cols)
```

**Threshold untuk Metode C:** Atribut dianggap leaky jika `drop_concentration > 0.7` dan `max_drop > 0.05`. Artinya: lebih dari 70% dampak permutasi terpusat di satu kategori, dan dampaknya signifikan.

---

## Fase 3 — Strategi Kombinasi dan Tiering

### Mengapa tidak cukup satu metode saja

Setiap metode memiliki sudut pandang berbeda:
- **Conditional probability** melihat dari sisi distribusi data mentah
- **Mutual information** melihat dari sisi information gain terhadap label
- **Permutation importance** melihat dari sisi kontribusi aktual dalam model

Menggunakan ketiganya bersama memberikan confidence yang lebih tinggi dalam keputusan filtering.

### Sistem tiering atribut

Daripada binary "hapus/pertahankan", klasifikasikan atribut ke dalam 4 tier:

```python
def assign_tier(row):
    """
    Tier 1 (REMOVE): Leaky dikonfirmasi oleh minimal 2 dari 3 metode
    Tier 2 (DOWNWEIGHT): Leaky menurut 1 metode, perlu investigasi manual
    Tier 3 (KEEP): Informatif, tidak leaky menurut semua metode
    Tier 4 (REVIEW): Skor rendah semua metode → kandidat noise/irrelevant
    """
    votes = sum([
        row['is_leaky_conditional'],  # dari Metode A
        row['is_leaky_mi'],           # dari Metode B
        row['is_leaky_permutation']   # dari Metode C
    ])
    
    if votes >= 2:
        return 'Tier1_Remove'
    elif votes == 1:
        return 'Tier2_Downweight'
    elif row['mi_score'] < mi_threshold_low and row['max_prob'] < 0.5:
        return 'Tier4_Review'
    else:
        return 'Tier3_Keep'

combined['tier'] = combined.apply(assign_tier, axis=1)

print(combined['tier'].value_counts())
print("\n--- Tier 1 (akan dihapus) ---")
print(combined[combined['tier'] == 'Tier1_Remove'].index.tolist())
```

### Variasi eksperimen yang akan dibandingkan

| Konfigurasi | Atribut yang digunakan | Tujuan |
|---|---|---|
| `baseline_old` | Filter string match saat ini | Titik acuan lama |
| `exp_tier1_only` | Hapus Tier 1 saja | Filtering konservatif |
| `exp_tier1_2` | Hapus Tier 1 dan 2 | Filtering moderat |
| `exp_tier1_4` | Hapus Tier 1 dan 4 (noise) | Filtering + denoising |
| `exp_all_tiers` | Hapus semua kecuali Tier 3 | Filtering paling agresif |

---

## Fase 4 — Evaluasi Fairness Representasi

### Tujuan
Memastikan filtering baru tidak **terlalu memiskinkan** representasi one-hot sehingga perbandingan dengan CNN menjadi tidak informatif.

### Metrik evaluasi yang digunakan

**4.1 F1-score per kategori (metric utama)**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def evaluate_onehot_config(X_filtered, y, config_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    report = classification_report(y_test, clf.predict(X_test), 
                                   target_names=le.classes_, output_dict=True)
    return pd.DataFrame(report).T['f1-score'], config_name
```

**4.2 Representation richness per kategori**

Setelah filtering, hitung rata-rata jumlah atribut aktif per item untuk setiap kategori. Distribusi yang terlalu timpang antar kategori menandakan filtering baru masih tidak adil.

```python
def representation_richness(X_filtered, y, label_encoder):
    df_temp = pd.DataFrame(X_filtered)
    df_temp['category'] = label_encoder.inverse_transform(y)
    
    richness = df_temp.groupby('category').mean().sum(axis=1)
    print("Rata-rata atribut aktif per kategori:")
    print(richness.sort_values())
    
    # Coefficient of variation — semakin kecil semakin seimbang
    cv = richness.std() / richness.mean()
    print(f"\nCoefficient of Variation: {cv:.3f} (target < 0.3)")
    return richness
```

**4.3 Intra-cluster vs inter-cluster distance**

Ukur apakah vektor atribut dari kategori yang sama lebih dekat satu sama lain dibanding dengan kategori berbeda (cluster quality).

```python
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_filtered, y_encoded, metric='cosine', sample_size=5000)
print(f"Silhouette score (cosine): {sil_score:.4f}")
# Semakin mendekati 1 → cluster kategori lebih tight dan separated
# Bandingkan silhouette score sebelum dan sesudah filtering
```

**4.4 Perbandingan akhir dengan CNN**

Setelah menemukan konfigurasi filtering terbaik, jalankan ulang perbandingan penuh:

```python
comparison_results = {}
for config_name, X_filtered in filtered_configs.items():
    f1_scores, _ = evaluate_onehot_config(X_filtered, y_encoded, config_name)
    comparison_results[config_name] = f1_scores

# Tambahkan hasil CNN sebagai kolom pembanding
comparison_df = pd.DataFrame(comparison_results)
comparison_df['VGG19'] = [0.9124, 0.6922, ...]  # dari hasil eksperimen sebelumnya
```

---

## Fase 5 — Analisis Kategori Kritis

Beberapa kategori memerlukan perhatian khusus karena karakteristiknya yang unik.

### Kategori yang perlu diperhatikan

**Denim** — sangat banyak atribut spesifik (`zip fly`, `five-pocket`, `skinny jean`, `boyfriend jean`, `denim washed`). Setelah filtering Tier 1, cek apakah kategori ini kehilangan terlalu banyak atribut sehingga representasinya kolaps.

**Rompers_Jumpsuits** — one-hot baseline sangat rendah (0.2847) bahkan sebelum filtering ketat. Perlu dianalisis apakah ini karena kurangnya atribut unik yang genuinely diskriminatif, atau karena atributnya sudah ikut terfilter di string match sebelumnya (`jumpsuit`, `romper`).

**Leggings** — score terendah di semua model. Mungkin perlu analisis terpisah apakah atribut yang tersisa setelah filtering masih cukup untuk membedakan Leggings dari Pants atau Shorts.

```python
# Analisis atribut per kategori kritis setelah filtering
critical_cats = ['Denim', 'Rompers_Jumpsuits', 'Leggings']

for cat in critical_cats:
    cat_items = df[df['category'] == cat]
    attr_frequency = cat_items[attribute_cols_filtered].mean().sort_values(ascending=False)
    
    print(f"\n=== {cat} ===")
    print(f"Jumlah atribut aktif rata-rata: {attr_frequency[attr_frequency > 0.1].sum():.2f}")
    print("Top 10 atribut:")
    print(attr_frequency.head(10))
```

---

## Fase 6 — Opsi Lanjutan (Opsional, jika waktu memungkinkan)

### 6.1 Soft weighting sebagai alternatif hard filtering

Daripada membuang atribut Tier 2, assign weight berdasarkan inverse specificity:

```python
def compute_attribute_weights(cond_prob_df, n_categories=16):
    """
    Atribut dengan distribusi merata mendapat weight 1.0
    Atribut yang sangat spesifik ke satu kategori mendapat weight mendekati 0
    """
    uniform = 1 / n_categories
    weights = {}
    
    for attr, row in cond_prob_df.iterrows():
        # Normalized entropy: 0 = semua di satu kategori, 1 = distribusi merata
        normalized_entropy = row['entropy'] / np.log2(n_categories)
        weights[attr] = normalized_entropy
    
    return weights

# Gunakan dalam representasi:
# X_weighted = X_onehot * np.array([weights.get(col, 1.0) for col in attribute_cols])
```

### 6.2 Hybrid late fusion

Jika filtering one-hot terbaik masih kalah dari CNN di beberapa kategori, pertimbangkan weighted ensemble:

```python
# Weight per kategori berdasarkan mana yang lebih unggul di validation set
def adaptive_fusion(onehot_probs, cnn_probs, category_weights):
    """
    Untuk setiap kategori, gunakan weight berbeda antara one-hot dan CNN
    berdasarkan performa di validation set
    """
    fused = {}
    for i, cat in enumerate(categories):
        w_oh = category_weights[cat]['onehot']
        w_cnn = category_weights[cat]['cnn']
        fused[cat] = w_oh * onehot_probs[:, i] + w_cnn * cnn_probs[:, i]
    return fused
```

---

## Checklist Eksperimen

### Fase 1
- [ ] Load dataset dan buat matrix atribut × label
- [ ] Jalankan `compute_conditional_prob()` untuk semua atribut
- [ ] Generate heatmap dan inspeksi visual
- [ ] Identifikasi kandidat leaky awal (`max_prob > 0.85`)

### Fase 2
- [ ] Jalankan Metode A (conditional probability) dengan threshold 0.75, 0.80, 0.85, 0.90
- [ ] Jalankan Metode B (MI + chi-square) dan buat tabel kombinasi
- [ ] Jalankan Metode C (permutation importance) — paling lambat, jalankan terakhir
- [ ] Catat jumlah atribut yang di-flag oleh setiap metode

### Fase 3
- [ ] Buat tabel `combined` dengan kolom tier untuk semua atribut
- [ ] Review manual atribut Tier 2 (grey area)
- [ ] Generate 5 konfigurasi filtered attribute set

### Fase 4
- [ ] Jalankan `evaluate_onehot_config()` untuk semua konfigurasi
- [ ] Hitung `representation_richness` per konfigurasi
- [ ] Hitung silhouette score per konfigurasi
- [ ] Bandingkan F1 per kategori dengan baseline CNN

### Fase 5
- [ ] Analisis mendalam untuk Denim, Rompers_Jumpsuits, Leggings
- [ ] Pastikan tidak ada kategori yang collapse ke < 5 atribut aktif rata-rata

### Fase 6 (opsional)
- [ ] Implementasi soft weighting
- [ ] Eksperimen late fusion jika ada kategori yang konsisten lebih baik di salah satu model

---

## Expected Outcome

Setelah eksperimen ini selesai, kita akan memiliki:

1. **Filtered attribute set yang lebih principled** — bukan hanya menghapus 8 kata, tetapi berdasarkan bukti statistik dari data aktual
2. **Pemahaman yang lebih jelas** mengapa kategori tertentu lebih cocok direpresentasikan dengan atribut teks vs. fitur visual CNN
3. **Perbandingan one-hot vs CNN yang lebih adil** — tanpa shortcut informasi yang menguntungkan salah satu secara artifisial
4. **Baseline untuk eksplorasi hybrid model** jika diperlukan sebagai eksperimen lanjutan

---

*Dokumen ini dibuat berdasarkan analisis eksperimen klasifikasi fashion menggunakan dataset DeepFashion dengan 16 kategori. Revisi threshold dan parameter disesuaikan dengan distribusi aktual dataset setelah Fase 1 selesai.*
