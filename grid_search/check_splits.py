import pandas as pd

df = pd.read_csv("dataset/full_dataset.csv")

print("=== DISTRIBUSI SPLIT ===")
print(df["split"].value_counts())
print("Total gambar:", len(df))
print("Total item unik:", df["item_id"].nunique())
print("Total kategori:", df["category"].nunique())

df_t = df[df["split"] == "train"]
print("\n=== DETAIL TRAIN ===")
print("Train gambar:", len(df_t))
print("Train item unik:", df_t["item_id"].nunique())
print("Train kategori:", df_t["category"].nunique())

n80 = int(len(df_t) * 0.8)
n20 = len(df_t) - n80
print("\nDengan split 80/20:")
print("  Aktual train:", n80)
print("  Aktual val  :", n20)

df_q = df[df["split"] == "query"]
df_g = df[df["split"] == "gallery"]
print("\nQuery gambar:", len(df_q))
print("Gallery gambar:", len(df_g))
print("\nTrain persen:", round(len(df_t)/len(df)*100, 1))
print("Query persen:", round(len(df_q)/len(df)*100, 1))
print("Gallery persen:", round(len(df_g)/len(df)*100, 1))
