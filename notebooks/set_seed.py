import json

fpath = r'd:\Antigravity\Visual Based Rekomender Sistem\notebooks\08_category_nn_visualization.ipynb'

with open(fpath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'seed=42' in line:
                source[i] = line.replace('seed=42', 'seed=265')

with open(fpath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Seed successfully changed to 265")
