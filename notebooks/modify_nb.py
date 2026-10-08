import json
import os

fpath = r'd:\Antigravity\Visual Based Rekomender Sistem\notebooks\08_category_nn_visualization.ipynb'

with open(fpath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        src = src.replace('k=5', 'k=3')
        if 'TARGET_CATEGORIES = [' in src:
            src = "TARGET_CATEGORIES = ['Denim', 'Dresses']\n"
        
        # reconstruct lines
        lines = [line + '\n' for line in src.split('\n')]
        # remove trailing extra newline if split created an empty last line
        if lines[-1] == '\n':
            lines = lines[:-1]
        
        cell['source'] = lines

with open(fpath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
