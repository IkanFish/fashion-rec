from PIL import Image, ImageChops
import os

images = [
    "nn_One-Hot_Dresses_k3.png",
    "nn_VGG19_Exp3_Dresses_k3.png",
    "nn_One-Hot_Denim_k3.png",
    "nn_VGG19_Exp3_Denim_k3.png"
]

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

for img_name in images:
    path = os.path.join(r"d:\Antigravity\Visual Based Rekomender Sistem\evaluation\vector_space_comparison\nn_viz", img_name)
    if os.path.exists(path):
        im = Image.open(path)
        im_trimmed = trim(im)
        im_trimmed.save(path)
        print(f"Trimmed {img_name} in evaluation dir")
        
    # Also trim the ones copied to Conference dir if they exist
    conf_path = os.path.join(r"d:\Antigravity\Visual Based Rekomender Sistem\skripsi\Conference", img_name)
    if os.path.exists(conf_path):
        im = Image.open(conf_path)
        im_trimmed = trim(im)
        im_trimmed.save(conf_path)
        print(f"Trimmed {img_name} in Conference dir")
