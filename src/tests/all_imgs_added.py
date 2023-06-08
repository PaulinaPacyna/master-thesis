import glob
import os

os.chdir(os.path.join(__file__, "..", "..", "..", "latex", "2. thesis"))
all_images = glob.glob("imgs/**/**.png", recursive=True)
thesis_text = open("thesis-en.tex").read()
for image in all_images:
    image = image.replace("\\", "/")
    if image not in thesis_text:
        print(image, "missing")
