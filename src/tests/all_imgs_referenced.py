import os
import re

os.chdir(os.path.join(__file__, "..", "..", "..", "latex", "2. thesis"))
thesis_text = open("thesis-en.tex").read()
label_regex = r"\\label\{([^\}]*)\}"
ref_regex = r"\\ref\{([^\}]*)\}"
labels = re.findall(label_regex, thesis_text)
refs = re.findall(ref_regex, thesis_text)
print(set(labels) - set(refs))
