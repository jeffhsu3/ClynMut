import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


FINDLEY_URL = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx"
findley = pd.read_excel(FINDLEY_URL, skiprows=2)
findley = findley.loc[findley.protein_variant.notna(),:]
findley["Protein_Annotation"] = [i.lstrip("p.") for i in findley.protein_variant]

pos_sre = re.compile("([0-9]+")

ERWOOD_URL = "https://www.biorxiv.org/content/biorxiv/early/2021/05/14/2021.05.11.443710/DC2/embed/media-2.xlsx?download=true"
erwood = pd.read_excel(ERWOOD_URL)

overlap_col = "Protein_Annotation"
erwood_pos = []
hm = np.intersect1d(findley[overlap_col], erwood[overlap_col])


print('hm')
