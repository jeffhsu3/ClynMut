""" Using data generated from Erwood et. al 2021
Saturation variant interpretation using CRISPR prime editing (2021)


The function score is the empricially determined outcome of the screen:
``` 
We derived a function score for
each mutation modelled which was based on the log-fold change from the low
fluorescence gate to the high fluorescence gate.
```

Saturation mutation of 256 mutations for the NPC1 gene and 465 for the BRCA2 gene.  

@article{findlay2018accurate,
  title={Accurate classification of BRCA1 variants with saturation genome editing},
  author={Findlay, Gregory M and Daza, Riza M and Martin, Beth and Zhang, Melissa D and Leith, Anh P and Gasperini, Molly and Janizek, Joseph D and Huang, Xingfan and Starita, Lea M and Shendure, Jay},
  journal={Nature},
  volume={562},
  number={7726},
  pages={217--222},
  year={2018},
  publisher={Nature Publishing Group}
}
"""

import pandas as pd
import torch
from clynmut import *
from IPython import embed
import esm
from alphafold2_pytorch.utils import get_esm_embedd

TEST_SET_PERC = 0.10

# NPC1 Table
SOURCE_URL = "https://www.biorxiv.org/content/biorxiv/early/2021/05/14/2021.05.11.443710/DC1/embed/media-1.xlsx?download=true"
# For transcript ID NM_007294.3
# Note that the sequence has been updated to NM_007294.4 since the publication.
npc1_sequence = list("".join(open("examples/npc1.fa", "r").read().split("\n")[1:]))
emp_df = pd.read_excel(SOURCE_URL)
# Filter to only coding variants and Missense
# Standardize datasets 
emp_df = emp_df.loc[emp_df.Consequence == "missense", :]
emp_df.to_pickle('npc1.pkl')

def split_seq(data, nsplit):
    """ Evenly split the seqeunces into equal sequence lengths
    """
    data_splits = []
    n = len(data[0][1])

MAX_BATCH = 200
#pred_dicts = model(seqs, pred_format="dict")
# Split the sequences 
# 1:120
# 1600:1855
# Load ESM-1b model

for _, j in emp_df.iterrows():
    if not args.uniprot:
        uniprot_id, chain = query_uniprot_mapping(j[id_label], j[chain_label])
        pdb_seq = query_pdb_seq(j[id_label], j[chain_label])
    else:
        uniprot_id = j[id_label]
    seqs.append(query_uniprot_seq(uniprot_id))
    #uniprot_id, chain = query_uniprot_mapping(test_query, "A")
    chains.append(chain)
    seqnames.append(uniprot_id)
    pdids.append(j[id_label])
    pdb_seqs.append(pdb_seq)


data = []

"""
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model.cuda()
batch_converter = alphabet.get_batch_converter()

batch_labels, batch_strs, batch_tokens = batch_converter(data)

data_loader = torch.utils.data.DataLoader(
    data,
    collate_fn = alphabet.get_batch_converter(),
    batch_sampler = [range(0, 5), range(5, 10)],
)

with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader): 
        toks = toks.to(device="cuda", non_blocking=True)
        results = model(toks, repr_layers=[33], return_contacts=True)

torch.save(
    results,
    'test.pt'
)
"""
