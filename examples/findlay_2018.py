""" Using data generated from Findlay et. al
Accurate classification of BRCA1 variants with saturation genome editing. (2018)

Saturation mutation of BRCA1 variants.  

Functional scores are derived from the log2 ratio of reads of the SNV to reads of the plasmid on 
day 11.  Positional biases in gene editing were accounted for.

Dataset:
3,893 SNVs, which comprise 96.5% of all possible SNVs within or immediately intronic
to these exons.  

Output to fasta file for ESM1b embeddings.


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

TEST_SET_PERC = 0.10

SOURCE_URL = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx"
# For transcript ID NM_007294.3
# Note that the sequence has been updated to NM_007294.4 since the publication.
brca_sequence = list("".join(open("examples/brca1.fa", "r").read().split("\n")[1:]))
brca = pd.read_excel(SOURCE_URL, skiprows=2)
# Filter to only coding variants and Missense
brca = brca.loc[brca.consequence == "Missense", :]

brca.to_pickle('brca.pkl')


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

data = []

with open('brca_sat.fa', 'w+') as fh:
    for i, j in brca.iterrows():
        identifier = f"brca_{i}"
        mut_seq = brca_sequence.copy()
        assert mut_seq[int(j.aa_pos)-1] == j.aa_ref
        mut_seq[int(j.aa_pos)-1] = j.aa_alt
        out_seq = mut_seq[1:120] + mut_seq[1600:1855]
        out_seq = "".join(out_seq)
        data.append((identifier, out_seq))
        fh.write(f">{identifier}\n")
        fh.write(f"{out_seq}\n")
        

    

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
