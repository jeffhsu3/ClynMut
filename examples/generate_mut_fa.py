""" Generate fasta files containing all the mutation sequences
to feed into es1m extract.py
"""
import argparse
import pandas as pd
import pathlib
import requests
import torch
import pickle

#from alphafold2_pytorch.utils import get_esm_embedd
def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  
    )

    parser.add_argument(
        "mutation_file", 
        type=pathlib.Path, 
        help="Mutation file" 
    )

    parser.add_argument(
        "output_file", 
        type=pathlib.Path, 
        help="Output file" 
    )

    parser.add_argument(
        "--uniprot", dest="uniprot", default=False, action="store_true"
    )

    return parser

def query_pdb_seq(pdiid, chain):
    """ 
    """
    SERVER_URL = f"https://www.rcsb.org/fasta/entry/{pdiid}/display"
    r = requests.get(SERVER_URL, timeout=10)
    r = r.content.decode('utf-8')
    results = r.split(">")
    for i in results[1:]:
        i = i.split("\n")
        chains = i[0].split("|")[1].lstrip("Chains ").lstrip("Chain ").split(",")
        if chain in chains:
            return "".join(i[1:])
    
    return ""
     
def query_uniprot_mapping(pdiid, chain):
    SERVER_URL = "https://www.ebi.ac.uk/pdbe/api"
    UNIPROT = "/mappings/uniprot"
    r = requests.get(f"{SERVER_URL}{UNIPROT}/{pdiid}", timeout=10).json()
    uniprot_id = None
    # :TODO grab the start and end positions as well
    for uniprot_id in r[pdiid.lower()]["UniProt"]:
        if chain in [
            i["chain_id"] for i in r[pdiid.lower()]["UniProt"][uniprot_id]['mappings']]:
            break
        else: pass
    return uniprot_id, chain

def query_uniprot_seq(uniprot_id):
    uniprot_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    r = requests.get(uniprot_url, timeout=10)
    r = r.content.decode('utf-8')
    results = r.split("\n")[1:]

    return "".join(results).strip()


def embed_mutations(args):
    # Turn these into arguments
    # :TODO need to get the uniprot ID from PDB id + chain ID
    pos_label = "RESID"
    chain_label = "CHAIN"

    if args.uniprot:
        id_label = "uniprot"
    else:
        id_label = "PDIID"
    
    df = pd.read_csv(args.mutation_file, '\t', usecols=range(0, 6))
    df = df.loc[~df[pos_label].isnull(), :]
    df[pos_label] = df[pos_label].astype(int)
    df["synth"] = df[id_label] + "_" + df["CHAIN"]
    # Unique protein identifiers
    unique_df = df.loc[~df["synth"].duplicated(), :]

    chains = []
    seqnames = []
    seqs = []
    pdb_seqs = []
    pdids = []

    for _, j in unique_df.iterrows():
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

    sdf = pd.DataFrame({
        'uniprotid': seqnames,
        'chains': chains, 
        'seq': seqs,
        'pdb_seq': pdb_seqs,
        },
        index=pd.MultiIndex.from_arrays(
            [pdids, chains], 
            names=['pdiid', 'chain']))
    
    bad = []
    embedd_dict = {}
    embedd_model, alphabet = torch.hub.load(
        "facebookresearch/esm", "esm1b_t33_650M_UR50S")
    batch_converter = alphabet.get_batch_converter()
    data = []
    uni_matches = []
    pdb_matches = []
    print('Sequences queried')
    # ESM max embedding position
    MAX_EMBEDDING_POS = 1024

    # Include wild-type embedds for comparison?
    for i, j in df.iterrows():
        # Starts with pdb sequence and if there is an IndexError or the 
        # wildtype residue does not match falls back onto the 
        resid = j[pos_label] - 1
        try:
            uniprot_seq = list(sdf.loc[(j[id_label], j[chain_label]), 'seq'])
            pdb_seq = list(sdf.loc[(j[id_label], j[chain_label]), 'pdb_seq'])
        except KeyError:
            bad.append((j[id_label], j[pos_label], "No Match"))
            continue
        try:
            uniprot_match = uniprot_seq[resid] == j.WILD
        except IndexError:
            uniprot_match = False
        try:
            pdb_match = pdb_seq[resid] == j.WILD
        except IndexError:
            pdb_match = False

        uni_matches.append(uniprot_match)
        pdb_matches.append(pdb_match)
        if pdb_match:
            pdb_seq[resid] = j.MUTANT
            seq = pdb_seq
        elif uniprot_match:
            uniprot_seq[resid] = j.MUTANT
            seq = uniprot_seq
        else:
            bad.append((j[id_label], j[pos_label], "Neither Matches"))
            continue
        seq = "".join(seq)
        identifier = f"{j[id_label]}_{j[pos_label]}_{j.MUTANT}_{i}"
        data.append((identifier, seq))

    with open(f'{args.output_file}_mismatch.txt', 'w+') as handle:
        for i, j, k in bad:
            handle.write(f"{i}\t{j}\t{k}\n")
    # How to set batch size for the batch_converter
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens[:, 0:MAX_EMBEDDING_POS]
    with torch.no_grad(): 
        results = embedd_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_reps = results["representations"][33]

    for i, (label, seq) in enumerate(data):
        embedd_dict[label] = (token_reps[i, 1:len(seq)+1].mean(dim=0))

    with open(args.output_file, 'wb') as handle:
        pickle.dump(embedd_dict, handle)

    with open(f'{args.output_file}_mismatch.txt', 'w+') as handle:
        for i, j, k in bad:
            handle.write(f"{i}\t{j}\t{k}\n")
    # Debugging:
    df['um'] = uni_matches
    df['pm'] = pdb_matches
    return(embedd_dict)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    embed_dict = embed_mutations(args)