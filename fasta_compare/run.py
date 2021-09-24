from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from fasta_compare.utils import AA_LIST, seqrecords_to_seqs
import streamlit as st
from typing import List
import seaborn as sns

def get_freqs(seq: str) -> Dict[str, float]:

    """Get amino acid frequencies for a single sequence.
    
    Parameters
    ----------
    seq : str
        Amino acid sequence
        
    Returns
    -------
    freqs : Dict[str, float]
        Dictionary with amino acids as keys and frequency as values
    """
    
    freqs = {}
    
    # counts for each amino acid
    for aa in AA_LIST:
        freqs[aa] = seq.count(aa)
        
    # normalize by length
    total = sum(freqs.values())
    freqs = {aa: [(count/total) * 100] for aa, count in freqs.items()}
        
    return freqs


def get_all_freqs(records: List[SeqRecord], source: str) -> pd.DataFrame:

    """Get amino acid frequencies for all sequences in FASTA.
    
    Parameters
    ----------
    records : List[SeqRecord]
        List of Bio.SeqRecords
    source: str
        Uploaded file name, to label on plot
        
    Returns
    -------
    freqs_df : pd.DataFrame
        Dataframe in tidy format with columns ['index', 'aa', 'freq', 'set']
    """
    
    # use fasta name as label
    fasta_name = source.split('.')[0]

    # shape [sample, amino acid freq]
    df = pd.DataFrame(columns=list(AA_LIST))
    
    for record in records:

        # get aa freqs from each sequence
        seq = str(record.seq)
        freqs = get_freqs(seq)
        name = record.name
        
        # row bind
        temp_df = pd.DataFrame.from_dict(freqs, orient='columns')
        temp_df['index'] = name
        df = df.append(temp_df)
    
    # gather to tidy
    freqs_df = pd.melt(df, id_vars=['index'], var_name='aa', value_name='freq')
    freqs_df['set'] = fasta_name
    
    return freqs_df

def plot_freqs_box(inputs: List[Dict]) -> None:
    
    """Boxplot comparing amino acid frequencies for input FASTA files.
    
    Parameters
    ----------
    inputs : List
        List of dicts with keys 'records' holding SeqRecords and 'name' with FASTA name
    """
    
    # read sequences
    records = [inp['records'] for inp in inputs]
    sources = [inp['name'] for inp in inputs]
    freqs = [get_all_freqs(record, source) for record, source in zip(records, sources)]
    
    combined = pd.concat(freqs, axis=0)
    
    # plot box
    fig = plt.figure(figsize=(12, 5.5))
    
    p = sns.boxplot(data=combined, x='aa', y='freq', hue='set', fliersize=0)
    
    plt.xlabel('Amino Acid', size=13)
    plt.ylabel('Frequency (%)', size=13)
    plt.legend()
    plt.grid(alpha=0.2)

    st.pyplot(fig)