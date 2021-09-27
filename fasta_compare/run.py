from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from fasta_compare.utils import AA_LIST, seqrecords_to_seqs, check_seq_valid
import streamlit as st
from typing import List
import seaborn as sns
import subprocess
import shlex
from prody import parseMSA, refineMSA, writeMSA
from io import StringIO
from tempfile import NamedTemporaryFile, TemporaryFile


def get_freqs(seq: str) -> Dict:

    """
    Get amino acid frequencies for a single sequence.
    
    Parameters
    ----------
    seq : str
        Amino acid sequence
        
    Returns
    -------
    freqs : Dict[str, float]
        Dictionary with amino acids as keys and frequency (or counts) as values
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

    """
    Get amino acid frequencies for all sequences in FASTA.
    
    Parameters
    ----------
    records : List[SeqRecord]
        List of Bio.SeqRecords
    source : str
        Uploaded file name, to label on plot
        
    Returns
    -------
    freqs_df : pd.DataFrame
        Dataframe in tidy format with columns ['index', 'aa', 'freq', 'set']
    """

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

    # use fasta name as label
    fasta_name = source.split('.')[0]
    freqs_df['set'] = fasta_name
    
    return freqs_df


def plot_freqs_box(inputs: List[Dict]) -> None:
    
    """
    Boxplot comparing amino acid frequencies for input FASTA files.
    
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


def add_template(fasta: str, template: str) -> TemporaryFile:

    """
    Add template sequence to FASTA file
    
    Parameters
    ----------
    fasta : str
        Path to FASTA file
    template : str
        Amino acid sequence for template

    Returns
    -------
    fasta_temp : TemporaryFile
        Tempfile holding FASTA with template sequence
    """

    # check that all characters are valid amino acids
    check_seq_valid(template)

    # add sequence to top of fasta
    header = f'>template\n{template}\n'
    text = Path(fasta).read_text()
    fasta_w_header = header + text

    # save to temp file
    fasta_temp = NamedTemporaryFile()
    fasta_temp.write(fasta_w_header.encode('utf-8'))
    fasta_temp.seek(0)

    return fasta_temp

def run_mafft_drop_gaps(fasta: str, template: str) -> TemporaryFile:

    """
    Run MAFFT alignment on FASTA and drop gap columns.

    Parameters
    ----------
    fasta : str
        Path to template file
    template : str
        Amino acid sequence for template

    msa_temp: TemporaryFile
        MSA with gaps dropped
    """    

    # add template
    fasta_temp = add_template(fasta, template)

    # align
    cmd = f'mafft {fasta_temp.name}'
    result = subprocess.run(cmd.split(' '), capture_output=True)
    print(result.stderr.decode('utf-8'))

    # save msa to temp
    msa_w_gaps = NamedTemporaryFile(suffix='.fasta')
    msa_w_gaps.write(result.stdout)
    msa_w_gaps.seek(0)

    # drop gaps
    msa = parseMSA(msa_w_gaps.name)
    msa_refine = refineMSA(msa, label='template')

    # write
    msa_temp = NamedTemporaryFile()
    writeMSA(msa_temp.name, msa_refine, format='FASTA')

    # close temps
    fasta_temp.close()
    msa_w_gaps.close()

    return msa_temp

def msa_to_df(msa):

    """
    Convert aligned MSA sequences to dataframe, one column for each position. 
    
    Parameters
    ----------
    msa : str
        Path to MSA file
    """

    records = SeqIO.parse(msa, 'fasta')

    res = []
    for record in records:
        name = record.name
        seq = list(str(record.seq))

        # column for each position
        res.append([name] + seq)

    df = pd.DataFrame(res, columns = ['name'] + [i for i in range(1, len(seq) + 1)])
    df.set_index('name', inplace=True)

    return df

