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
    fig = plt.figure(figsize=(12, 5.5), dpi=150)
    
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

def msa_to_df(msa: str) -> pd.DataFrame:

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


def calc_entropy_position(msa_df: pd.DataFrame, pos: int, eps: float=1e-12) -> float:

    """Calculate Shannon entropy at selected position.
    
    Parameters
    ----------
    msa_df : pd.DataFrame
        DataFrame with rows for each sample, column for each position
        with values being the amino acid
    pos : int
        Position (1-index) to get entropy
    eps : float (default: 1e-12)
        Avoid nan after log

    Returns
    -------
    entropy : float
        Shannon entropy in bits 
    """

    # get msa column
    msa_col = msa_df[pos].tolist()

    # counts for each aa
    counts = {aa: msa_col.count(aa) for aa in AA_LIST}

    # total canonical and normalize to freqs
    total = sum(counts.values())
    freqs = {aa: (count/total) + eps for aa, count in counts.items()}

    # shannon entropy in bits
    entropy = -np.sum([freq * np.log2(freq) for freq in freqs.values()])

    return entropy


def get_seq_entropy(msa_df):

    """
    Calculate Shannon entropy at all positions. Ignore gaps.
    
    Parameters
    ----------
    msa_df : pd.DataFrame
        DataFrame with rows for each sample, column for each position
        with values being the amino acid
    
    Returns
    -------
    entropy : List[float] 
        Shannon entropy at each position, in bits
    """

    positions = msa_df.columns
    entropy = [calc_entropy_position(msa_df, pos) for pos in positions]

    return entropy

def fasta_to_entropy(records, template):

    """
    Run pipeline from raw FASTA to entropy values at template positions.

    Parameters
    ----------
    records : List[SeqRecord]
        SeqRecords from input FASTA file 
    template : str
        Amino acid sequence for template

    Returns
    -------
    entropy : List[float]
        Shannon entropy at each position, in bits
    """

    # convert records to FASTA
    fasta_temp = NamedTemporaryFile()
    SeqIO.write(records, fasta_temp.name, 'fasta')

    # pipeline
    msa_temp = run_mafft_drop_gaps(fasta_temp.name, template)
    msa_df = msa_to_df(msa_temp.name)
    entropy = get_seq_entropy(msa_df)

    # close temp files
    fasta_temp.close()
    msa_temp.close()

    return entropy

def plot_entropy(inputs: List[Dict], template: str, window: int) -> None:

    """
    Plot smoothed sequence entropy.
    
    Parameters
    ----------
    inputs : List[Dict]
        List of dicts with keys 'records' holding SeqRecords, 
        'name' with FASTA name, and 'template' with template seq
    template : str
        Amino acid sequence for template
    window : int
        Sliding window size for rolling mean
    """

    # read sequences
    records = [inp['records'] for inp in inputs]
    sources = [inp['name'].split('.')[0] for inp in inputs]

    ents = [fasta_to_entropy(rec, template) for rec in records]

    fig = plt.figure(figsize=(7, 4), dpi=150)

    # smooth input with rolling window
    ent_1 = pd.Series(ents[0]).rolling(window=window).mean().values
    ent_2 = pd.Series(ents[1]).rolling(window=window).mean().values

    plt.plot(ent_1, label=sources[0])
    plt.plot(ent_2, label=sources[1])

    plt.grid(alpha=0.2)
    plt.xlabel('Position')
    plt.ylabel('Shannon Entropy (bits)')
    plt.legend()

    st.pyplot(fig)