from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from typing import List
import random

def save_fasta(records: List[SeqRecord], label: str) -> None:

    """
    Write FASTA output
    
    Parameters
    ----------
    records : List[SeqRecord]
        FASTA records to write
    label : str
        Output file name
    """

    with open(label, 'w') as f:
        SeqIO.write(records, f, 'fasta')


def shuffle_split_fasta(fasta: str, train_ratio: float) -> None:

    """
    Shuffle fasta sequences and split into two files.

    Parameters
    ----------
    fasta : str
        Path to FASTA file to split
    train_ratio : float
        Ratio of samples to keep in the training set
    """

    # read
    records = list(SeqIO.parse(fasta, 'fasta'))

    # shuffle
    random.shuffle(records)

    # split
    split_idx = int(train_ratio * len(records))
    train_samples = records[:split_idx]
    test_samples = records[split_idx:]

    # save
    save_fasta(train_samples, 'train.fasta')
    save_fasta(test_samples, 'test.fasta')


if __name__ == '__main__':
    shuffle_split_fasta(fasta = 'seqdump.txt',
                        train_ratio = 0.5)

