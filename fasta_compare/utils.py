from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from io import StringIO
import streamlit as st
from streamlit.uploaded_file_manager import UploadedFile
from typing import Dict, Union, List
import re
from tempfile import NamedTemporaryFile
import subprocess
from subprocess import CompletedProcess

# global
SESSION_STATE = st.session_state
SESSION_STATE['fastas'] = None
SESSION_STATE['template'] = None

AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'

def parse_uploaded_fasta(upload: UploadedFile) -> Dict[str, Union[SeqRecord, str]]:

    text = StringIO(upload.read().decode('utf-8'))
    records = list(SeqIO.parse(text, 'fasta'))

    return {'records' : records, 'name' : upload.name}

def seqrecords_to_seqs(seqrecords: List[SeqRecord]) -> List[str]:

    # remove noncanonical amino acids and gaps
    # return [re.sub(f'[^{AA_LIST}]', '', str(record.seq)) for record in seqrecords]

    return [str(record.seq) for record in seqrecords]

def check_seq_valid(sequence: str) -> None:

    pat = re.compile(f'[^{AA_LIST}]')
    results = pat.search(sequence)

    assert results is None, 'invalid sequence'


def records_to_fasta(records) -> NamedTemporaryFile:

    fasta_temp = NamedTemporaryFile()
    SeqIO.write(records, fasta_temp.name, 'fasta')

    return fasta_temp


def run_cmd(cmd: str) -> CompletedProcess:

    res = subprocess.run(cmd.split(' '), capture_output=True)
    print(res.stdout.decode('utf-8'))
    print(res.stderr.decode('utf-8'))

    return res