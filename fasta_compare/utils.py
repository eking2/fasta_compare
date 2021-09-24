from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from io import StringIO
import streamlit as st
from streamlit.uploaded_file_manager import UploadedFile
from typing import Dict, Union, List
import re

# global
SESSION_STATE = st.session_state
SESSION_STATE['fasta1'] = None
SESSION_STATE['fasta2'] = None

AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'

def uploaded_fasta_to_state(upload: UploadedFile) -> Dict[str, Union[SeqRecord, str]]:

    text = StringIO(upload.read().decode('utf-8'))
    records = list(SeqIO.parse(text, 'fasta'))

    return {'records' : records, 'name' : upload.name}

def seqrecords_to_seqs(seqrecords: List[SeqRecord]) -> List[str]:

    # remove noncanonical amino acids and gaps
    # return [re.sub(f'[^{AA_LIST}]', '', str(record.seq)) for record in seqrecords]

    return [str(record.seq) for record in seqrecords]
