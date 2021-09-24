from Bio import SeqIO
import streamlit as st
from fasta_compare.utils import uploaded_fasta_to_state, seqrecords_to_seqs, SESSION_STATE

def upload():

    st.header('Upload')

    with st.form(key="input_form"):
        fasta = st.file_uploader(label = 'FASTA files',
                                  type=['fasta', 'fa'],
                                  accept_multiple_files=True,
                                  key='fa1',
                                  help='Upload 2 FASTA files')

        submit = st.form_submit_button(label='Submit')
    
    # only allow 2 fasta
    if submit:
        if len(fasta) != 2:
            st.error('Please upload 2 FASTA files')
        else:

            # state = [List[SeqRecord], name]
            state1 = uploaded_fasta_to_state(fasta[0])
            state2 = uploaded_fasta_to_state(fasta[1])

            SESSION_STATE['fasta1'] = state1
            SESSION_STATE['fasta2'] = state2
            st.success('Uploaded')
    
    # display current files
    if SESSION_STATE.get('fasta1') is not None:
        state1_name = SESSION_STATE['fasta1']['name']
        state2_name = SESSION_STATE['fasta1']['name']

        state1_len = len(SESSION_STATE['fasta1']['records'])
        state2_len = len(SESSION_STATE['fasta2']['records'])

        st.write(f"Current inputs: {state1_name} ({(state1_len)} sequences) "
                    f"and {state2_name} ({state2_len} sequences)")

def results():

    st.header('Results')

    # no fasta inputs
    if SESSION_STATE.get('fasta1') is None:
        st.warning('Input 2 FASTA files on the Upload tab')

    # read sequences
    seqs1 = seqrecords_to_seqs(SESSION_STATE['fasta1']['records'])
    seqs2 = seqrecords_to_seqs(SESSION_STATE['fasta2']['records'])

    # plot amino acid distributions

    # plot sequence entropy

    # plot pair-wise sequence identities