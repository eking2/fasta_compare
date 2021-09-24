from Bio import SeqIO
import streamlit as st
from fasta_compare.utils import parse_uploaded_fasta, seqrecords_to_seqs, SESSION_STATE
from fasta_compare.run import plot_freqs_box

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
            SESSION_STATE['fastas'] = [parse_uploaded_fasta(upload) for upload in fasta]
            st.success('Uploaded')
    
    # display current files
    if SESSION_STATE.get('fastas') is not None:
        inputs = SESSION_STATE['fastas']
        names = [inp['name'] for inp in inputs]
        lens = [len(inp['records']) for inp in inputs]

        st.write(f"Current inputs: {names[0]} ({lens[0]} sequences) "
                    f"and {names[1]} ({lens[1]} sequences)")

def results():

    st.header('Results')

    # no fasta inputs
    if SESSION_STATE.get('fastas') is None:
        st.warning('Input 2 FASTA files on the Upload tab')
    else:
        inputs = SESSION_STATE.get('fastas')

        st.write(f"{inputs[0]['name']} ({len(inputs[0]['records'])} sequences)")
        st.write(f"{inputs[1]['name']} ({len(inputs[1]['records'])} sequences)")
        st.markdown("""---""")

        # plot amino acid distributions
        st.write('Amino acid frequencies')
        with st.spinner('Loading plot...'):
            plot_freqs_box(inputs)

        # plot sequence entropy

        # plot pair-wise sequence identities