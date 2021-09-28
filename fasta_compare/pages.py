from Bio import SeqIO
import streamlit as st
import textwrap
from fasta_compare.utils import parse_uploaded_fasta, seqrecords_to_seqs, SESSION_STATE
from fasta_compare.run import plot_freqs_box, plot_entropy

def upload():

    st.header('Upload')

    with st.form(key="input_form"):
        fasta = st.file_uploader(label = 'FASTA files',
                                  type=['fasta', 'fa'],
                                  accept_multiple_files=True,
                                  key='fa1',
                                  help='Upload 2 FASTA files')

        template = st.text_area(label = 'Template sequence',
                                help='Enter template amino acid sequence')

        submit = st.form_submit_button(label='Submit')
    
    # only allow 2 fasta
    if submit:
        if len(fasta) != 2:
            st.error('Please upload 2 FASTA files')

        if len(template) == 0:
            st.error('Please enter template amino acid sequence')

        else:
            SESSION_STATE['fastas'] = [parse_uploaded_fasta(upload) for upload in fasta]
            SESSION_STATE['template'] = template
            st.success('Uploaded')
    
    # display current files
    if SESSION_STATE.get('fastas') is not None:
        template = SESSION_STATE['template']
        inputs = SESSION_STATE['fastas']
        names = [inp['name'] for inp in inputs]
        lens = [len(inp['records']) for inp in inputs]

        st.write(f"Current inputs: {names[0]} ({lens[0]} sequences) "
                    f"and {names[1]} ({lens[1]} sequences)")
        st.write()
        st.write(f'Template ({len(template)} bp): {textwrap.fill(template)}')

def results():

    st.header('Results')

    # no fasta inputs
    if SESSION_STATE.get('fastas') is None:
        st.warning('Input 2 FASTA files on the Upload tab')
    else:
        inputs = SESSION_STATE.get('fastas')
        template = SESSION_STATE.get('template')

        st.write(f"{inputs[0]['name']} ({len(inputs[0]['records'])} sequences)")
        st.write(f"{inputs[1]['name']} ({len(inputs[1]['records'])} sequences)")
        st.write(f"Template ({len(template)}): {textwrap.fill(template)}")
        st.markdown("""---""")

        # plot amino acid distributions
        st.write('Amino acid frequencies')
        with st.spinner('Loading plot...'):
            plot_freqs_box(inputs)

        # plot sequence entropy
        st.write('Sequence conservation')
        window = st.slider('Window size', min_value=0, max_value=20, value=12, step=1)
        with st.spinner('Loading plot...'):
            plot_entropy(inputs, template, window)

        # plot pair-wise sequence identities