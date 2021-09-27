import streamlit as st
from fasta_compare.utils import SESSION_STATE
from fasta_compare import pages
from fasta_compare.run import add_template, run_mafft_drop_gaps, get_all_freqs, msa_to_df

PAGES = {
    'Upload' : pages.upload,
    'Results' : pages.results
}

def main():

    from Bio import SeqIO

    fasta = 'assets/train.fasta'
    template = 'AGR'

    msa = run_mafft_drop_gaps(fasta, template)
    out = msa_to_df(msa.name)
    st.write(out)


    msa.close()


    # st.title('FASTA Compare')

    # st.sidebar.title('Navigation')
    # selection = st.sidebar.radio('Go to', list(PAGES.keys()))

    # page = PAGES[selection]()

if __name__ == '__main__':
    main()