import streamlit as st
from fasta_compare.utils import SESSION_STATE
from fasta_compare import pages
from fasta_compare.run import fasta_to_entropy

PAGES = {
    'Upload' : pages.upload,
    'Results' : pages.results
}

def main():

    st.title('FASTA Compare')

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio('Go to', list(PAGES.keys()))

    page = PAGES[selection]()

if __name__ == '__main__':
    main()