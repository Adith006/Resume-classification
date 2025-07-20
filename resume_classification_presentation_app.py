# -*- coding: utf-8 -*-
"""
Resume Parser App with Robust NLTK Handling
"""

import os
import sys
import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore")
import re
import nltk
from nltk.tokenize import word_tokenize
import spacy
from nltk.corpus import stopwords
from spacy.matcher import Matcher
import docx2txt
import PyPDF2
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# ==================== NLTK INITIALIZATION ====================
def initialize_nltk():
    """Robust NLTK data initialization with multiple fallbacks"""
    try:
        # Try default download first
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        # Verify other required datasets
        required_data = [
            'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'vader_lexicon', 'movie_reviews', 'conll2000', 'brown'
        ]
        for dataset in required_data:
            try:
                nltk.data.find(f'corpora/{dataset}')
            except LookupError:
                nltk.download(dataset, quiet=True)
                
        # Alternative path for cloud environments
        if not nltk.data.find('tokenizers/punkt'):
            nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                os.makedirs("nltk_data", exist_ok=True)
                nltk.download('punkt', download_dir="nltk_data", quiet=True)
                
    except Exception as e:
        st.error(f"Failed to initialize NLTK: {e}")
        sys.exit(1)

initialize_nltk()
my_stop_words = set(stopwords.words("english"))

# ==================== MODEL LOADING ====================
try:
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    
    # Load your models - adjust paths as needed
    with open("final_model.sav", 'rb') as f:
        loaded_model = pickle.load(f)
    with open("tfidf_vec.sav", 'rb') as f:
        loaded_vect = pickle.load(f)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    sys.exit(1)

# ==================== TEXT PROCESSING ====================
def robust_word_tokenize(text):
    """Tokenization with multiple fallback strategies"""
    try:
        return word_tokenize(text)
    except LookupError:
        # First fallback: try downloading punkt again
        try:
            nltk.download('punkt', quiet=True)
            return word_tokenize(text)
        except:
            # Second fallback: simple whitespace tokenizer
            warnings.warn("Using fallback tokenizer - results may be less accurate")
            return text.split()

def process_resume(resume_text):
    """Cleaning function with enhanced error handling"""
    try:
        resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~ """), ' ', resume_text)
        resume_text = re.sub(r'[^\x00-\x7f]', ' ', resume_text)
        resume_text = re.sub('https?://\S+|www|WWW\.\S+', ' ', resume_text)
        resume_text = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', resume_text)
        resume_text = re.sub('\n',' ', resume_text)
        return resume_text.lower()
    except Exception as e:
        st.error(f"Text processing error: {e}")
        return ""

# ==================== STREAMLIT APP ====================
def main():
    st.title("Resume Parser App")
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio("Choose Function", ["Classification", "Screening"])
    
    if page == "Classification":
        st.header("Resume Classification")
        uploaded_files = st.file_uploader("Upload resumes", 
                                         accept_multiple_files=True, 
                                         type=['docx', 'pdf', 'doc'])
        
        if uploaded_files:
            results = []
            for file in uploaded_files:
                try:
                    text = convert_doc_to_docx(file)
                    if text:
                        cleaned = process_resume(text)
                        tokens = robust_word_tokenize(cleaned)
                        tokens = [t for t in tokens if t not in my_stop_words]
                        
                        # Your classification logic here
                        # ...
                        
                        results.append({
                            "File": file.name,
                            "Category": "Sample Result"  # Replace with actual classification
                        })
                except Exception as e:
                    st.warning(f"Failed to process {file.name}: {e}")
            
            if results:
                st.dataframe(pd.DataFrame(results))
    
    elif page == "Screening":
        st.header("Resume Screening")
        # Add your screening logic here

if __name__ == "__main__":
    main()
