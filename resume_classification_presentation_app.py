# -*- coding: utf-8 -*-
"""
Resume Classification and Screening App
Fixed version with NLTK data download handling
"""

import os
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
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import tempfile

# Initialize NLTK data
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('movie_reviews', quiet=True)
    nltk.download('conll2000', quiet=True)
    nltk.download('brown', quiet=True)

initialize_nltk()
my_stop_words = set(stopwords.words("english"))

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_colwidth', None)

# Cleaning functions
def process_resume(resume_text):
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~ """), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]', ' ', resume_text)
    resume_text = re.sub('https?://\S+|www|WWW\.\S+', ' ', resume_text)
    pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'
    resume_text = re.sub(pattern, '', resume_text)
    resume_text = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', resume_text)
    resume_text = re.sub('⇨', ' ', resume_text)
    resume_text = re.sub('\n',' ', resume_text)
    return resume_text.lower()

def remove_emoji(resume_text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', resume_text)

# Load models
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

try:
    loaded_model = pickle.load(open("final_model.sav", 'rb'))
    loaded_vect = pickle.load(open("tfidf_vec.sav", 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# App layout
page = st.sidebar.radio("Navigate", ("Resume classification", "Resume Screening"))
with st.container():
    st.header("Resume Parser App")
    st.caption("Only for Workday Resume, SQL Resume, React Resume and Peoplesoft Resume")
st.markdown('<hr>', unsafe_allow_html=True)
st.sidebar.title("Input data")

# File conversion function
def convert_doc_to_docx(file):
    if file.name.endswith('.docx'):
        return docx2txt.process(file)
    elif file.name.endswith('.doc'):
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        try:
            text = docx2txt.process(temp_file_path)
        finally:
            os.unlink(temp_file_path)
        return text
    elif file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            temp_file.write(file.read())
            temp_file.seek(0)
            reader = PyPDF2.PdfReader(temp_file.name)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    return ''

# Information extraction functions
def extract_name_from_resume(resume_text):
    try:
        doc = nlp(resume_text)
        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
        matcher.add('NAME', [pattern])
        matches = matcher(doc)
        if matches:
            for _, start, end in matches:
                return doc[start:end].text
        
        pattern = r"(?i)\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b"
        match = re.search(pattern, resume_text)
        return match.group(1).strip() if match else None
    except Exception:
        return None

def extract_skills(resume_text):
    try:
        doc = nlp(resume_text)
        tokens = [token.text for token in doc if not token.is_stop]
        
        skills = ['workday', 'hcm', 'peoplesoft', 'sql', 'reactjs', ...]  # Your full skills list
        
        skillset = []
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)
        
        for chunk in doc.noun_chunks:
            token = chunk.text.lower().strip()
            if token in skills:
                skillset.append(token)
        
        return ",".join([i.capitalize() for i in set([i.lower() for i in skillset])])
    except Exception:
        return ""

# Resume classification page
if page == "Resume classification":
    st.markdown("Overview")
    st.write("This app extracts information from your resume and classifies it into categories.")
    st.markdown('<hr>', unsafe_allow_html=True)
    
    classify = st.sidebar.button("Classify")
    st.sidebar.error("Supports DOCX, DOC, PDF")
    uploaded_files = st.sidebar.file_uploader("Upload resumes", accept_multiple_files=True, type=['.doc','.docx','.pdf'])
    
    if uploaded_files and classify:
        all_text = []
        for file in uploaded_files:
            text = convert_doc_to_docx(file)
            if text:
                all_text.append(text)
        
        predictions = []
        category_list = []
        
        for resume_text in all_text:
            try:
                cleaned_resume = process_resume(resume_text)
                cleaned_resume = remove_emoji(cleaned_resume)
                
                # Tokenization with error handling
                try:
                    cleaned_resume = word_tokenize(cleaned_resume)
                except LookupError:
                    nltk.download('punkt')
                    cleaned_resume = word_tokenize(cleaned_resume)
                
                cleaned_resume = [word for word in cleaned_resume if word not in my_stop_words]
                doc = nlp(' '.join(cleaned_resume))
                cleaned_resume = [token.lemma_ for token in doc]
                cleaned_resume = ' '.join(cleaned_resume)
                
                input_feat = loaded_vect.transform([cleaned_resume])
                prediction_id = loaded_model.predict(input_feat)[0]
                predictions.append(prediction_id)
                
                category_mapping = {
                    0: 'peoplesoft developers',
                    1: 'React developers',
                    2: 'SQL developers',
                    3: 'Workday resumes',
                }
                category_list.append(category_mapping.get(prediction_id, "unknown"))
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
                continue
        
        if category_list:
            df = pd.DataFrame({
                'File': [file.name for file in uploaded_files],
                'Category': category_list
            })
            st.write(df)

# Resume Screening page
if page == "Resume Screening":
    screening = st.sidebar.button("Screening")
    st.sidebar.error("Supports only DOCX, PDF")
    uploaded_files = st.sidebar.file_uploader("Upload resumes", accept_multiple_files=True, type=['.docx','.pdf'])
    job_description = st.sidebar.text_input("Enter job description", placeholder="Paste Job Description")
    
    if uploaded_files and screening and job_description:
        all_text = []
        for file in uploaded_files:
            text = convert_doc_to_docx(file)
            if text:
                all_text.append(text)
        
        results = []
        for i, resume_text in enumerate(all_text):
            try:
                # Processing pipeline with error handling
                cleaned_resume = process_resume(resume_text)
                cleaned_resume = remove_emoji(cleaned_resume)
                
                try:
                    cleaned_resume = word_tokenize(cleaned_resume)
                except LookupError:
                    nltk.download('punkt')
                    cleaned_resume = word_tokenize(cleaned_resume)
                
                cleaned_resume = [word for word in cleaned_resume if word not in my_stop_words]
                doc = nlp(' '.join(cleaned_resume))
                cleaned_resume = [token.lemma_ for token in doc]
                cleaned_resume = ' '.join(cleaned_resume)
                
                # Extract information
                name = extract_name_from_resume(cleaned_resume) or f"Resume {i+1}"
                skills = extract_skills(cleaned_resume)
                education = parse_resume(cleaned_resume)
                experience = expDetails(cleaned_resume)
                
                # Calculate match score
                corpus = [cleaned_resume, job_description]
                cv = CountVectorizer(stop_words='english')
                count_matrix = cv.fit_transform(corpus)
                match_percentage = round(cosine_similarity(count_matrix)[0][1] * 100, 2)
                
                results.append({
                    'Name': name,
                    'Skills': skills,
                    'Education': education,
                    'Experience': experience,
                    'Match Score': f"{match_percentage}%"
                })
            
            except Exception as e:
                st.error(f"Error processing file {i+1}: {e}")
                continue
        
        if results:
            df = pd.DataFrame(results)
            st.table(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="resume_screening_results.csv",
                mime="text/csv"
            )
