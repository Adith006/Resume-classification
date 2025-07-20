# -*- coding: utf-8 -*-
"""Resume Parser App (NLTK-Free Version)"""

import os
import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore")
import re
import spacy
from spacy.matcher import Matcher
import docx2txt
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import tempfile

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ==================== TEXT PROCESSING FUNCTIONS ====================
def process_resume(resume_text):
    """Clean resume text without NLTK"""
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~ """), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]', ' ', resume_text)
    resume_text = re.sub('https?://\S+|www|WWW\.\S+', ' ', resume_text)
    resume_text = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', resume_text)
    resume_text = re.sub('\n',' ', resume_text)
    return resume_text.lower()

def remove_emoji(text):
    """Remove emojis using regex"""
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def tokenize_with_spacy(text):
    """Tokenize using spaCy instead of NLTK"""
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

# ==================== RESUME PARSING FUNCTIONS ====================
def convert_doc_to_docx(file):
    """Convert DOC/PDF to text"""
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

def extract_name_from_resume(text):
    """Extract name using spaCy"""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_skills(text):
    """Skill extraction with spaCy"""
    skills = ['workday', 'hcm', 'peoplesoft', 'sql', 'reactjs', 'python']  # Your skill list
    doc = nlp(text.lower())
    found_skills = set()
    
    # Check tokens
    for token in doc:
        if token.text in skills:
            found_skills.add(token.text.capitalize())
    
    # Check noun chunks
    for chunk in doc.noun_chunks:
        if chunk.text.lower() in skills:
            found_skills.add(chunk.text.capitalize())
    
    return ", ".join(found_skills)

# ==================== STREAMLIT APP ====================
def main():
    st.title("Resume Parser App (NLTK-Free)")
    page = st.sidebar.radio("Navigate", ["Classification", "Screening"])
    
    # Load models
    try:
        with open("final_model.sav", 'rb') as f:
            model = pickle.load(f)
        with open("tfidf_vec.sav", 'rb') as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
    
    if page == "Classification":
        st.header("Resume Classification")
        uploaded_files = st.file_uploader("Upload resumes", 
                                       accept_multiple_files=True,
                                       type=['docx', 'doc', 'pdf'])
        
        if uploaded_files and st.button("Classify"):
            results = []
            for file in uploaded_files:
                try:
                    text = convert_doc_to_docx(file)
                    if text:
                        cleaned = process_resume(text)
                        cleaned = remove_emoji(cleaned)
                        tokens = tokenize_with_spacy(cleaned)
                        processed_text = " ".join(tokens)
                        
                        # Vectorize and predict
                        features = vectorizer.transform([processed_text])
                        pred = model.predict(features)[0]
                        
                        categories = {
                            0: "Peoplesoft",
                            1: "React",
                            2: "SQL",
                            3: "Workday"
                        }
                        results.append({
                            "File": file.name,
                            "Category": categories.get(pred, "Unknown")
                        })
                except Exception as e:
                    st.warning(f"Error processing {file.name}: {e}")
            
            if results:
                st.dataframe(pd.DataFrame(results))
    
    elif page == "Screening":
        st.header("Resume Screening")
        uploaded_files = st.file_uploader("Upload resumes", 
                                       accept_multiple_files=True,
                                       type=['docx', 'pdf'])
        job_desc = st.text_area("Paste job description")
        
        if uploaded_files and job_desc and st.button("Screen"):
            results = []
            for file in uploaded_files:
                try:
                    text = convert_doc_to_docx(file)
                    if text:
                        cleaned = process_resume(text)
                        cleaned = remove_emoji(cleaned)
                        
                        # Extract info
                        name = extract_name_from_resume(text) or file.name
                        skills = extract_skills(cleaned)
                        
                        # Calculate match score
                        cv = CountVectorizer(stop_words='english')
                        matrix = cv.fit_transform([cleaned, job_desc])
                        score = round(cosine_similarity(matrix)[0][1] * 100, 2)
                        
                        results.append({
                            "Name": name,
                            "Skills": skills,
                            "Match Score": f"{score}%"
                        })
                except Exception as e:
                    st.warning(f"Error processing {file.name}: {e}")
            
            if results:
                st.dataframe(pd.DataFrame(results))
                csv = pd.DataFrame(results).to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="screening_results.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
