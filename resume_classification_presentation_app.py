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
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from keybert import KeyBERT
from datetime import datetime

pd.set_option('display.max_columns', None)  # To display all columns
pd.set_option('display.expand_frame_repr', True)  # To expand the DataFrame width
pd.set_option('display.max_colwidth', None)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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
    skills = ['workday', 'hcm', 'eib', 'picof','workday hcm',
              'workday studio','nnbound/outbound integrations',
              'peoplesoft', 'pia','ccb','birt','peci','ccw','pum','people tools',
              'peoplesoft implementation','peoplesoft components',
              'peoplesoft dba','peoplesoft admin','peoplesoft admin/dba','peopleSoft fscm', 
              'peopletoolsupgrade','peopletools upgrade','process scheduler servers',
              'peoplesoft hrms','peopleSoft consultant','peopledoft cloud',
              'PeopleSoft migrations','peoplesoft Testing Framework','pure internet architecture',
              'sql','sql server', 'ms sql server','msbi', 'sql developer', 'ssis','ssrs',
              'ssms','t-sql','tsql','Razorsql', 'razor sql','triggers','powerbi','power bi',
              'oracle sql', 'pl/sql', 'pl\sql','oracle', 'oracle 11g','oledb','cte','ddl',
              'dml','etl','mariadb','maria db','reactjs', 'react js', 'react js developer', 'html', 
              'css3','xml','javascript','html5','boostrap','jquery', 'redux','php', 'node js',
              'nodejs','apache','netbeans','nestjs','nest js','react developer','react hooks',
              'jenkins','rdbms','core connectors','PICOF','workday web services']

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

@st.cache(allow_output_mutation=True)
def extract_resume_summary(resume_text, max_length=100):
    my_model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    text = "summarize: " + resume_text
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    
    summary_ids = my_model.generate(input_ids, max_length=max_length, num_beams=4, no_repeat_ngram_size=2)
    t5_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return t5_summary
@st.cache(allow_output_mutation=True)
def load_model():
    model = KeyBERT("distilbert-base-nli-mean-tokens")
    return model
model = load_model()
def extract_keywords(resume):
    keywords_scores = model.extract_keywords(
    resume,
    top_n=10,
    keyphrase_ngram_range=(1, 3),
    use_maxsum = True,
    stop_words="english",)
    keywords = [keyword for keyword, _ in keywords_scores]
    return ",".join (keywords)



def expDetails(text):
    text = text.lower()
    duration_pattern = r'(\d+\.?\d*)\s*(year|years|yr|yrs|month|months|mo|mos)'
    durations = re.finditer(duration_pattern, text)
    
    total_months = 0
    for match in durations:
        value = float(match.group(1))
        unit = match.group(2)
        
        if unit in ['year', 'years', 'yr', 'yrs']:
            total_months += value * 12
        elif unit in ['month', 'months', 'mo', 'mos']:
            total_months += value
    
    if total_months > 0:
        return f"{round(total_months)} Months"
    
    date_range_pattern = r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{1,2}/\d{4})\s*(?:-|to)\s*(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{1,2}/\d{4}|present|current|now)'
    date_ranges = re.finditer(date_range_pattern, text)
    
    for match in date_ranges:
        start_date = parse_date(match.group(1))
        end_date = parse_date(match.group(2)) if match.group(2) not in ['present', 'current', 'now'] else datetime.now()
        
        if start_date and end_date:
            delta = end_date - start_date
            months = delta.days // 30  # Approximate
            if months > 0:
                return f"{months} Months"
    
    for i in range(len(text.split()) - 2):
        if text[i] in ['year', 'years', 'yr', 'yrs', 'month', 'months', 'mo', 'mos']:
            exp_text = ' '.join(text[i-2:i+3])
            matches = re.findall(r'\d+\.?\d*', exp_text)
            if matches:
                experience = float(matches[0])
                if any(m in exp_text for m in ['month', 'months', 'mo', 'mos']):
                    return f"{round(experience)} Months"
                else:
                    return f"{round(experience * 12)} Months"
    
    return "Experience not specified"

def parse_date(date_str):
    """Helper function to parse various date formats"""
    try:
        # Handle month-year formats (e.g., "Jan 2020" or "01/2020")
        if '/' in date_str:
            month, year = map(int, date_str.split('/'))
            return datetime(year, month, 1)
        else:
            month_str, year = date_str.split()
            month_dict = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month = month_dict.get(month_str[:3].lower(), 1)
            return datetime(int(year), month, 1)
    except:
        return None
    return None
#extracting education details from the resume
nlp = spacy.load('en_core_web_sm')

def parse_resume(resume_text):
    doc = nlp(resume_text)

    # Initialize variables to store education information
    education = []

    # Define education keywords
    education_keywords = ['education','EDUCATION', 'qualification', 'academic background','university','school','college','degree','engineering','educational qualification']

    # Iterate over each sentence in the resume
    for sent in doc.sents:
        lower_sent = sent.text.lower()

        # Check if the sentence contains any education keywords
        if any(keyword in lower_sent for keyword in education_keywords):
            # Extract the entities in the sentence
            for ent in sent.ents:
                # Check if the entity label is related to education
                if ent.label_ in ['ORG', 'NORP']:
                    education.append(ent.text)

    return ",".join(education)

def main():
    st.title("Resume Parser App" )
    st.write("This app extracts information from your resume and gives you an idea about how well your resume matches to the description of job portals, the idea is to classify resume according to the job description")
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
                        name = extract_name_from_resume(cleaned)
                        skills = extract_skills(cleaned)
                        education = parse_resume(cleaned)
                        experience = expDetails(cleaned)
                        keyword = extract_keywords(cleaned)
                        summary = extract_resume_summary(cleaned)
                        
                        # Calculate match score
                        cv = CountVectorizer(stop_words='english')
                        matrix = cv.fit_transform([cleaned, job_desc])
                        score = round(cosine_similarity(matrix)[0][1] * 100, 2)
                        
                        results.append({
                            "Name": name,
                            "Skills": skills,
                            "Match Score": f"{score}%",
                            "Education":education,
                            "Experience":experience,
                            "Keywords":keyword,
                            "Summary":summary
                        })
                except Exception as e:
                    st.warning(f"Error processing {file.name}: {e}")
            
            if results:
                st.dataframe(pd.DataFrame(results))
                csv = pd.DataFrame(results)
                st.table(csv)
                csv= results.to_pdf(index = False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="screening_results.pdf",
                    mime="text/pdf"
                )

if __name__ == "__main__":
    main()
