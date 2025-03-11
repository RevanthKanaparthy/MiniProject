import streamlit as st
import os
import io
import uuid
import time
import random
import socket
import secrets
import datetime
import platform
import base64
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
import plotly.express as px
import geocoder
from geopy.geocoders import Nominatim
from pdfminer.high_level import extract_text
from PIL import Image
import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.cli import download

# Set page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 10px;
        background-color: #f8f9fa;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2e7d32;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .sidebar-content {
        padding: 15px;
    }
    .recommendation-item {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 8px;
        border-left: 4px solid #ff9800;
    }
    .job-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .job-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .job-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1565c0;
    }
    .job-company {
        font-weight: 500;
        color: #333;
    }
    .job-location {
        color: #555;
        font-size: 0.9rem;
    }
    .progress-label {
        font-size: 0.85rem;
        color: #555;
    }
    .feedback-form {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading the 'en_core_web_sm' model...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Sample course data
ds_course = [
    "Data Science Specialization - Coursera",
    "Applied Data Science with Python - Coursera",
    "Machine Learning - Stanford Online",
    "Data Science: R Basics - Harvard",
    "Python for Data Science and Machine Learning Bootcamp - Udemy",
    "Deep Learning Specialization - Coursera",
    "Statistics with R - Duke University",
    "Data Science MicroMasters - edX",
    "IBM Data Science Professional Certificate - Coursera"
]

web_course = [
    "The Complete Web Developer in 2023 - Udemy",
    "Full Stack Web Development - Coursera",
    "JavaScript: Understanding the Weird Parts - Udemy",
    "React - The Complete Guide - Udemy",
    "The Web Developer Bootcamp - Udemy",
    "Modern JavaScript From The Beginning - Udemy",
    "CSS - The Complete Guide - Udemy",
    "Node.js, Express & MongoDB - Udemy",
    "Advanced CSS and Sass - Udemy"
]

android_course = [
    "Android App Development Specialization - Coursera",
    "The Complete Android Developer Course - Udemy",
    "Android Java Masterclass - Udemy",
    "Kotlin for Android: Beginner to Advanced - Udemy",
    "Android Architecture Masterclass - Udemy",
    "Flutter & Dart - The Complete Guide - Udemy",
    "Modern Android App Development - edX",
    "Android App Development with Kotlin - Pluralsight",
    "Firebase in a Weekend: Android - Udacity"
]

ios_course = [
    "iOS App Development with Swift Specialization - Coursera",
    "iOS & Swift - The Complete iOS App Development Bootcamp - Udemy",
    "SwiftUI Masterclass - Udemy",
    "iOS 13 & Swift 5 - The Complete iOS App Development Bootcamp - Udemy",
    "iOS Development with Swift - edX",
    "Swift 5 Programming - LinkedIn Learning",
    "Objective-C for Swift Developers - Udemy",
    "Core Data for iOS Developers - Pluralsight",
    "ARKit for iOS Developers - Udemy"
]

uiux_course = [
    "UI / UX Design Specialization - Coursera",
    "User Experience Research and Design - Coursera",
    "The Complete App Design Course - Udemy",
    "UI Design - Udemy",
    "UX & Web Design Master Course - Udemy",
    "Adobe XD - UI/UX Design - Udemy",
    "Figma - UI/UX Design Essential Training - LinkedIn Learning",
    "Design Thinking - edX",
    "Human-Computer Interaction - Coursera"
]

skills_keywords = [
    "python", "java", "machine learning", "data analysis", "sql", "project management",
    "cloud computing", "aws", "azure", "docker", "react", "node.js", "deep learning"
]

# Helper functions
def generate_session_token():
    return secrets.token_hex(16)

def get_geolocation():
    try:
        g = geocoder.ip('me')
        return g.latlng, g.city, g.state, g.country
    except:
        return None, "Unknown", "Unknown", "Unknown"

def get_device_info():
    return {
        "ip_address": socket.gethostbyname(socket.gethostname()),
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
    }

def generate_unique_id():
    return str(uuid.uuid4())

def parse_resume(file_path):
    text = extract_text(file_path)
    lines = text.split('\n')
    
    # Basic parsing
    parsed_data = {
        'name': lines[0] if lines else 'Not found',
        'email': next((line for line in lines if '@' in line), 'Not found'),
        'phone': next((line for line in lines if any(char.isdigit() for char in line) and len(line) < 20), 'Not found'),
        'skills': [],
        'education': [],
        'experience': [],
        'total_experience': 0,
        'degree': 'Not found',
        'college_name': 'Not found',
        'projects': [],
        'certifications': [],
        'achievements': [],
        'summary': ''
    }
    
    # More advanced parsing with NLP
    doc = nlp(text)
    
    # Extract skills
    skills_list = [
        "Python", "Machine Learning", "Data Analysis", "Project Management",
        "Cloud Computing", "SQL", "Java", "C++", "AWS", "TensorFlow", "Keras",
        "Docker", "HTML", "CSS", "JavaScript", "Django", "MySQL", "Kali Linux",
        "Metasploit", "SEO", "pandas", "scikit-learn", "Gensim", "NLTK", "BeautifulSoup",
        "React", "Node.js", "Angular", "Vue.js", "Git", "GitHub", "Agile", "Scrum",
        "DevOps", "CI/CD", "REST API", "GraphQL", "MongoDB", "PostgreSQL", "Flask",
        "FastAPI", "Spring Boot", "Kubernetes", "Linux", "Windows", "MacOS", "Android",
        "iOS", "Swift", "Kotlin", "R", "Tableau", "Power BI", "Excel", "Word", "PowerPoint"
    ]
    
    for skill in skills_list:
        if skill.lower() in text.lower():
            parsed_data['skills'].append(skill)
    
    # Extract education
    education_keywords = ["bachelor", "master", "phd", "b.tech", "m.tech", "b.e", "m.e", "bsc", "msc", "mba"]
    education_sentences = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(keyword in sent_text for keyword in education_keywords):
            education_sentences.append(sent.text)
            
            # Try to extract degree
            if parsed_data['degree'] == 'Not found':
                for keyword in education_keywords:
                    if keyword in sent_text:
                        # Find the degree with some context
                        start_idx = max(0, sent_text.find(keyword) - 10)
                        end_idx = min(len(sent_text), sent_text.find(keyword) + 20)
                        parsed_data['degree'] = sent_text[start_idx:end_idx].strip()
                        break
    
    parsed_data['education'] = education_sentences
    
    # Extract experience (simplified)
    experience_keywords = ["experience", "work", "job", "position", "role", "employment"]
    experience_sentences = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(keyword in sent_text for keyword in experience_keywords):
            experience_sentences.append(sent.text)
            
            # Try to extract years of experience
            year_patterns = [r'\d+ years', r'\d+ year', r'\d+\+ years', r'\d+\+ year']
            for pattern in year_patterns:
                import re
                matches = re.findall(pattern, sent_text)
                if matches:
                    for match in matches:
                        try:
                            years = int(re.findall(r'\d+', match)[0])
                            parsed_data['total_experience'] = max(parsed_data['total_experience'], years)
                        except:
                            pass
    
    parsed_data['experience'] = experience_sentences
    
    # Extract projects (simplified)
    project_keywords = ["project", "developed", "created", "built", "implemented"]
    project_sentences = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(keyword in sent_text for keyword in project_keywords):
            project_sentences.append(sent.text)
    
    parsed_data['projects'] = project_sentences[:5]  # Limit to 5 projects
    
    return parsed_data

@st.cache_data
def analyze_resume(resume_text):
    import re
    import spacy
    from spacy.matcher import PhraseMatcher
    from collections import Counter
    
    nlp = spacy.load("en_core_web_sm")

    # Clean text
    resume_text = re.sub(r',+', ', ', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text)
    resume_text = resume_text.strip()

    resume_doc = nlp(resume_text)

    # Extract entities
    name = None
    email = None
    phone = None
    
    for ent in resume_doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
    
    # Extract email with regex
    email_pattern = re.compile(r'[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+')
    emails = re.findall(email_pattern, resume_text)
    email = emails[0] if emails else "Not found"

    # Extract phone with regex
    phone_pattern = re.compile(r'\+?\d[\d -]{8,12}\d')
    phones = re.findall(phone_pattern, resume_text)
    phone = phones[0] if phones else "Not found"

    # Extract education
    education = []
    education_degrees = [
        "Bachelor", "Baccalaureate", "Undergraduate", "BA", "BS", "BSc",
        "Master", "Graduate", "MA", "MS", "MSc", "MBA",
        "Doctorate", "PhD", "Doctoral", "B.E", "B.Tech", "M.E", "M.Tech"
    ]
    
    for sent in resume_doc.sents:
        for degree in education_degrees:
            if degree.lower() in sent.text.lower():
                education.append(sent.text.strip())
                break

    # Extract skills
    skills_list = [
        "Python", "Machine Learning", "Data Analysis", "Project Management",
        "Cloud Computing", "SQL", "Java", "C++", "AWS", "TensorFlow", "Keras",
        "Docker", "HTML", "CSS", "JavaScript", "Django", "MySQL", "Kali Linux",
        "Metasploit", "SEO", "pandas", "scikit-learn", "Gensim", "NLTK", "BeautifulSoup",
        "React", "Node.js", "Angular", "Vue.js", "Git", "GitHub", "Agile", "Scrum"
    ]

    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(skill.lower()) for skill in skills_list]
    matcher.add("SKILLS", patterns)

    matches = matcher(resume_doc)
    skills_found = set()
    
    for match_id, start, end in matches:
        span = resume_doc[start:end]
        skills_found.add(span.text)

    # Filter out personal info from skills
    personal_info = set()
    if name:
        personal_info.update(name.lower().split())
    if email and email != "Not found":
        personal_info.update(email.lower().split('@'))
    if phone and phone != "Not found":
        personal_info.update(phone.lower().split())
        
    skills_found = {skill for skill in skills_found if skill.lower() not in personal_info}

    # Extract experience
    experience = []
    for ent in resume_doc.ents:
        if ent.label_ == "ORG":
            experience.append(ent.text)

    experience = list(set(experience))

    # Calculate resume score
    required_skills = set([
        "Python", "Machine Learning", "Data Analysis", "Project Management",
        "Cloud Computing", "SQL"
    ])
    matched_skills = required_skills.intersection(skills_found)
    resume_score = len(matched_skills) / len(required_skills) * 100
    resume_score = round(resume_score, 2)

    # Prepare result
    resume_data = {
        "name": name if name else "Not found",
        "email": email,
        "mobile_number": phone,
        "skills": list(skills_found),
        "education": education,
        "experience": experience,
        "resume_score": resume_score
    }

    return resume_data

def generate_pdf_report(resume_data, resume_score, score_breakdown, recommended_skills, recommended_field, recommended_courses):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = styles["Title"]
    title_style.textColor = colors.darkblue
    elements.append(Paragraph("Resume Analysis Report", title_style))
    elements.append(Spacer(1, 20))

    # Basic Information
    elements.append(Paragraph("Basic Information", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    basic_info = [
        ["Name", resume_data.get('name', 'Not found')],
        ["Email", resume_data.get('email', 'Not found')],
        ["Phone", resume_data.get('mobile_number', 'Not found')],
        ["Degree", resume_data.get('degree', 'Not found')]
    ]
    
    t = Table(basic_info, colWidths=[100, 300])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    # Resume Score
    elements.append(Paragraph(f"Resume Score: {resume_score}/100", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    # Score visualization (simple bar)
    score_viz = [["", "0", "25", "50", "75", "100"],
                ["Score", "█" * int(resume_score/5), "", "", "", ""]]
    
    t = Table(score_viz, colWidths=[50, 50, 50, 50, 50, 50])
    t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.darkgreen),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    # Score Breakdown
    elements.append(Paragraph("Score Breakdown", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    score_table = [["Category", "Score"]]
    for category, score in score_breakdown.items():
        score_table.append([category, f"{score}/10"])
    
    t = Table(score_table, colWidths=[200, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    # Recommendations
    elements.append(Paragraph("Recommendations", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph(f"<b>Recommended Field:</b> {recommended_field}", styles['Normal']))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("<b>Recommended Skills:</b>", styles['Normal']))
    for skill in recommended_skills:
        elements.append(Paragraph(f"• {skill}", styles['Normal']))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("<b>Recommended Courses:</b>", styles['Normal']))
    for course in recommended_courses[:5]:
        elements.append(Paragraph(f"• {course}", styles['Normal']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def user_page():
    """
    Renders the user page of the AI Resume Analyzer application.
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_unique_id()

    # Header
    st.markdown('<div class="main-header">AI Resume Analyzer</div>', unsafe_allow_html=True)
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Enhance Your Resume with AI</h3>
            <p>Upload your resume to get personalized insights and recommendations:</p>
            <ul>
                <li>Detailed resume analysis</li>
                <li>Skills assessment and recommendations</li>
                <li>Career field suggestions</li>
                <li>Course recommendations</li>
                <li>Resume score with breakdown</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://img.icons8.com/fluency/240/resume.png", width=150)
    
    # File uploader
    st.markdown('<div class="sub-header">Upload Your Resume</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose your resume (PDF format)", type="pdf")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing your resume..."):
                # Create temp directory if it doesn't exist
                temp_directory = "temp"
                if not os.path.exists(temp_directory):
                    os.makedirs(temp_directory)  

                # Save uploaded file temporarily
                temp_file_path = os.path.join(temp_directory, f"resume_{st.session_state.session_id}.pdf")
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extract text from PDF
                resume_text = extract_text(temp_file_path)
                
                # Show progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Parse resume
                resume_data = parse_resume(temp_file_path)
            
            # Display analysis results
            display_resume_analysis(resume_data)
            
            # Offer PDF download
            offer_pdf_download(resume_data)
            
            # Show additional resources
            display_additional_resources()
            
            # Clean up
            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"An error occurred while processing your resume: {str(e)}")
            st.error("Please make sure you've uploaded a valid PDF file and try again.")
    
    # Clear session button
    if st.button("Clear Session Data", help="This will reset all your data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session data cleared successfully!")
        st.experimental_rerun()

def display_resume_analysis(resume_data):
    """
    Displays the results of the resume analysis.
    """
    st.markdown('<div class="sub-header">Resume Analysis Results</div>', unsafe_allow_html=True)
    
    # Basic Information
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Basic Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {resume_data.get('name', 'Not found')}")
        st.markdown(f"**Email:** {resume_data.get('email', 'Not found')}")
    with col2:
        st.markdown(f"**Phone:** {resume_data.get('mobile_number', 'Not found')}")
        st.markdown(f"**Degree:** {resume_data.get('degree', 'Not found')}")
    
    # Experience level
    experience = resume_data.get('total_experience', 0)
    level = "Fresher" if experience == 0 else "Intermediate" if experience < 3 else "Experienced"
    
    st.markdown(f"**Experience Level:** {level}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Skills
    skills = resume_data.get('skills', [])
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Skills Analysis</h3>', unsafe_allow_html=True)
    
    if skills:
        # Create columns for skills display
        cols = st.columns(3)
        for i, skill in enumerate(skills):
            cols[i % 3].markdown(f"<div class='highlight'>✓ {skill}</div>", unsafe_allow_html=True)
    else:
        st.info("No skills were detected in your resume. Consider adding more specific technical skills.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Resume Score
    resume_score = calculate_resume_score(resume_data)
    score_breakdown = get_resume_score_breakdown(resume_data)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Resume Score</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{resume_score}/100</div>
            <div class="metric-label">Overall Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Score breakdown visualization
        for category, score in score_breakdown.items():
            st.markdown(f"<div class='progress-label'>{category}</div>", unsafe_allow_html=True)
            st.progress(score/10)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Recommendations</h3>', unsafe_allow_html=True)
    
    # Field recommendation
    recommended_field = recommend_field(skills)
    st.markdown(f"""
    <div class="recommendation-item">
        <strong>Recommended Field:</strong> {recommended_field}
    </div>
    """, unsafe_allow_html=True)
    
    # Skills recommendation
    recommended_skills = recommend_skills(skills)
    st.markdown("<strong>Recommended Skills to Develop:</strong>", unsafe_allow_html=True)
    
    skill_cols = st.columns(3)
    for i, skill in enumerate(recommended_skills):
        skill_cols[i % 3].markdown(f"<div class='highlight'>+ {skill}</div>", unsafe_allow_html=True)
    
    # Course recommendation
    recommended_courses = recommend_courses(recommended_field)
    st.markdown("<strong>Recommended Courses:</strong>", unsafe_allow_html=True)
    
    for course in recommended_courses[:5]:
        st.markdown(f"<div class='recommendation-item'>📚 {course}</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Store analysis in session state
    if 'analyzed_resumes' not in st.session_state:
        st.session_state.analyzed_resumes = []
    
    st.session_state.analyzed_resumes.append({
        "name": resume_data.get('name', 'Not found'),
        "email": resume_data.get('email', 'Not found'),
        "resume_score": resume_score,
        "recommended_field": recommended_field,
        "experience_level": level,
        "timestamp": datetime.datetime.now()
    })
    
    st.success("Your resume analysis is complete!")

def offer_pdf_download(resume_data):
    """
    Generates and offers a downloadable PDF report of the resume analysis.
    """
    resume_score = calculate_resume_score(resume_data)
    score_breakdown = get_resume_score_breakdown(resume_data)
    recommended_skills = recommend_skills(resume_data.get('skills', []))
    recommended_field = recommend_field(resume_data.get('skills', []))
    recommended_courses = recommend_courses(recommended_field)

    pdf_buffer = generate_pdf_report(resume_data, resume_score, score_breakdown, recommended_skills, recommended_field, recommended_courses)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Download Analysis Report</h3>', unsafe_allow_html=True)
    st.markdown('Get a comprehensive PDF report of your resume analysis that you can reference later.', unsafe_allow_html=True)
    
    st.download_button(
        label="📄 Download Resume Analysis Report",
        data=pdf_buffer,
        file_name="resume_analysis_report.pdf",
        mime="application/pdf"
    )
    st.markdown('</div>', unsafe_allow_html=True)

def display_additional_resources():
    """
    Displays additional resources such as resume writing tips and interview preparation videos.
    """
    st.markdown('<div class="sub-header">Additional Resources</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Resume Writing Tips</h3>', unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=y8YH0Qbu5h4")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Interview Preparation Tips</h3>', unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=Ji46s5BHdr0")
        st.markdown('</div>', unsafe_allow_html=True)

def calculate_resume_score(resume_data):
    score_breakdown = get_resume_score_breakdown(resume_data)
    return sum(score_breakdown.values())

def get_resume_score_breakdown(resume_data):
    score_breakdown = {
        "Contact Information": 0,
        "Education": 0,
        "Skills": 0,
        "Experience": 0,
        "Projects": 0,
        "Certifications": 0,
        "Summary/Objective": 0,
        "Achievements": 0,
        "Formatting": 0,
        "Keywords": 0
    }
    
    # Contact Information scoring
    if resume_data.get('name'): score_breakdown["Contact Information"] += 3
    if resume_data.get('email'): score_breakdown["Contact Information"] += 3
    if resume_data.get('mobile_number'): score_breakdown["Contact Information"] += 4
    
    # Education scoring
    if resume_data.get('degree'): score_breakdown["Education"] += 5
    if resume_data.get('college_name'): score_breakdown["Education"] += 5
    
    # Skills scoring
    skills = resume_data.get('skills', [])
    score_breakdown["Skills"] = min(len(skills), 10)
    
    # Experience scoring
    experience = resume_data.get('total_experience', 0)
    score_breakdown["Experience"] = min(experience * 2, 10)
    
    # Projects scoring
    projects = resume_data.get('projects', [])
    score_breakdown["Projects"] = min(len(projects) * 2, 10)
    
    # Certifications scoring
    certifications = resume_data.get('certifications', [])
    score_breakdown["Certifications"] = min(len(certifications) * 2, 10)
    
    # Summary/Objective scoring
    if resume_data.get('summary'): score_breakdown["Summary/Objective"] = 10
    
    # Achievements scoring
    achievements = resume_data.get('achievements', [])
    score_breakdown["Achievements"] = min(len(achievements) * 2, 10)
    
    # Default scores for formatting and keywords
    score_breakdown["Formatting"] = 8
    score_breakdown["Keywords"] = 7
    
    return score_breakdown

def recommend_skills(skills):
    all_skills = set([
        "Python", "Java", "C++", "JavaScript", "HTML", "CSS", "SQL", 
        "Machine Learning", "Data Analysis", "React", "Node.js", "Angular", 
        "Vue.js", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git", "Agile",
        "TensorFlow", "PyTorch", "NLP", "Computer Vision", "Data Visualization",
        "Flask", "Django", "RESTful API", "GraphQL", "MongoDB", "PostgreSQL"
    ])
    
    # Filter out skills the user already has
    recommended = list(all_skills - set(skills))
    
    # Return a random sample of 5 skills (or fewer if less than 5 are available)
    return random.sample(recommended, min(5, len(recommended)))

def recommend_field(skills):
    fields = {
        "Data Science": ["Python", "Machine Learning", "Data Analysis", "SQL", "Statistics", "TensorFlow", "PyTorch"],
        "Web Development": ["JavaScript", "HTML", "CSS", "React", "Node.js", "Angular", "Vue.js"],
        "Android Development": ["Java", "Kotlin", "Android SDK", "Mobile Development"],
        "iOS Development": ["Swift", "Objective-C", "iOS SDK", "Mobile Development"],
        "UI/UX Design": ["Figma", "Adobe XD", "Sketch", "User Research", "Wireframing"],
        "DevOps": ["Docker", "Kubernetes", "AWS", "Azure", "CI/CD", "Jenkins", "Git"],
        "Cybersecurity": ["Network Security", "Penetration Testing", "Kali Linux", "Cryptography"]
    }
    
    # Calculate match score for each field
    max_match = 0
    recommended_field = "General Software Development"
    
    for field, field_skills in fields.items():
        match = len(set(skills) & set(field_skills))
        if match > max_match:
            max_match = match
            recommended_field = field
    
    return recommended_field

def recommend_courses(field):
    courses = {
        "Data Science": ds_course,
        "Web Development": web_course,
        "Android Development": android_course,
        "iOS Development": ios_course,
        "UI/UX Design": uiux_course,
        "DevOps": [
            "Docker and Kubernetes: The Complete Guide - Udemy",
            "AWS Certified DevOps Engineer - A Cloud Guru",
            "DevOps with GitHub - LinkedIn Learning",
            "CI/CD with Jenkins - Pluralsight",
            "Terraform for AWS - Udemy"
        ],
        "Cybersecurity": [
            "Ethical Hacking - Udemy",
            "CompTIA Security+ Certification - Coursera",
            "Cybersecurity Specialization - Coursera",
            "Web Security - Stanford Online",
            "Network Security - edX"
        ]
    }
    
    # Return courses for the recommended field, or data science courses as default
    return courses.get(field, ds_course)

def find_jobs_page():
    """
    Renders the job search page of the AI Resume Analyzer application.
    """
    st.markdown('<div class="main-header">Find Jobs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>Job Search</h3>
        <p>Search for jobs based on your skills and location. We'll help you find relevant opportunities that match your profile.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g., Data Scientist")
    
    with col2:
        location = st.text_input("Location", placeholder="e.g., New York")
    
    search_button = st.button("🔍 Search Jobs")
    
    if search_button:
        if job_title and location:
            with st.spinner("Searching for jobs..."):
                jobs = scrape_linkedin_jobs(job_title, location)
            
            if jobs:
                display_job_results(jobs)
            else:
                st.warning("No jobs found. Try different search terms or check your internet connection.")
        else:
            st.warning("Please enter both job title and location to search for jobs.")

def scrape_linkedin_jobs(job_title, location):
    """
    Scrapes job listings from LinkedIn based on the given job title and location.
    """
    url = f"https://www.linkedin.com/jobs/search/?keywords={job_title}&location={location}"
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = None
    try:
        driver = webdriver.Firefox(options=options)
        driver.get(url)
        
        # Wait for the page to load
        wait = WebDriverWait(driver, 15)
        
        # Try different selectors for job cards
        job_card_selectors = [
            "base-card",
            "job-search-card",
            "jobs-search-results__list-item",
            "scaffold-layout__list-item"
        ]
        
        job_cards = []
        for selector in job_card_selectors:
            try:
                job_cards = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, selector)))
                if job_cards:
                    st.success(f"Found {len(job_cards)} job listings")
                    break
            except Exception:
                continue
        
        if not job_cards:
            st.warning("Could not find job listings. LinkedIn may have updated their page structure.")
            return []
        
        jobs = []
        for card in job_cards[:10]:  # Limit to first 10 jobs
            try:
                # Try multiple selectors for each element
                title_selectors = [
                    "base-card__full-link", 
                    "job-card-list__title",
                    "base-search-card__title"
                ]
                company_selectors = [
                    "job-card-container__company-name",
                    "base-search-card__subtitle",
                    "base-card-entity__secondary-title",
                    "job-card-container__primary-description"
                ]
                location_selectors = [
                    "job-card-container__metadata-item",
                    "job-search-card__location",
                    "base-card-entity__metadata"
                ]
                link_selectors = [
                    "base-card__full-link",
                    "job-card-list__title",
                    "base-card-entity__title-link"
                ]
                
                # Find title
                title = None
                for selector in title_selectors:
                    try:
                        title_element = card.find_element(By.CLASS_NAME, selector)
                        title = title_element.text
                        break
                    except Exception:
                        continue
                
                # Find company
                company = None
                for selector in company_selectors:
                    try:
                        company_element = card.find_element(By.CLASS_NAME, selector)
                        company = company_element.text
                        break
                    except Exception:
                        continue
                
                # Find location
                job_location = None
                for selector in location_selectors:
                    try:
                        location_element = card.find_element(By.CLASS_NAME, selector)
                        job_location = location_element.text
                        break
                    except Exception:
                        continue
                
                # Find link
                link = None
                for selector in link_selectors:
                    try:
                        link_element = card.find_element(By.CLASS_NAME, selector)
                        link = link_element.get_attribute("href")
                        break
                    except Exception:
                        continue
                
                # Only add job if we found at least title and company
                if title and company:
                    jobs.append({
                        "title": title,
                        "company": company,
                        "location": job_location or "Location not specified",
                        "link": link or "#"
                    })
            except Exception as e:
                st.error(f"Error scraping job card: {str(e)}")
        
        return jobs
    except Exception as e:
        st.error(f"Error initializing web driver: {str(e)}")
        return []
    finally:
        if driver:
            driver.quit()

def display_job_results(jobs):
    """
    Displays the job search results and provides an option to download them as a CSV file.
    """
    st.markdown('<div class="sub-header">Job Results</div>', unsafe_allow_html=True)
    
    if jobs:
        for job in jobs:
            st.markdown(f"""
            <div class="job-card">
                <div class="job-title">{job['title']}</div>
                <div class="job-company">{job['company']}</div>
                <div class="job-location">📍 {job['location']}</div>
                <a href="{job['link']}" target="_blank">View Job</a>
            </div>
            """, unsafe_allow_html=True)
        
        # Download results as CSV
        df = pd.DataFrame(jobs)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Results as CSV",
            data=csv,
            file_name='job_results.csv',
            mime='text/csv'
        )
    else:
        st.info("No jobs found. Try different search terms.")

def feedback_page():
    """
    Renders the feedback page of the AI Resume Analyzer application.
    """
    st.markdown('<div class="main-header">Feedback</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>We Value Your Feedback</h3>
        <p>Your feedback helps us improve the AI Resume Analyzer. Please share your thoughts and suggestions with us.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="feedback-form">', unsafe_allow_html=True)
    
    name = st.text_input("Name")
    email = st.text_input("Email")
    rating = st.slider("Rate your experience", 1, 5, 3, help="1 = Poor, 5 = Excellent")
    
    # Display rating as stars
    st.markdown(f"Your rating: {'⭐' * rating}")
    
    comments = st.text_area("Comments", placeholder="Share your thoughts, suggestions, or report any issues...")
    
    if st.button("Submit Feedback"):
        if name and email and comments:
            # Store feedback in session state
            if 'feedback_data' not in st.session_state:
                st.session_state.feedback_data = []
            
            st.session_state.feedback_data.append({
                "name": name,
                "email": email,
                "rating": rating,
                "comments": comments,
                "timestamp": datetime.datetime.now()
            })
            
            st.success("Thank you for your feedback! We appreciate your input.")
            
            # Show a thank you message
            st.balloons()
        else:
            st.warning("Please fill out all required fields before submitting.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def about_page():
    """
    Renders the about page of the AI Resume Analyzer application.
    """
    st.markdown('<div class="main-header">About AI Resume Analyzer</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Our Mission</h3>
            <p>The AI Resume Analyzer is designed to help job seekers improve their resumes and find suitable job opportunities. 
            Our mission is to empower individuals in their job search by providing data-driven insights and personalized recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://img.icons8.com/fluency/240/goal.png", width=150)
    
    st.markdown('<div class="sub-header">Key Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Resume Analysis</h3>
            <p>Get detailed insights on your resume's strengths and weaknesses. Our AI analyzes your resume and provides a comprehensive breakdown.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Personalized Recommendations</h3>
            <p>Receive tailored skill and course recommendations based on your profile and career goals.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>Job Search</h3>
            <p>Find relevant job opportunities that match your skills and preferences with our integrated job search feature.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">How to Use</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <ol>
            <li><strong>Upload your resume</strong> in PDF format on the User page</li>
            <li><strong>Review the analysis</strong> and recommendations provided by our AI</li>
            <li><strong>Use the job search feature</strong> to find relevant opportunities</li>
            <li><strong>Improve your resume</strong> based on the suggestions</li>
            <li><strong>Download the analysis report</strong> for future reference</li>
            <li><strong>Repeat the process</strong> with your updated resume to track improvements</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">Privacy Policy</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>We value your privacy. Your resume data is only used for analysis during your session and is not stored permanently on any server. 
        We do not share your personal information with third parties.</p>
        
        <p>Session data is temporarily stored in your browser and is cleared when you end your session or click the "Clear Session" button.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """
    Main function to run the AI Resume Analyzer application.
    """
    try:
        # Initialize session state
        if 'session_token' not in st.session_state:
            st.session_state.session_token = generate_session_token()
        
        # Get geolocation and device info
        latlng, city, state, country = get_geolocation()
        device_info = get_device_info()
        
        # Store user info in session state
        user_info = {
            "session_token": st.session_state.session_token,
            "geolocation": {
                "latitude": latlng[0] if latlng else None,
                "longitude": latlng[1] if latlng else None,
                "city": city,
                "state": state,
                "country": country
            },
            "device_info": device_info
        }
        
        # Sidebar
        with st.sidebar:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            
            st.title("AI Resume Analyzer")
            st.image(r"C:\Users\Vamshikrishna\Downloads\WhatsApp Image 2025-03-07 at 13.13.51_060aec48.jpg", width=100)
            
            # Navigation
            st.subheader("Navigation")
            pages = ["User", "Find Jobs", "Feedback", "About"]
            page = st.radio("", pages)
            
            # Session info
            st.subheader("Session Info")
            st.info(f"""
            Session ID: {st.session_state.session_token[:8]}...
            Location: {city}, {country}
            """)
            
            st.markdown("---")
            st.info("AI Resume Analyzer v1.0\nDesigned by ThunderBirds")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Main content
        if page == "User":
            user_page()
        elif page == "Find Jobs":
            find_jobs_page()
        elif page == "Feedback":
            feedback_page()
        elif page == "About":
            about_page()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()