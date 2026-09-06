import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📄 AI Resume Analyzer")

st.write(
    "Upload your resume and compare it with a job description "
    "to identify matching skills, missing skills, ATS keywords "
    "and resume improvement opportunities."
)


# ============================================================
# SKILL DATABASE
# ============================================================

SKILLS = [
    # Data / Analytics
    "data analysis",
    "data analytics",
    "data visualization",
    "statistics",
    "excel",
    "advanced excel",
    "power bi",
    "tableau",
    "sql",
    "mysql",
    "postgresql",
    "pandas",
    "numpy",
    "matplotlib",

    # Programming
    "python",
    "java",
    "javascript",
    "c",
    "c++",
    "r programming",

    # AI / ML
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "generative ai",
    "computer vision",
    "tensorflow",
    "pytorch",
    "scikit-learn",

    # Business / Soft Skills
    "communication",
    "teamwork",
    "leadership",
    "problem solving",
    "analytical thinking",
    "customer service",
    "time management",
    "presentation",
    "project management",

    # Business Analysis
    "business analysis",
    "requirements gathering",
    "requirement analysis",
    "salesforce",
    "crm",
    "testing",
    "documentation",

    # IT
    "cloud computing",
    "aws",
    "azure",
    "microsoft azure",
    "linux",
    "windows",
    "networking",
    "database",
    "technical support",
]


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):

    try:
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception as e:

        st.error(f"Unable to read the PDF: {e}")

        return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    cleaned_text = clean_text(text)

    found_skills = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, cleaned_text):

            found_skills.append(skill.title())

    return sorted(set(found_skills))


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(text):

    cleaned_text = clean_text(text)

    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b", cleaned_text)

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "you",
        "your",
        "our",
        "will",
        "have",
        "has",
        "job",
        "role",
        "work",
        "working",
        "candidate",
        "experience",
        "years",
        "using",
        "into",
        "about",
        "their",
        "they",
        "them",
        "required",
        "responsibilities",
        "skills"
    }

    keywords = []

    for word in words:

        if word not in stop_words and len(word) > 3:

            keywords.append(word)

    return sorted(set(keywords))


# ============================================================
# KEYWORD MATCHING
# ============================================================

def calculate_keyword_score(resume_text, job_text):

    resume_keywords = set(extract_keywords(resume_text))

    job_keywords = set(extract_keywords(job_text))

    if not job_keywords:

        return 0, []

    matching_keywords = resume_keywords.intersection(job_keywords)

    score = (len(matching_keywords) / len(job_keywords)) * 100

    return round(min(score, 100)), sorted(matching_keywords)


# ============================================================
# SKILL MATCHING
# ============================================================

def calculate_skill_score(resume_skills, job_skills):

    if not job_skills:

        return 0

    matching = set(resume_skills).intersection(set(job_skills))

    score = (len(matching) / len(job_skills)) * 100

    return round(score)


# ============================================================
# NLP SIMILARITY
# ============================================================

def calculate_similarity(resume_text, job_text):

    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    if not resume_clean or not job_clean:
        return 0

    # Word-level TF-IDF similarity
    word_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    word_matrix = word_vectorizer.fit_transform(
        [resume_clean, job_clean]
    )

    word_similarity = cosine_similarity(
        word_matrix[0:1],
        word_matrix[1:2]
    )[0][0]

    # Character-level similarity
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1
    )

    char_matrix = char_vectorizer.fit_transform(
        [resume_clean, job_clean]
    )

    char_similarity = cosine_similarity(
        char_matrix[0:1],
        char_matrix[1:2]
    )[0][0]

    # Combine word and character similarity
    similarity = (
        (word_similarity * 0.70) +
        (char_similarity * 0.30)
    )

    return round(similarity * 100)
# ============================================================
# RESUME SECTION CHECK
# ============================================================

def check_resume_sections(resume_text):

    text = clean_text(resume_text)

    sections = {

        "Contact Information": [
            "email",
            "phone",
            "linkedin"
        ],

        "Professional Summary": [
            "summary",
            "objective",
            "profile"
        ],

        "Education": [
            "education",
            "academic",
            "degree",
            "university",
            "college"
        ],

        "Experience": [
            "experience",
            "employment",
            "work history"
        ],

        "Projects": [
            "projects",
            "project"
        ],

        "Skills": [
            "skills",
            "technical skills"
        ],

        "Certifications": [
            "certification",
            "certifications"
        ]
    }

    results = {}

    for section, keywords in sections.items():

        found = any(
            keyword in text
            for keyword in keywords
        )

        results[section] = found

    return results


# ============================================================
# IMPROVEMENT SUGGESTIONS
# ============================================================

def generate_suggestions(
    resume_skills,
    missing_skills,
    sections,
    keyword_score
):

    suggestions = []

    if missing_skills:

        suggestions.append(
            "Add relevant missing skills only if you genuinely "
            "have those skills or experience: "
            + ", ".join(missing_skills)
            + "."
        )

    if keyword_score < 50:

        suggestions.append(
            "Improve ATS keyword alignment by naturally using "
            "important keywords from the job description."
        )

    if not sections.get("Professional Summary", False):

        suggestions.append(
            "Add a concise professional summary targeted to the job."
        )

    if not sections.get("Projects", False):

        suggestions.append(
            "Add relevant academic, personal or internship-style "
            "projects with measurable outcomes."
        )

    if not sections.get("Certifications", False):

        suggestions.append(
            "Add relevant certifications or training that you have completed."
        )

    if not sections.get("Experience", False):

        suggestions.append(
            "Clearly present your relevant experience, training, "
            "projects or practical work."
        )

    if not suggestions:

        suggestions.append(
            "Your resume has good alignment with this job. "
            "Continue tailoring the resume for specific job descriptions."
        )

    return suggestions


# ============================================================
# ANALYSIS SUMMARY
# ============================================================

def generate_summary(overall_score, matching_skills, missing_skills):

    if overall_score >= 80:

        return (
            "Excellent match. Your resume is strongly aligned with "
            "the job description."
        )

    elif overall_score >= 60:

        return (
            "Good match. Your resume has several relevant skills, "
            "but some areas can be improved."
        )

    elif overall_score >= 40:

        return (
            "Moderate match. Consider tailoring your resume more "
            "closely to the requirements of this position."
        )

    else:

        return (
            "Low match. Consider significantly tailoring your resume "
            "for this position."
        )


# ============================================================
# INPUT AREA
# ============================================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Upload Resume")

    uploaded_file = st.file_uploader(
        "Upload your resume in PDF format",
        type=["pdf"]
    )


with col2:

    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="Paste the complete job description..."
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_button:

    if uploaded_file is None:

        st.warning("⚠️ Please upload your resume PDF.")

        st.stop()

    if not job_description.strip():

        st.warning("⚠️ Please enter the job description.")

        st.stop()

    # --------------------------------------------------------
    # Extract resume
    # --------------------------------------------------------

    resume_text = extract_resume_text(uploaded_file)

    if not resume_text:

        st.error(
            "Could not extract text from the resume. "
            "Please check that the PDF contains selectable text."
        )

        st.stop()

    # --------------------------------------------------------
    # Extract skills
    # --------------------------------------------------------

    resume_skills = extract_skills(resume_text)

    job_skills = extract_skills(job_description)

    # --------------------------------------------------------
    # Matching skills
    # --------------------------------------------------------

    matching_skills = sorted(
        set(resume_skills).intersection(set(job_skills))
    )

    # --------------------------------------------------------
    # Missing skills
    # --------------------------------------------------------

    missing_skills = sorted(
        set(job_skills) - set(resume_skills)
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    similarity_score = calculate_similarity(
        resume_text,
        job_description
    )

    keyword_score, matching_keywords = calculate_keyword_score(
        resume_text,
        job_description
    )

    skill_score = calculate_skill_score(
        resume_skills,
        job_skills
    )

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    overall_score = round(
    (similarity_score * 0.50)
    + (keyword_score * 0.20)
    + (skill_score * 0.30)
)
    # --------------------------------------------------------
    # Resume sections
    # --------------------------------------------------------

    sections = check_resume_sections(resume_text)

    # --------------------------------------------------------
    # Suggestions
    # --------------------------------------------------------

    suggestions = generate_suggestions(
        resume_skills,
        missing_skills,
        sections,
        keyword_score
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = generate_summary(
        overall_score,
        matching_skills,
        missing_skills
    )

    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success("✅ Resume analysis completed successfully!")


    # ========================================================
    # SCORE
    # ========================================================

    st.header("📊 Resume Match Score")

    score_col1, score_col2, score_col3 = st.columns(3)

    with score_col1:

        st.metric(
            "Overall Match",
            f"{overall_score}%"
        )

    with score_col2:

        st.metric(
            "NLP Similarity",
            f"{similarity_score}%"
        )

    with score_col3:

        st.metric(
            "Skill Match",
            f"{skill_score}%"
        )

    st.progress(overall_score / 100)


    # ========================================================
    # SCORE INTERPRETATION
    # ========================================================

    if overall_score >= 80:

        st.success(
            "🟢 Excellent match. Your resume is highly relevant "
            "to this job."
        )

    elif overall_score >= 60:

        st.info(
            "🔵 Good match. Consider emphasizing some relevant "
            "skills and keywords."
        )

    elif overall_score >= 40:

        st.warning(
            "🟡 Moderate match. Tailor your resume more closely "
            "to the requirements of this position."
        )

    else:

        st.error(
            "🔴 Low match. Consider significantly tailoring "
            "your resume for this position."
        )


    # ========================================================
    # MATCHING SKILLS
    # ========================================================

    st.divider()

    st.header("✅ Matching Skills")

    if matching_skills:

        for skill in matching_skills:

            st.write(f"🟢 {skill}")

    else:

        st.write("No matching skills found.")


    # ========================================================
    # MISSING SKILLS
    # ========================================================

    st.header("❌ Missing Skills")

    if missing_skills:

        for skill in missing_skills:

            st.write(f"🔴 {skill}")

    else:

        st.success(
            "No major missing skills found."
        )


    # ========================================================
    # RESUME SKILLS
    # ========================================================

    st.header("📄 Skills Found in Your Resume")

    if resume_skills:

        for skill in resume_skills:

            st.write(f"• {skill}")

    else:

        st.write("No predefined skills detected.")


    # ========================================================
    # JOB DESCRIPTION SKILLS
    # ========================================================

    st.header("💼 Skills Detected in Job Description")

    if job_skills:

        for skill in job_skills:

            st.write(f"• {skill}")

    else:

        st.write(
            "No predefined skills detected in the job description."
        )


    # ========================================================
    # ATS KEYWORDS
    # ========================================================

    st.divider()

    st.header("🔑 ATS Keyword Analysis")

    ats_col1, ats_col2 = st.columns(2)

    with ats_col1:

        st.metric(
            "Keyword Match Score",
            f"{keyword_score}%"
        )

    with ats_col2:

        st.metric(
            "Matching Keywords",
            len(matching_keywords)
        )

    if matching_keywords:

        st.write("**Matching Keywords:**")

        st.write(
            ", ".join(matching_keywords)
        )


    # ========================================================
    # RESUME SECTION ANALYSIS
    # ========================================================

    st.divider()

    st.header("📋 Resume Section Analysis")

    section_col1, section_col2 = st.columns(2)

    with section_col1:

        for section, found in sections.items():

            if found:

                st.success(
                    f"✅ {section}"
                )

            else:

                st.warning(
                    f"⚠️ {section} not detected"
                )


    # ========================================================
    # SKILL COMPARISON
    # ========================================================

    st.divider()

    st.header("📈 Skill Comparison")

    chart_data = {
        "Category": [
            "Matching Skills",
            "Missing Skills"
        ],

        "Count": [
            len(matching_skills),
            len(missing_skills)
        ]
    }

    st.bar_chart(
        chart_data,
        x="Category",
        y="Count"
    )


    # ========================================================
    # IMPROVEMENT SUGGESTIONS
    # ========================================================

    st.divider()

    st.header("💡 Resume Improvement Suggestions")

    st.write(
        "Consider highlighting the following improvements "
        "only when they genuinely reflect your skills or experience:"
    )

    for suggestion in suggestions:

        st.info(
            f"➡️ {suggestion}"
        )


    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    st.divider()

    st.header("🎯 Analysis Summary")

    st.write(summary)


    # ========================================================
    # EXTRACTED RESUME TEXT
    # ========================================================

    st.divider()

    with st.expander("📖 View Extracted Resume Text"):

        st.text_area(
            "Extracted Text",
            resume_text,
            height=400
        )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.divider()

    st.header("📥 Download Analysis Report")

    report = f"""
AI RESUME ANALYZER
==================

OVERALL MATCH SCORE
{overall_score}%

NLP SIMILARITY
{similarity_score}%

KEYWORD MATCH
{keyword_score}%

SKILL MATCH
{skill_score}%

MATCHING SKILLS
---------------
{chr(10).join("- " + skill for skill in matching_skills)}

MISSING SKILLS
-------------
{chr(10).join("- " + skill for skill in missing_skills)}

RESUME SKILLS
-------------
{chr(10).join("- " + skill for skill in resume_skills)}

JOB DESCRIPTION SKILLS
----------------------
{chr(10).join("- " + skill for skill in job_skills)}

ATS MATCHING KEYWORDS
---------------------
{chr(10).join("- " + keyword for keyword in matching_keywords)}

ANALYSIS SUMMARY
----------------
{summary}

IMPROVEMENT SUGGESTIONS
-----------------------
{chr(10).join("- " + suggestion for suggestion in suggestions)}
"""

    st.download_button(
        label="📥 Download Analysis Report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        use_container_width=True
    )
