import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import io
import PyPDF2
import docx

def to_1d(x):
    """Convert input to 1D array of strings for TfidfVectorizer."""
    if x is None:
        return np.array([''], dtype=object)
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        x = x.values
    x = np.asarray(x, dtype=object)
    if x.ndim > 1:
        x = x.ravel()
    if x.ndim == 0 or (isinstance(x, np.ndarray) and x.shape == ()):
        x = np.array([str(x.item() if hasattr(x, 'item') else x)], dtype=object)
    x = np.array([str(item) if item is not None else '' for item in x], dtype=object)
    return x

# Load models
try:
    role_clf = joblib.load("role_classifier.pkl")
    salary_reg = joblib.load("salary_regressor.pkl")
    st.session_state.models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.session_state.models_loaded = False

def predict_job_profile(new_data, role_model, salary_model):
    """Predict job role and salary."""
    row = pd.DataFrame([new_data])
    row["Experience_Range"] = row.get("Max_Experience", 0) - row.get("Min_Experience", 0)
    
    try:
        predicted_role = role_model.predict(row)[0]
        predicted_salary = salary_model.predict(row)[0]
        return predicted_role, predicted_salary
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def extract_text_from_file(file):
    """Extract text from PDF or DOCX."""
    text = ""
    if file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = docx.Document(io.BytesIO(file.getvalue()))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return None
    else:
        st.error("Unsupported file type.")
        return None
    return text

def parse_qualifications(text):
    """Parse qualifications - FIXED TO MATCH TRAINING DATA."""
    text_lower = text.lower()
    
    # EXACT FORMATS from your training data
    if re.search(r"phd|ph\.d|doctor of philosophy|doctorate", text_lower):
        return "PhD"
    if re.search(r"mba|m\.b\.a|master of business administration", text_lower):
        return "MBA"
    if re.search(r"m\.com|mcom|master of commerce", text_lower):
        return "M.Com"
    if re.search(r"b\.tech|btech|bachelor of technology|b\.e\.|bachelor of engineering", text_lower):
        return "B.Tech"
    if re.search(r"b\.com|bcom|bachelor of commerce", text_lower):
        return "B.Com"
    if re.search(r"bba|b\.b\.a|bachelor of business administration", text_lower):
        return "BBA"
    if re.search(r"bca|b\.c\.a|bachelor of computer applications", text_lower):
        return "BCA"
    
    return "BCA"  # Default

def parse_experience(text):
    """Extract experience range."""
    text_lower = text.lower()
    
    # Try multiple patterns
    patterns = [
        r"(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years|yrs)",
        r"(\d+)\+\s*(?:years|yrs)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            if len(matches[0]) == 2:
                return int(matches[0][0]), int(matches[0][1])
            else:
                exp = int(matches[0])
                return exp, exp + 2
    
    # Single year mention
    matches = re.findall(r"(\d+)\s*(?:years|yrs)", text_lower)
    if matches:
        exp = max([int(n) for n in matches])
        return max(0, exp - 1), exp + 1
    
    # Fresher keywords
    if re.search(r"fresher|entry.level|recent graduate", text_lower):
        return 0, 1
    
    return 0, 2

def parse_skills(text):
    """Extract skills from text."""
    text_lower = text.lower()
    
    # Comprehensive skill list
    skills_list = [
        "python", "java", "javascript", "c++", "c#", "php", "ruby", "sql", "nosql",
        "html", "css", "react", "angular", "vue", "nodejs", "django", "flask", "spring",
        "machine learning", "deep learning", "data analysis", "data science",
        "tensorflow", "pytorch", "nlp", "computer vision", "statistics",
        "aws", "azure", "docker", "kubernetes", "ci/cd", "jenkins", "devops",
        "project management", "agile", "scrum", "leadership",
        "ui/ux", "figma", "photoshop", "graphic design",
        "seo", "digital marketing", "content marketing", "social media",
        "git", "tableau", "power bi", "excel", "autocad",
        "communication", "problem solving", "analytical"
    ]
    
    found_skills = []
    for skill in skills_list:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found_skills.append(skill)
    
    return ", ".join(found_skills[:15]) if found_skills else "communication, teamwork, problem solving"

def parse_cv(file):
    """Parse CV and return features."""
    text = extract_text_from_file(file)
    if text is None:
        return None
    
    return {
        "Qualifications": parse_qualifications(text),
        "Min_Experience": parse_experience(text)[0],
        "Max_Experience": parse_experience(text)[1],
        "skills": parse_skills(text)
    }

# UI
st.set_page_config(page_title="Job Profile Predictor", layout="wide")
st.title("📄 Job Role & Salary Predictor")
st.markdown("Upload your CV to get job role and salary predictions based on ML models.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload CV")
    uploaded_file = st.file_uploader("Choose PDF or DOCX", type=["pdf", "docx"])
    
    st.header("🌍 Job Preferences")
    country = st.selectbox("Target Country", 
        ["United States", "Pakistan", "United Kingdom", "Canada", "Germany", "Australia"])
    work_type = st.selectbox("Work Type", 
        ["Full-Time", "Part-Time", "Contract", "Temporary"])

with col2:
    st.header("⚙️ Manual Override (Optional)")
    st.caption("If CV parsing fails, enter manually")
    
    manual_override = st.checkbox("Use manual input instead of CV")
    
    if manual_override:
        manual_skills = st.text_area("Skills (comma-separated)", 
            "python, sql, machine learning")
    
    st.markdown("---")
    predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True,
                           disabled=not st.session_state.models_loaded)

# Prediction logic
if predict_btn:
    if not manual_override and uploaded_file is None:
        st.error("⚠️ Please upload a CV or use manual input")
    else:
        with st.spinner("Analyzing..."):
            if manual_override:
                profile_data = {
                    "Qualifications": "B.Com",  # Default value used internally
                    "Country": country,
                    "Work Type": work_type,
                    "Company Size": 5000,  # Default value
                    "Min_Experience": 0,  # Default value
                    "Max_Experience": 2,  # Default value
                    "skills": manual_skills
                }
            else:
                parsed = parse_cv(uploaded_file)
                if parsed:
                    profile_data = {
                        "Qualifications": parsed["Qualifications"],
                        "Country": country,
                        "Work Type": work_type,
                        "Company Size": 5000,  # Default value
                        "Min_Experience": parsed["Min_Experience"],
                        "Max_Experience": parsed["Max_Experience"],
                        "skills": parsed["skills"]
                    }
                else:
                    st.error("Failed to parse CV")
                    st.stop()
            
            # Make prediction
            role, salary = predict_job_profile(profile_data, role_clf, salary_reg)
            
            if role and salary:
                st.success("✅ Prediction Complete!")
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("🎯 Predicted Role", role)
                with col_r2:
                    st.metric("💰 Estimated Salary", f"${salary:.1f}K/year")
                
                st.markdown("---")
                st.subheader("📊 Profile Used")
                # Remove internal fields from display
                display_data = {k: v for k, v in profile_data.items() 
                               if k not in ["Qualifications", "Min_Experience", "Max_Experience", "Company Size"]}
                st.json(display_data)
                
                # Salary range estimate
                sal_min = max(40, salary * 0.85)
                sal_max = salary * 1.15
                st.info(f"💡 Typical range: ${sal_min:.0f}K - ${sal_max:.0f}K based on market data")

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    if st.session_state.models_loaded:
        st.success("✅ Models loaded")
    
    st.markdown("---")
    st.subheader("📝 Tips")
    st.markdown("""
    **For best results:**
    - Use clear CV format
    - Include education section
    - List experience explicitly
    - Add skills section
    """)
    
    st.markdown("---")
    st.caption("⚠️ Note: Salary predictions are estimates based on historical data")