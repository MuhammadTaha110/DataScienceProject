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

# =====================================================================
# 1. LOAD MODELS (From your teammate's notebook)
# =====================================================================

try:
    # Load the three model files
    role_clf = joblib.load("role_classifier.pkl")
    salary_reg = joblib.load("salary_regressor.pkl")
    # The preprocessor is loaded inside the pipelines, 
    # so we just need the main model files.
    st.session_state.models_loaded = True
except FileNotFoundError as e:
    st.error(f"Error loading model file: {e}")
    st.error("Please make sure 'role_classifier.pkl' and 'salary_regressor.pkl' are in the same directory.")
    st.session_state.models_loaded = False
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")
    st.session_state.models_loaded = False

# =====================================================================
# 2. PREDICTION FUNCTION (From your teammate's notebook)
# =====================================================================

def predict_job_profile(new_data, role_model, salary_model):
    """
    Predict the ideal job role and corresponding salary range for a new profile.
    
    Parameters:
    -----------
    new_data : dict
        Dictionary containing the profile features
    role_model : Pipeline
        Trained role classification model
    salary_model : Pipeline
        Trained salary regression model
    
    Returns:
    --------
    tuple : (predicted_role, predicted_salary)
    """
    # Convert single dictionary to DataFrame
    row = pd.DataFrame([new_data])

    # Feature Engineering (must match the notebook)
    row["Experience_Range"] = row.get("Max_Experience", 0) - row.get("Min_Experience", 0)
  
    # Make predictions
    predicted_role = role_model.predict(row)[0]
    predicted_salary = salary_model.predict(row)[0]
    
    return predicted_role, predicted_salary

# =====================================================================
# 3. CV PARSING LOGIC (New Component)
# =====================================================================

def extract_text_from_file(file):
    """Extracts raw text from an uploaded file (PDF or DOCX)."""
    text = ""
    if file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
            for page in pdf_reader.pages:
                text += page.extract_text()
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
        st.error("Unsupported file type. Please upload a PDF or DOCX.")
        return None
    return text.lower()

def parse_qualifications(text):
    """Finds the highest qualification from text."""
    # This list should be expanded based on the data in 'jobs.csv'
    qual_patterns = {
        "phd": r"phd|doctor of philosophy",
        "mba": r"mba|master of business administration",
        "m.com": r"m\.com|master of commerce",
        "b.tech": r"b\.tech|bachelor of technology",
        "b.com": r"b\.com|bachelor of commerce",
        "bba": r"bba|bachelor of business administration",
        "bca": r"bca|bachelor of computer applications"
    }
    # Check in order of hierarchy (e.g., PhD first)
    for qual, pattern in qual_patterns.items():
        if re.search(pattern, text):
            return qual.upper()
    return "Not Found" # Default

def parse_experience(text):
    """Estimates min and max experience from text."""
    # This is a simple estimator. Real-world parsing is more complex.
    matches = re.findall(r"(\d+)\s*to\s*(\d+)\s*years", text)
    if matches:
        nums = [int(n) for n in matches[0]]
        return min(nums), max(nums)
    
    matches = re.findall(r"(\d+)\s*-\s*(\d+)\s*years", text)
    if matches:
        nums = [int(n) for n in matches[0]]
        return min(nums), max(nums)

    matches = re.findall(r"(\d+)\s*years", text)
    if matches:
        nums = [int(n) for n in matches]
        exp = max(nums)
        return exp - 1, exp # Assume a 1-year range
    
    return 0, 1 # Default for entry-level if nothing is found

def parse_skills(text):
    """Finds a list of skills from text."""
    # This list should be built from the 'skills' column in your dataset
    skills_list = [
        "python", "sql", "data analysis", "project management", "machine learning",
        "java", "javascript", "html", "css", "react", "nodejs",
        "deep learning", "tensorflow", "pytorch", "nlp", "computer vision", "research",
        "aws", "docker", "kubernetes", "devops", "ci/cd", "supplier diversity",
        "autocad", "2d modeling", "3d modeling", "art education", "curriculum",
        "environmental impact analysis", "data collection", "sap", "power bi"
    ]
    
    found_skills = []
    for skill in skills_list:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found_skills.append(skill)
    
    return ", ".join(found_skills) if found_skills else "Not Found"

def parse_cv(file):
    """Main function to parse the CV and return a features dictionary."""
    text = extract_text_from_file(file)
    if text is None:
        return None
    
    qualifications = parse_qualifications(text)
    min_exp, max_exp = parse_experience(text)
    skills = parse_skills(text)
    
    # Check if parsing was successful
    if qualifications == "Not Found" and skills == "Not Found":
        st.warning("Could not extract Qualifications or Skills from the CV. Please check the file.")
        
    return {
        "Qualifications": qualifications,
        "Min_Experience": min_exp,
        "Max_Experience": max_exp,
        "skills": skills
    }

# =====================================================================
# 4. STREAMLIT UI LAYOUT
# =====================================================================

st.set_page_config(page_title="Job Profile Predictor", layout="wide")
st.title("📄 Job Role & Salary Predictor")
st.markdown("""
Upload a CV to automatically extract skills, qualifications, and experience. 
Then, fill in the remaining details to predict an ideal job role and estimated average salary.
""")

st.markdown("---")

# Use two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Upload Your CV")
    uploaded_file = st.file_uploader("Upload a .pdf or .docx file", type=["pdf", "docx"])
    
    st.header("Step 2: Add Profile Details")
    st.caption("These features are also required by the model.")
    
    # These inputs are required by the model but are not usually in a CV
    country = st.text_input("Your Target Country", "United States")
    work_type = st.selectbox("Preferred Work Type", ["Full-Time", "Part-Time", "Contract", "Temporary"])
    company_size = st.number_input("Target Company Size (e.g., 1000)", min_value=1, value=1000)

with col2:
    st.header("Step 3: Predict")
    predict_button = st.button("Analyze CV and Predict", type="primary", use_container_width=True, disabled=not st.session_state.models_loaded)
    
    st.markdown("---")
    
    # This area will show the results
    result_container = st.container(border=True)

# =====================================================================
# 5. UI INTERACTION LOGIC
# =====================================================================

if predict_button:
    if uploaded_file is None:
        st.error("Please upload a CV file first.")
    else:
        with st.spinner("Analyzing CV and running predictions..."):
            # 1. Parse CV
            parsed_data = parse_cv(uploaded_file)
            
            if parsed_data:
                # 2. Combine with form data
                profile_data = {
                    "Qualifications": parsed_data["Qualifications"],
                    "Country": country,
                    "Work Type": work_type,
                    "Company Size": company_size,
                    "Min_Experience": parsed_data["Min_Experience"],
                    "Max_Experience": parsed_data["Max_Experience"],
                    "skills": parsed_data["skills"]
                }
                
                # 3. Make prediction
                try:
                    role, salary = predict_job_profile(profile_data, role_clf, salary_reg)
                    
                    # 4. Display results
                    with result_container:
                        st.subheader("Prediction Results")
                        st.markdown(f"### Ideal Job Role: **{role}**")
                        st.markdown(f"### Predicted Avg. Salary: **${salary:,.2f}K**")
                        
                        st.markdown("---")
                        st.subheader("Extracted CV Data")
                        st.text_area("Parsed Skills:", value=profile_data['skills'], height=100)
                        st.text(f"Parsed Qualification: {profile_data['Qualifications']}")
                        st.text(f"Parsed Experience: {profile_data['Min_Experience']}-{profile_data['Max_Experience']} years")

                except Exception as e:
                    with result_container:
                        st.error(f"An error occurred during prediction: {e}")
                        st.error("This may be due to the preprocessor.pkl file not matching the pipelines.")

# Add a warning about the salary model based on the notebook's output
if st.session_state.models_loaded:
    st.sidebar.title("Project Notes")
    st.sidebar.warning(
        "**Important:** Based on your notebook's output (R² ≈ -0.006), the **Salary Prediction Model** has very low accuracy and is not reliable. "
        "It appears to be predicting a similar average value for all inputs. "
        "The **Role Prediction Model**, however, reported 100% accuracy and should be functional."
    )