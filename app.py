from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import resume_parser  # Your custom module
from dotenv import load_dotenv
import uuid
from werkzeug.utils import secure_filename
import traceback
import time
from functools import wraps
import requests
import json
import re
# Load environment variables from .env file
load_dotenv()

# Configure HuggingFace API Key
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Note: Ensure HUGGINGFACEHUB_API_TOKEN is in your .env file

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for debugging

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Limit file size to 2MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- SECURITY & UTILS ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Basic in-memory rate limiter
request_history = {}

def rate_limit(limit=5, per=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            ip = request.remote_addr
            now = time.time()
            if ip not in request_history:
                request_history[ip] = []
            # Filter out requests older than 'per' seconds
            request_history[ip] = [t for t in request_history[ip] if now - t < per]
            if len(request_history[ip]) >= limit:
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
            request_history[ip].append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator

def analyze_resume_with_llm(resume_text, job_role):
    """
    Strict LLM wrapper function.
    Only this function calls the model.
    """
    # Removed API key requirement as requested by user
    # if not hf_api_key:
    #     return "Error: LLM API key not configured."

    try:
        if not hf_api_key or hf_api_key.strip() == "":
            return (
                "## AI Resume Analysis (Simulated Open Source Fallback)\n\n"
                "*Note: No valid HuggingFace API key was provided. This is a simulated response based on your formatting requirements limit.*\n\n"
                "### Strengths\n"
                "- The resume format appears clean and parsable by ATS systems.\n"
                "- Your skills section aligns moderately with the target role.\n\n"
                "### Areas for Improvement\n"
                "- **Quantify Impact**: Add more metrics (e.g., 'increased performance by X%') to your experience bullets.\n"
                "- **Keyword Optimization**: Ensure keywords from the exact job description for this role are heavily featured.\n"
                "- **Action Verbs**: Start bullet points with strong action verbs (Developed, Spearheaded, Optimized).\n"
            )

        # Truncate text to avoid hitting context length limits
        truncated_resume = resume_text[:3000]
        
        # Using the standard OpenAI-compatible Hugging Face Serverless Router API
        api_url = "https://router.huggingface.co/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [
                {"role": "system", "content": "You are a professional ATS resume optimization assistant. Analyze the resume against the target job role and provide a detailed, structured report. Your report MUST include ALL of the following sections: 1) STRENGTHS - What the resume does well for this role. 2) FLAWS & WEAKNESSES - Specific problems, gaps, or issues in the resume that could hurt the candidate. 3) MISSING ELEMENTS - Specific skills, keywords, certifications, or sections that are absent but expected for this role. 4) ACTIONABLE ADDITIONS - Concrete suggestions on exactly what to add, rewrite, or restructure to maximize shortlisting chances. Be specific and mention exact keywords, phrases, tools, or technologies to include. 5) ATS OPTIMIZATION TIPS - Formatting and keyword tips to pass Applicant Tracking Systems. Do not provide coding help or answer general questions."},
                {"role": "user", "content": f"Analyze this resume for the job role: {job_role}. Identify all flaws, missing elements, and provide specific suggestions on what to add to maximize chances of getting shortlisted.\n\nResume content:\n{truncated_resume}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"HuggingFace API Error: {response.text}")
            return f"Error: API returned status code {response.status_code}. Response: {response.text}"
            
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            ai_text = result['choices'][0]['message']['content'].strip()
            # Strip DeepSeek's <think>...</think> chain-of-thought tags for clean output
            ai_text = re.sub(r'<think>.*?</think>', '', ai_text, flags=re.DOTALL).strip()
            return ai_text
        else:
            return "Error: Unexpected API response format."

    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error: Could not complete analysis. Check server console for details."

# --- ROUTES ---

@app.route('/')
def home():
    # This serves the 'index.html' file from the 'templates' folder
    return render_template('index.html')

@app.route('/parse', methods=['POST'])
def parse_resume():
    """
    Recruiter Mode Endpoint: Parses resume and extracts structured data.
    """
    if 'resumes' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['resumes']
    # Optional: Recruiter might paste a JD to compare, but parsing is primary
    job_description = request.form.get('job_description', '') 

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. PDF or DOCX only."}), 400

    # Securely save
    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{original_filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    try:
        # 1. Extract Text based on extension
        if file_path.endswith('.docx'):
            text = resume_parser.extract_text_from_docx(file_path)
        else:
            text = resume_parser.extract_text_from_pdf(file_path)

        # 2. Extract Details
        details = resume_parser.extract_details(text)
        
        # 3. Calculate Score if JD provided
        score = 0
        matches = []
        missing = []
        
        if job_description:
            jd_skills = resume_parser.extract_skills_list(job_description)
            score, matches, missing = resume_parser.calculate_match_score(details['skills_list'], jd_skills)

        # 4. Clean up
        os.remove(file_path)

        result = {
            "filename": file.filename,
            "email": details.get('email', 'Not Found'),
            "skills": details.get('skills_list', []),
            "score": score,
            "matched_skills": matches,
            "missing_skills": missing,
            "raw_text_snippet": text[:500] + "..." # Preview
        }

        return jsonify({"results": [result]})

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
@rate_limit(limit=5, per=60) # 6️⃣ Security: Rate limiting
def analyze_resume_endpoint():
    """
    Student Mode Endpoint: Analyzes resume against a job role using LLM.
    """
    if 'resume' not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400
    
    file = request.files['resume']
    job_role = request.form.get('job_role', '')

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. PDF or DOCX only."}), 400

    if not job_role:
        return jsonify({"error": "Target Job Role is mandatory"}), 400

    # Securely save
    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{original_filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    try:
        # 1. Extract Text
        if file_path.endswith('.docx'):
            text = resume_parser.extract_text_from_docx(file_path)
        else:
            text = resume_parser.extract_text_from_pdf(file_path)

        # 6️⃣ Security: Sanitize extracted text (basic)
        text = text.replace('\x00', '') # Remove null bytes

        # 2. Call Backend LLM Guard
        analysis = analyze_resume_with_llm(text, job_role)

        # 3. Clean up
        os.remove(file_path)

        return jsonify({"analysis": analysis})

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during analysis."}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
