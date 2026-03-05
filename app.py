from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
import os
import resume_parser  # Your custom module
from dotenv import load_dotenv
import uuid
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
import traceback
import time
from functools import wraps
import requests
import re
import models
# Load environment variables from .env file
load_dotenv()

# Configure HuggingFace API Key
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Note: Ensure HUGGINGFACEHUB_API_TOKEN is in your .env file

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'resumeai-dev-secret-key-change-in-production')
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for debugging

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Limit file size to 2MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
models.init_db()

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

# --- AUTH HELPERS ---

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "auth_required", "message": "Please log in to continue."}), 401
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    """Return the current logged-in user or None."""
    user_id = session.get('user_id')
    if user_id:
        return models.get_user_by_id(user_id)
    return None

# --- ROUTES ---

@app.route('/')
def home():
    user = get_current_user()
    return render_template('index.html', user=user)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        if not name or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('signup.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')

        # Basic email format check
        if '@' not in email or '.' not in email:
            flash('Please enter a valid email address.', 'error')
            return render_template('signup.html')

        # Create user
        user_id = models.create_user(name, email, password)
        if user_id is None:
            flash('An account with this email already exists.', 'error')
            return render_template('signup.html')

        # Log the user in
        session['user_id'] = user_id
        session['user_name'] = name
        flash('Account created successfully! Welcome to ResumeAI.', 'success')
        return redirect(url_for('home'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('login.html')

        user = models.get_user_by_email(email)
        if user is None or not models.verify_password(user['password_hash'], password):
            flash('Invalid email or password.', 'error')
            return render_template('login.html')

        # Set session
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        flash(f'Welcome back, {user["name"]}!', 'success')
        return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/check-auth')
def check_auth():
    """API endpoint for frontend to check auth state and free usage."""
    user_id = session.get('user_id')
    if user_id:
        user = models.get_user_by_id(user_id)
        return jsonify({
            "authenticated": True,
            "user_name": user['name'] if user else None,
            "analysis_count": user['resume_analysis_count'] if user else 0
        })
    else:
        free_used = session.get('free_usage_used', False)
        return jsonify({
            "authenticated": False,
            "free_usage_used": free_used
        })

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
    Requires authentication after the first free analysis.
    """
    # --- ACCESS CONTROL ---
    user_id = session.get('user_id')
    is_authenticated = user_id is not None

    if not is_authenticated:
        # Anonymous user — check if free analysis already used
        if session.get('free_usage_used', False):
            return jsonify({
                "error": "auth_required",
                "message": "You have used your free resume analysis. Please sign up or log in to continue using ResumeAI."
            }), 401

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

        # --- TRACK USAGE ---
        if is_authenticated:
            models.increment_analysis_count(user_id)
            # Save analysis to history
            models.save_analysis(user_id, job_role, file.filename, text, analysis)
        else:
            # Mark free usage as consumed for anonymous user
            session['free_usage_used'] = True

        return jsonify({"analysis": analysis})

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during analysis."}), 500
    
# --- HISTORY & COMPARISON ROUTES ---

@app.route('/history')
def history_page():
    """Render the analysis history page (login required)."""
    user_id = session.get('user_id')
    if not user_id:
        flash('Please log in to view your analysis history.', 'error')
        return redirect(url_for('login'))
    user = models.get_user_by_id(user_id)
    return render_template('history.html', user=user)

@app.route('/api/history')
def api_history():
    """JSON API: return all analyses for the logged-in user."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "auth_required"}), 401
    analyses = models.get_user_analyses(user_id)
    return jsonify({"analyses": analyses})

@app.route('/api/analysis/<int:analysis_id>')
def api_analysis_detail(analysis_id):
    """JSON API: return a single analysis by id."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "auth_required"}), 401
    analysis = models.get_analysis_by_id(analysis_id, user_id)
    if not analysis:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify({"analysis": analysis})

@app.route('/api/compare', methods=['POST'])
@rate_limit(limit=5, per=60)
def api_compare():
    """Run a new analysis and return it alongside a previous one for comparison."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "auth_required"}), 401

    old_id = request.form.get('old_analysis_id')
    if not old_id:
        return jsonify({"error": "Previous analysis ID is required"}), 400

    old_analysis = models.get_analysis_by_id(int(old_id), user_id)
    if not old_analysis:
        return jsonify({"error": "Previous analysis not found"}), 404

    if 'resume' not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    file = request.files['resume']
    job_role = request.form.get('job_role', old_analysis['job_role'])

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. PDF or DOCX only."}), 400

    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{original_filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    try:
        if file_path.endswith('.docx'):
            text = resume_parser.extract_text_from_docx(file_path)
        else:
            text = resume_parser.extract_text_from_pdf(file_path)

        text = text.replace('\x00', '')
        new_analysis_text = analyze_resume_with_llm(text, job_role)
        os.remove(file_path)

        # Save the new analysis
        models.increment_analysis_count(user_id)
        new_id = models.save_analysis(user_id, job_role, file.filename, text, new_analysis_text)
        new_analysis = models.get_analysis_by_id(new_id, user_id)

        return jsonify({
            "old": old_analysis,
            "new": new_analysis
        })

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during comparison."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
