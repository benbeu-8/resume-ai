import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resumeai.db')


def get_db():
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            free_usage_used BOOLEAN DEFAULT 0,
            resume_analysis_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_role TEXT NOT NULL,
            filename TEXT NOT NULL,
            resume_text TEXT NOT NULL,
            analysis_result TEXT NOT NULL,
            score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()


def create_user(name, email, password):
    """Create a new user. Returns the user id on success, None if email exists."""
    conn = get_db()
    try:
        cursor = conn.execute(
            'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
            (name, email, generate_password_hash(password))
        )
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_email(email):
    """Retrieve a user by email address."""
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    """Retrieve a user by their id."""
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user


def mark_free_usage(user_id):
    """Mark that a user has used their free analysis."""
    conn = get_db()
    conn.execute('UPDATE users SET free_usage_used = 1 WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()


def increment_analysis_count(user_id):
    """Increment the resume analysis count for a user."""
    conn = get_db()
    conn.execute(
        'UPDATE users SET resume_analysis_count = resume_analysis_count + 1 WHERE id = ?',
        (user_id,)
    )
    conn.commit()
    conn.close()


def verify_password(stored_hash, password):
    """Verify a password against a stored hash."""
    return check_password_hash(stored_hash, password)


# --- ANALYSIS HISTORY ---

def _extract_score(analysis_text):
    """Try to extract a numeric score from AI analysis output."""
    import re
    # Look for patterns like "87%", "Score: 75%", "ATS Score: 92%"
    patterns = [
        r'(?:ATS|ats|score|Score|match|Match)[:\s]*(\d{1,3})\s*%',
        r'(\d{1,3})\s*%\s*(?:ATS|ats|score|match|compatibility)',
        r'(\d{1,3})\s*(?:out of 100|/100)',
        r'(?:overall|total|final)[\s\w]*?(\d{1,3})\s*%',
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis_text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return val
    return None


def save_analysis(user_id, job_role, filename, resume_text, analysis_result):
    """Save an analysis result and return the analysis id."""
    score = _extract_score(analysis_result)
    conn = get_db()
    cursor = conn.execute(
        '''INSERT INTO analyses (user_id, job_role, filename, resume_text, analysis_result, score)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (user_id, job_role, filename, resume_text, analysis_result, score)
    )
    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()
    return analysis_id


def get_user_analyses(user_id, limit=50):
    """Get all analyses for a user, newest first."""
    conn = get_db()
    rows = conn.execute(
        '''SELECT id, job_role, filename, score, created_at
           FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?''',
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id, user_id):
    """Get a full analysis by id, only if it belongs to the user."""
    conn = get_db()
    row = conn.execute(
        'SELECT * FROM analyses WHERE id = ? AND user_id = ?',
        (analysis_id, user_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None
