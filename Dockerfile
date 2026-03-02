# 1. Start with a lightweight Python 3.12 installation (same as your PC)
FROM python:3.12-slim

# 2. Set the folder inside the container where your app will live
WORKDIR /app

# 3. Copy your "requirements.txt" file into the container first
#    (We do this separately to make re-building faster)
COPY requirements.txt .

# 4. Install the libraries (Flask, spaCy, Google Gemini, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download the English language brain for spaCy (CRITICAL STEP)
RUN python -m spacy download en_core_web_sm

# 6. Now, copy the rest of your code (app.py, templates folder, etc.)
COPY . .

# 7. Create the uploads folder inside the container so it doesn't crash
RUN mkdir -p uploads

# 8. Tell Docker that this app runs on port 5000
EXPOSE 5000

# 9. The command to start your website when the container turns on
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]