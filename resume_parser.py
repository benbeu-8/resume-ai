import pdfplumber
import spacy
import re 
import docx

# Load the "brain" (the pre-trained English model)
# This model knows what English words look like, what verbs are, etc.
nlp = spacy.load("en_core_web_sm")

# Extensive Database of Skills
SKILLS_DB = [
    # Programming Languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript", "ruby", "php", "swift", "kotlin", "go", "rust", "scala", "r", "matlab", "perl", "lua", "dart", "objective-c", "assembly", "shell", "bash", "powershell", "vba", "cobol", "fortran", "ada", "lisp", "haskell", "erlang", "elixir", "clojure", "f#", "groovy", "julia", "visual basic", "sas", "abap", "apex", "solidity", "pl/sql", "t-sql", "scheme", "prolog", "scratch", "logo", "smalltalk", "ocaml", "racket", "alice", "actionscript", "awk", "sed", "vhdl", "verilog", "systemverilog",
    # Web Development
    "html", "html5", "css", "css3", "sass", "less", "bootstrap", "tailwind", "material ui", "chakra ui", "ant design", "foundation", "bulma", "react", "react.js", "angular", "angularjs", "vue", "vue.js", "svelte", "jquery", "backbone.js", "ember.js", "knockout.js", "polymer", "aurelia", "meteor", "preact", "alpine.js", "lit", "stimulus", "next.js", "nuxt.js", "gatsby", "gridsome", "jekyll", "hugo", "eleventy", "webpack", "parcel", "rollup", "vite", "babel", "eslint", "prettier", "npm", "yarn", "pnpm", "bower", "gulp", "grunt", "webassembly", "wasm", "pwa", "service workers", "local storage", "indexeddb", "websockets", "webrtc", "canvas", "svg", "webgl", "three.js", "d3.js", "chart.js", "leaflet", "mapbox",
    "node.js", "express", "express.js", "nestjs", "koa", "sails.js", "fastify", "django", "flask", "fastapi", "pyramid", "bottle", "tornado", "ruby on rails", "sinatra", "hanami", "laravel", "symfony", "codeigniter", "yii", "cakephp", "zend framework", "phalcon", "spring", "spring boot", "jakarta ee", "hibernate", "struts", "grails", "play framework", "ktor", "asp.net", "asp.net core", "entity framework", "blazor", "phoenix", "gin", "echo", "fiber", "beego", "revel", "buffalo", "actix", "rocket", "warp", "axum", "vapor", "kitura", "perfect", "graphql", "apollo", "prisma", "sequelize", "typeorm", "mongoose",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "sql server", "mssql", "mariadb", "mongodb", "cassandra", "redis", "elasticsearch", "dynamodb", "firestore", "couchdb", "couchbase", "neo4j", "hbase", "hive", "snowflake", "bigquery", "redshift", "teradata", "db2", "informix", "sybase", "ingres", "firebird", "h2", "derby", "hsqldb", "rocksdb", "leveldb", "berkeleydb", "memcached", "etcd", "zookeeper", "consul", "cockroachdb", "tidb", "voltdb", "scylladb", "arangodb", "orientdb", "ravendb", "realm", "firebase", "supabase", "planetscale",
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "microsoft azure", "google cloud", "gcp", "ibm cloud", "oracle cloud", "alibaba cloud", "digitalocean", "linode", "vultr", "heroku", "netlify", "vercel", "render", "railway", "fly.io", "docker", "kubernetes", "k8s", "openshift", "rancher", "nomad", "swarm", "mesos", "jenkins", "gitlab ci", "github actions", "circleci", "travis ci", "bamboo", "teamcity", "azure devops", "aws codepipeline", "ansible", "terraform", "pulumi", "cloudformation", "puppet", "chef", "saltstack", "vagrant", "packer", "prometheus", "grafana", "elk", "elastic stack", "splunk", "datadog", "new relic", "pagerduty", "opsgenie", "victorops", "nagios", "zabbix", "icinga", "cacti", "wireshark", "tcpdump", "nginx", "apache", "httpd", "caddy", "traefik", "envoy", "haproxy", "istio", "linkerd", "kong", "tyk", "zuul", "eureka", "ribbon", "hystrix", "archiva", "nexus", "artifactory", "sonarqube",
    # AI / ML / Data
    "machine learning", "deep learning", "ai", "artificial intelligence", "data science", "data engineering", "big data", "hadoop", "spark", "kafka", "flink", "storm", "samza", "beam", "airflow", "luigi", "prefect", "dagster", "nifi", "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "torch", "mxnet", "caffe", "theano", "cntk", "xgboost", "lightgbm", "catboost", "opencv", "nltk", "spacy", "gensim", "hugging face", "transformers", "bert", "gpt", "llm", "langchain", "llamaindex", "pinecone", "weaviate", "milvus", "chroma", "faiss", "matplotlib", "seaborn", "plotly", "bokeh", "altair", "tableau", "power bi", "looker", "qlik", "domo", "sisense", "thoughtspot", "microstrategy", "cognos", "sas", "spss", "stata", "rapidminer", "knime", "alteryx", "databricks", "sagemaker", "vertex ai", "azure ml", "mlflow", "kubeflow", "tfx",
    # Mobile
    "android", "ios", "react native", "flutter", "xamarin", "ionic", "cordova", "phonegap", "capacitor", "native script", "titanium", "corona", "unity", "unreal", "godot", "cocos2d", "libgdx", "monogame", "cryengine", "lumberyard", "swiftui", "uikit", "cocoa touch", "jetpack compose", "kotlin multiplatform", "kmm", "objective-c", "swift", "java", "kotlin", "dart",
    # Security
    "cybersecurity", "network security", "app security", "web security", "cloud security", "penetration testing", "ethical hacking", "vulnerability assessment", "incident response", "forensics", "malware analysis", "reverse engineering", "cryptography", "pki", "ssl", "tls", "ssh", "vpn", "ipsec", "firewall", "ids", "ips", "siem", "soc", "dlp", "iam", "pam", "sso", "mfa", "2fa", "oauth", "openid", "jwt", "saml", "ldap", "kerberos", "active directory", "group policy", "owasp", "burp suite", "metasploit", "nmap", "nessus", "qualys", "wireshark", "kali", "parrot", "tails", "qubes", "tor", "i2p", "freenet", "gdpr", "ccpa", "hipaa", "pci dss", "iso 27001", "soc 2", "nist", "cis",
    # OS & Networking
    "linux", "unix", "windows", "macos", "android", "ios", "ubuntu", "debian", "centos", "rhel", "fedora", "arch linux", "manjaro", "gentoo", "alpine", "kali", "mint", "pop!_os", "elementary os", "zarin os", "freebsd", "openbsd", "netbsd", "dragonfly bsd", "solaris", "aix", "hp-ux", "tcp/ip", "udp", "icmp", "arp", "dns", "dhcp", "http", "https", "ftp", "sftp", "ftps", "tftp", "smtp", "pop3", "imap", "snmp", "ssh", "telnet", "rdp", "vnc", "bgp", "ospf", "eigrp", "rip", "is-is", "mpls", "sd-wan", "vlan", "vxlan", "stp", "rstp", "mstp", "lacp", "pagp", "nat", "pat", "acl", "qos",
    # Tools & Others
    "git", "svn", "mercurial", "cvs", "perforce", "tfs", "bitbucket", "gitlab", "github", "jira", "confluence", "trello", "asana", "monday", "clickup", "notion", "slack", "discord", "teams", "zoom", "webex", "skype", "office", "excel", "word", "powerpoint", "outlook", "visio", "project", "access", "sharepoint", "onedrive", "google workspace", "docs", "sheets", "slides", "forms", "drive", "meet", "chat", "photoshop", "illustrator", "indesign", "xd", "figma", "sketch", "invision", "zeplin", "after effects", "premiere", "final cut", "davinci", "blender", "maya", "3ds max", "cinema 4d", "zbrush", "substance painter", "unity", "unreal",
    # Soft Skills
    "communication", "teamwork", "leadership", "problem solving", "critical thinking", "adaptability", "flexibility", "creativity", "innovation", "time management", "organization", "planning", "decision making", "negotiation", "conflict resolution", "emotional intelligence", "empathy", "patience", "resilience", "stress management", "motivation", "initiative", "work ethic", "integrity", "accountability", "responsibility", "reliability", "dependability", "punctuality", "professionalism", "customer service", "client relations", "sales", "marketing", "public speaking", "presentation", "writing", "editing", "research", "analysis", "mentoring", "coaching", "training", "teaching", "supervision", "management", "strategic planning", "project management", "agile", "scrum", "kanban", "lean", "six sigma"
]

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts text page by page.
    """
    text = ""
    # 'with' acts as a context manager. It opens the file and guarantees
    # it closes safely, even if your code crashes halfway through.
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through every page in the PDF
        for page in pdf.pages:
            # Extract the text and add a newline character to separate pages
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(docx_path):
    """
    Opens a DOCX file and extracts text.
    """
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def extract_skills_list(text):
    """
    Scans the text for skills present in the SKILLS_DB.
    Returns a list of unique matched skills.
    """
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        # Use regex boundary to match exact words (e.g., avoid matching "java" in "javascript")
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return list(found_skills)


def extract_details(text):
    """
    Extracts contact info and specific sections using Regex.
    """
    results = {}

    # 1. Extract Email
    # Look for: text + @ + text + . + text
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, text)
    if email_match:
        results['email'] = email_match.group(0)
    else:
        results['email'] = "Not Found"

    # 2. Extract Skills Section
    # This regex looks for the word "SKILLS" followed by a newline, 
    # and captures everything until it hits another major header (like "EDUCATION")
    # Note: This is a simple pattern; real resumes are messy!
    
    # We use re.IGNORECASE so it matches "Skills", "SKILLS", or "skills"
    # The pattern matches: "Skills" -> (capture content) -> Stop at "Education" or "Experience"
    skills_pattern = r'(Skills|Technical Skills|Core Competencies)([\s\S]*?)(Education|Experience|Work History|Projects)'
    
    skills_match = re.search(skills_pattern, text, re.IGNORECASE)
    
    if skills_match:
        # group(2) contains the actual content between the headers
        results['skills_section'] = skills_match.group(2).strip()
    else:
        results['skills_section'] = "Could not isolate skills section."

    # 3. Extract Skills List (NEW)
    # If we found a section, search there. Otherwise, search the whole text.
    search_area = results['skills_section'] if results['skills_section'] != "Could not isolate skills section." else text
    results['skills_list'] = extract_skills_list(search_area)

    return results

def calculate_match_score(candidate_skills, jd_skills):
    """
    Compares candidate skills with JD skills.
    Returns: score (0-100), matched_list, missing_list
    """
    if not jd_skills:
        return 0, [], []
    
    candidate_set = set(candidate_skills)
    jd_set = set(jd_skills)
    
    matches = list(candidate_set.intersection(jd_set))
    missing = list(jd_set - candidate_set)
    
    score = (len(matches) / len(jd_set)) * 100 if len(jd_set) > 0 else 0
    return round(score, 2), matches, missing

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    pdf_file_path = "sample_resume.pdf" 
    
    try:
        print(f"--- Processing {pdf_file_path} ---")
        
        # Step 1: Extract
        raw_data = extract_text_from_pdf(pdf_file_path)
        
        # Step 2: Extract Specific Details (NEW STEP)
        details = extract_details(raw_data)
        
        print("\n--- Extracted Info ---")
        print(f"Candidate Email: {details['email']}")
        print(f"Skills Found:    {details['skills_section'][:100]}...") # Print first 100 chars of skills
        
    except FileNotFoundError:
        print("Error: Could not find the PDF file.")