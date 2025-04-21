import os
import json
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_ml.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(_name_)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Constants
MODEL_PATH = "models/search_model.h5"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
EMBEDDING_SIZE = 100
JOBS_DATA_FILE = "data/jobs.json"

class JobSearchML:
    """Machine Learning model for job search functionality."""
    
    def _init_(self):
        """Initialize the JobSearchML model."""
        self.vectorizer = None
        self.model = None
        self.word_vectors = {}
        self.tech_skills_dict = {
            "programming_languages": [
                "java", "python", "javascript", "typescript", "c++", "c#", "go", 
                "ruby", "php", "swift", "kotlin", "scala", "rust", "perl"
            ],
            "frameworks": [
                "spring", "spring boot", "django", "flask", "react", "angular", 
                "vue", "express", "node.js", "hibernate", "jpa", "asp.net", 
                "laravel", "symfony", ".net core", "tensorflow", "pytorch"
            ],
            "databases": [
                "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", 
                "redis", "cassandra", "dynamodb", "couchbase", "neo4j", "elasticsearch"
            ],
            "cloud_services": [
                "aws", "ec2", "s3", "lambda", "azure", "google cloud", "gcp", 
                "firebase", "heroku", "digitalocean", "cloudflare", "cloudwatch"
            ],
            "devops": [
                "docker", "kubernetes", "jenkins", "gitlab ci", "github actions", 
                "terraform", "ansible", "puppet", "chef", "prometheus", "grafana"
            ],
            "tools": [
                "git", "svn", "maven", "gradle", "npm", "yarn", "webpack", "babel", 
                "jira", "confluence", "slack", "notion", "figma", "sketch"
            ],
            "concepts": [
                "rest", "api", "microservices", "serverless", "ci/cd", "agile", 
                "scrum", "kanban", "tdd", "bdd", "design patterns", "mvc", "orm"
            ]
        }
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load existing model and vectorizer
        self.load_model()
        self.load_vectorizer()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for ML model."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from text."""
        text = text.lower()
        found_skills = []
        
        # Flatten the skills dictionary
        all_skills = []
        for category, skills in self.tech_skills_dict.items():
            all_skills.extend(skills)
        
        # Look for skills in the text
        for skill in all_skills:
            # Use word boundary to match whole words
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                # Capitalize skill names properly
                words = skill.split()
                capitalized = ' '.join(word.capitalize() if word not in ['and', 'or', 'the', 'in', 'on', 'at'] else word for word in words)
                found_skills.append(capitalized)
        
        return found_skills
    
    def create_vectorizer(self, job_texts: List[str]) -> None:
        """Create and fit a TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectorizer.fit(job_texts)
    
    def save_vectorizer(self) -> None:
        """Save the TF-IDF vectorizer to disk."""
        if self.vectorizer:
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info("Vectorizer saved successfully")
    
    def load_vectorizer(self) -> bool:
        """Load the TF-IDF vectorizer from disk."""
        try:
            if os.path.exists(VECTORIZER_PATH):
                with open(VECTORIZER_PATH, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Vectorizer loaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading vectorizer: {str(e)}")
            return False
    
    def create_model(self) -> None:
        """Create a neural network model for job search."""
        # Input for TF-IDF vectors
        tfidf_input = Input(shape=(5000,), name='tfidf_input')
        
        # Dense layers for TF-IDF processing
        tfidf_dense1 = Dense(512, activation='relu')(tfidf_input)
        tfidf_dropout1 = Dropout(0.3)(tfidf_dense1)
        tfidf_dense2 = Dense(256, activation='relu')(tfidf_dropout1)
        tfidf_dropout2 = Dropout(0.3)(tfidf_dense2)
        
        # Final dense layers
        dense1 = Dense(128, activation='relu')(tfidf_dropout2)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(dropout2)
        
        # Create model
        self.model = Model(inputs=tfidf_input, outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model created successfully")
    
    def save_model(self) -> None:
        """Save the model to disk."""
        if self.model:
            self.model.save(MODEL_PATH)
            logger.info("Model saved successfully")
    
    def load_model(self) -> bool:
        """Load the model from disk."""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = load_model(MODEL_PATH)
                logger.info("Model loaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_training_data(self, jobs: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data from job listings."""
        X = []
        y = []
        
        # Process each job
        for job in jobs:
            job_text = f"{job['title']} {job['description']} {job['requirements']}"
            job_text = self.preprocess_text(job_text)
            
            # Create positive examples (job matches itself)
            if self.vectorizer:
                job_vector = self.vectorizer.transform([job_text]).toarray()[0]
                X.append(job_vector)
                y.append(1.0)  # Perfect match
            
            # Create negative examples (job doesn't match random skills)
            for _ in range(3):  # Create 3 negative examples per job
                # Generate random skills text
                random_skills = []
                for category, skills in self.tech_skills_dict.items():
                    # Pick 1-3 random skills from each category
                    num_skills = np.random.randint(1, 4)
                    if len(skills) > 0:
                        random_category_skills = np.random.choice(skills, size=min(num_skills, len(skills)), replace=False)
                        random_skills.extend(random_category_skills)
                
                # Shuffle and take a subset
                np.random.shuffle(random_skills)
                random_skills = random_skills[:np.random.randint(3, 8)]
                
                # Create text from random skills
                random_text = ' '.join(random_skills)
                random_text = self.preprocess_text(random_text)
                
                if self.vectorizer:
                    random_vector = self.vectorizer.transform([random_text]).toarray()[0]
                    X.append(random_vector)
                    y.append(0.0)  # Poor match
        
        return np.array(X), np.array(y)
    
    def train(self, epochs: int = 20, batch_size: int = 32) -> None:
        """Train the model on job data."""
        # Load job data
        try:
            with open(JOBS_DATA_FILE, 'r') as f:
                jobs = json.load(f)
        except Exception as e:
            logger.error(f"Error loading job data: {str(e)}")
            # Create sample job data if file doesn't exist
            jobs = self.create_sample_jobs()
            with open(JOBS_DATA_FILE, 'w') as f:
                json.dump(jobs, f, indent=2)
        
        # Preprocess job texts
        job_texts = []
        for job in jobs:
            job_text = f"{job['title']} {job['description']} {job['requirements']}"
            job_text = self.preprocess_text(job_text)
            job_texts.append(job_text)
        
        # Create vectorizer if it doesn't exist
        if not self.vectorizer:
            self.create_vectorizer(job_texts)
            self.save_vectorizer()
        
        # Create model if it doesn't exist
        if not self.model:
            self.create_model()
        
        # Generate training data
        X, y = self.generate_training_data(jobs)
        
        if len(X) == 0:
            logger.error("No training data generated")
            return
        
        # Train model
        logger.info(f"Training model with {len(X)} samples...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Save model
        self.save_model()
        
        logger.info("Model training completed")
    
    def create_sample_jobs(self) -> List[Dict[str, Any]]:
        """Create sample job data for training."""
        logger.info("Creating sample job data...")
        
        sample_jobs = [
            {
                "id": "job1",
                "title": "Senior Java Developer",
                "company": "TechCorp Solutions",
                "location": "Remote",
                "description": "We are looking for a Senior Java Developer to join our team and help build scalable microservices.",
                "requirements": "Strong coding skills in Core Java 8.0, Expertise in J2EE, Spring Boot, Spring Data JPA, and Hibernate. Experience with REST API development using microservices architecture. Proficient in PostgreSQL and SQL. Familiarity with AWS Cloud services (EC2, S3, CloudWatch). Experience with Terraform for infrastructure as code. Knowledge of CI/CD pipelines and deployment tools.",
                "experience_level": "Senior",
                "job_type": "Full-time",
                "salary_range": "$120,000 - $150,000",
                "posted_date": "2023-04-10"
            },
            {
                "id": "job2",
                "title": "Full Stack JavaScript Developer",
                "company": "WebApp Innovations",
                "location": "New York, NY",
                "description": "Join our dynamic team building modern web applications with JavaScript frameworks.",
                "requirements": "Proficiency in JavaScript, TypeScript, React, and Node.js. Experience with Express.js and RESTful API design. Knowledge of MongoDB and SQL databases. Familiarity with AWS or Azure cloud services. Understanding of CI/CD pipelines and Git version control. Experience with responsive design and CSS frameworks like Tailwind or Bootstrap.",
                "experience_level": "Mid-level",
                "job_type": "Full-time",
                "salary_range": "$90,000 - $120,000",
                "posted_date": "2023-04-15"
            },
            {
                "id": "job3",
                "title": "DevOps Engineer",
                "company": "Cloud Systems Inc.",
                "location": "Remote",
                "description": "Help us build and maintain our cloud infrastructure and deployment pipelines.",
                "requirements": "Experience with AWS services including EC2, S3, RDS, and Lambda. Proficiency with infrastructure as code using Terraform or CloudFormation. Experience with containerization using Docker and Kubernetes. Knowledge of CI/CD tools like Jenkins, GitLab CI, or GitHub Actions. Scripting skills in Python, Bash, or PowerShell. Understanding of monitoring and logging solutions like Prometheus, Grafana, and ELK stack.",
                "experience_level": "Senior",
                "job_type": "Full-time",
                "salary_range": "$130,000 - $160,000",
                "posted_date": "2023-04-12"
            },
            {
                "id": "job4",
                "title": "Data Scientist",
                "company": "DataInsights Analytics",
                "location": "Boston, MA",
                "description": "Join our data science team to develop machine learning models and extract insights from large datasets.",
                "requirements": "Strong programming skills in Python. Experience with data science libraries like NumPy, Pandas, and Scikit-learn. Knowledge of machine learning algorithms and statistical analysis. Experience with deep learning frameworks like TensorFlow or PyTorch. Familiarity with SQL and NoSQL databases. Good communication skills to present findings to non-technical stakeholders.",
                "experience_level": "Mid-level",
                "job_type": "Full-time",
                "salary_range": "$100,000 - $130,000",
                "posted_date": "2023-04-18"
            },
            {
                "id": "job5",
                "title": "Frontend React Developer",
                "company": "UI Experts Ltd.",
                "location": "Remote",
                "description": "Create beautiful and responsive user interfaces for our web applications.",
                "requirements": "Proficiency in React.js and its ecosystem (Redux, React Router, etc.). Strong JavaScript and TypeScript skills. Experience with modern CSS and CSS-in-JS solutions. Knowledge of responsive design principles and accessibility standards. Familiarity with testing frameworks like Jest and React Testing Library. Experience with build tools like Webpack and Babel.",
                "experience_level": "Junior",
                "job_type": "Full-time",
                "salary_range": "$70,000 - $90,000",
                "posted_date": "2023-04-20"
            },
            {
                "id": "job6",
                "title": "Backend Python Developer",
                "company": "ServerSide Solutions",
                "location": "Austin, TX",
                "description": "Develop and maintain our backend services and APIs using Python.",
                "requirements": "Strong Python programming skills. Experience with web frameworks like Django or Flask. Knowledge of RESTful API design principles. Familiarity with SQL databases (PostgreSQL preferred). Understanding of asynchronous programming. Experience with Docker containerization. Knowledge of testing and debugging tools.",
                "experience_level": "Mid-level",
                "job_type": "Full-time",
                "salary_range": "$85,000 - $110,000",
                "posted_date": "2023-04-14"
            },
            {
                "id": "job7",
                "title": "Cloud Solutions Architect",
                "company": "Enterprise Cloud Services",
                "location": "Remote",
                "description": "Design and implement cloud-based solutions for our enterprise clients.",
                "requirements": "Deep knowledge of AWS or Azure cloud services. Experience designing scalable and resilient cloud architectures. Understanding of networking, security, and compliance in cloud environments. Knowledge of infrastructure as code using Terraform or CloudFormation. Experience with containerization and orchestration. Strong communication skills to work with clients and technical teams.",
                "experience_level": "Senior",
                "job_type": "Full-time",
                "salary_range": "$140,000 - $180,000",
                "posted_date": "2023-04-08"
            },
            {
                "id": "job8",
                "title": "Mobile App Developer (iOS)",
                "company": "MobileFirst Apps",
                "location": "San Francisco, CA",
                "description": "Develop native iOS applications for our clients in various industries.",
                "requirements": "Proficiency in Swift and iOS SDK. Experience with UIKit and SwiftUI. Knowledge of iOS app architecture patterns (MVC, MVVM, etc.). Familiarity with RESTful APIs and JSON. Experience with Core Data and local storage solutions. Understanding of App Store submission process and guidelines.",
                "experience_level": "Mid-level",
                "job_type": "Full-time",
                "salary_range": "$100,000 - $130,000",
                "posted_date": "2023-04-16"
            },
            {
                "id": "job9",
                "title": "Database Administrator",
                "company": "DataSafe Solutions",
                "location": "Chicago, IL",
                "description": "Manage and optimize our database systems to ensure high performance and availability.",
                "requirements": "Experience with PostgreSQL and MySQL database administration. Knowledge of database design, optimization, and performance tuning. Experience with backup and recovery procedures. Understanding of high availability and disaster recovery solutions. Familiarity with database security best practices. Knowledge of data migration and ETL processes.",
                "experience_level": "Senior",
                "job_type": "Full-time",
                "salary_range": "$110,000 - $140,000",
                "posted_date": "2023-04-11"
            },
            {
                "id": "job10",
                "title": "QA Automation Engineer",
                "company": "Quality Software Inc.",
                "location": "Remote",
                "description": "Develop and maintain automated test suites for our software products.",
                "requirements": "Experience with test automation frameworks like Selenium, Cypress, or Playwright. Programming skills in Java, Python, or JavaScript. Knowledge of API testing using tools like Postman or RestAssured. Experience with continuous integration tools. Understanding of test methodologies and best practices. Familiarity with agile development processes.",
                "experience_level": "Mid-level",
                "job_type": "Full-time",
                "salary_range": "$80,000 - $110,000",
                "posted_date": "2023-04-19"
            }
        ]
        
        return sample_jobs
    
    def search(self, query: str, jobs: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for jobs matching the query using ML model.
        
        Args:
            query: The search query (skills, job title, etc.)
            jobs: List of job dictionaries
            limit: Maximum number of results to return
            
        Returns:
            List of job dictionaries with match scores
        """
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Check if vectorizer and model exist
        if not self.vectorizer or not self.model:
            logger.warning("Vectorizer or model not found. Using fallback search method.")
            return self.fallback_search(query, jobs, limit)
        
        # Vectorize query
        try:
            query_vector = self.vectorizer.transform([processed_query]).toarray()[0]
        except Exception as e:
            logger.error(f"Error vectorizing query: {str(e)}")
            return self.fallback_search(query, jobs, limit)
        
        # Calculate match scores for each job
        results = []
        for job in jobs:
            # Combine job title, description, and requirements
            job_text = f"{job['title']} {job['description']} {job['requirements']}"
            job_text = self.preprocess_text(job_text)
            
            try:
                # Vectorize job text
                job_vector = self.vectorizer.transform([job_text]).toarray()[0]
                
                # Predict match score using model
                match_score = self.model.predict(np.array([query_vector]), verbose=0)[0][0]
                
                # Add job with match score to results
                job_with_score = job.copy()
                job_with_score['similarity_score'] = float(match_score)
                job_with_score['match_percentage'] = int(match_score * 100)
                
                # Extract skills from job
                job_skills = self.extract_skills(job_text)
                job_with_score['extracted_skills'] = job_skills
                
                results.append(job_with_score)
            except Exception as e:
                logger.error(f"Error processing job {job.get('id', 'unknown')}: {str(e)}")
        
        # Sort by match score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Get top results
        top_results = results[:limit]
        
        return top_results
    
    def fallback_search(self, query: str, jobs: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback search method using TF-IDF and cosine similarity."""
        # Create a temporary vectorizer for this search
        temp_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Prepare job texts
        job_texts = []
        for job in jobs:
            job_text = f"{job['title']} {job['description']} {job['requirements']}"
            job_texts.append(job_text)
        
        # Fit vectorizer and transform texts
        try:
            job_vectors = temp_vectorizer.fit_transform(job_texts)
            query_vector = temp_vectorizer.transform([query])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, job_vectors)[0]
            
            # Add scores to jobs
            results = []
            for i, job in enumerate(jobs):
                job_with_score = job.copy()
                job_with_score['similarity_score'] = float(similarity_scores[i])
                job_with_score['match_percentage'] = int(similarity_scores[i] * 100)
                
                # Extract skills from job
                job_text = f"{job['title']} {job['description']} {job['requirements']}"
                job_skills = self.extract_skills(job_text)
                job_with_score['extracted_skills'] = job_skills
                
                results.append(job_with_score)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Get top results
            top_results = results[:limit]
            
            return top_results
        except Exception as e:
            logger.error(f"Error in fallback search: {str(e)}")
            
            # Return jobs without scoring as last resort
            return jobs[:limit]

# Example usage
if _name_ == "_main_":
    # Initialize the search model
    search_model = JobSearchML()
    
    # Train the model if needed
    if not search_model.load_model() or not search_model.load_vectorizer():
        print("Training new model...")
        search_model.train(epochs=10, batch_size=32)
    
    # Load jobs data
    try:
        with open(JOBS_DATA_FILE, 'r') as f:
            jobs = json.load(f)
    except Exception as e:
        print(f"Error loading job data: {str(e)}")
        jobs = search_model.create_sample_jobs()
        with open(JOBS_DATA_FILE, 'w') as f:
            json.dump(jobs, f, indent=2)
    
    # Test search functionality
    query = "Java developer with Spring Boot and AWS experience"
    results = search_model.search(query, jobs, limit=5)
    
    print(f"Search results for: '{query}'")
    for i, job in enumerate(results):
        print(f"\n{i+1}. {job['title']} at {job['company']}")
        print(f"   Match: {job['match_percentage']}%")
        print(f"   Skills: {', '.join(job['extracted_skills'])}")
