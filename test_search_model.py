import requests
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from search_ml_model import JobSearchML

def test_search_model():
    """Test the ML search model with various queries."""
    print("=== Testing ML Job Search Model ===")

    # Initialize the search model
    search_model = JobSearchML()

    # Load the model
    if not search_model.load_model() or not search_model.load_vectorizer():
        print("Training new model...")
        search_model.train(epochs=10, batch_size=32)

    print("\nModel loaded successfully!")

    # Load jobs data
    JOBS_DATA_FILE = "data/jobs.json"
    try:
        with open(JOBS_DATA_FILE, 'r') as f:
            jobs = json.load(f)
    except Exception as e:
        print(f"Error loading job data: {str(e)}")
        jobs = search_model.create_sample_jobs()
        with open(JOBS_DATA_FILE, 'w') as f:
            json.dump(jobs, f, indent=2)

    # Test search queries
    test_queries = [
        {
            "name": "Java Developer",
            "query": "Java developer with Spring Boot and AWS experience"
        },
        {
            "name": "Frontend Developer",
            "query": "React developer with TypeScript and responsive design experience"
        },
        {
            "name": "DevOps Engineer",
            "query": "DevOps engineer with Kubernetes, Docker, and CI/CD experience"
        },
        {
            "name": "Data Scientist",
            "query": "Data scientist with Python, machine learning, and statistical analysis skills"
        },
        {
            "name": "Generic IT",
            "query": "IT professional with technical skills"
        }
    ]

    # Test each query
    results = []

    for query_info in test_queries:
        print(f"\n=== Testing Query: {query_info['name']} ===")
        print(f"Query: {query_info['query']}")
        
        # Measure search time
        start_time = time.time()
        search_results = search_model.search(query_info['query'], jobs, limit=5)
        search_time = time.time() - start_time
        
        print(f"Search completed in {search_time:.4f} seconds")
        print(f"Found {len(search_results)} matching jobs")
        
        # Print top results
        for i, job in enumerate(search_results[:3]):
            print(f"\n{i+1}. {job['title']} at {job['company']}")
            print(f"   Match: {job['match_percentage']}%")
            print(f"   Skills: {', '.join(job['extracted_skills'])}")
        
        # Store results for visualization
        for job in search_results:
            results.append({
                "query": query_info['name'],
                "job_title": job["title"],
                "match_score": job["similarity_score"]
            })

    # Visualize results
    plt.figure(figsize=(12, 8))

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(results)

    # Plot match scores by query and job
    sns.barplot(x="job_title", y="match_score", hue="query", data=df)
    plt.title("Match Scores by Query and Job")
    plt.xlabel("Job Title")
    plt.ylabel("Match Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("models/search_test_results.png")

    print("\nTest results visualization saved to models/search_test_results.png")

    # Test skill extraction
    print("\n=== Testing Skill Extraction ===")
    test_texts = [
        "Java developer with Spring Boot, Hibernate, and AWS experience",
        "Frontend developer proficient in React, TypeScript, and responsive design",
        "DevOps engineer experienced with Docker, Kubernetes, and CI/CD pipelines",
        "Data scientist with Python, TensorFlow, and statistical analysis skills"
    ]

    for text in test_texts:
        skills = search_model.extract_skills(text)
        print(f"\nText: {text}")
        print(f"Extracted skills: {', '.join(skills)}")

    # Test API if it's running
    try:
        print("\n=== Testing Search API ===")
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("API is running!")
            
            # Test search endpoint
            search_response = requests.post(
                "http://localhost:8000/api/search/",
                json={
                    "query": "Java developer with Spring Boot and AWS experience",
                    "limit": 3
                }
            )
            
            if search_response.status_code == 200:
                print("Search API endpoint is working!")
                result = search_response.json()
                print(f"Found {len(result['jobs'])} matching jobs")
                print(f"Query time: {result['query_time_ms']}ms")
                
                if result['jobs']:
                    first_job = result['jobs'][0]
                    print(f"\nTop match: {first_job['title']} at {first_job['company']}")
                    print(f"Match percentage: {first_job['match_percentage']}%")
                    print(f"Skills: {', '.join(first_job['extracted_skills'])}")
            else:
                print(f"Search API endpoint failed: {search_response.status_code}")
        else:
            print("API is not running. Start it with 'python search_api.py'")
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        print("API is not running. Start it with 'python search_api.py'")

if _name_ == "_main_":
    test_search_model()
