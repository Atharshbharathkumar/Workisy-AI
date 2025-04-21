from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os
import logging
from datetime import datetime
import time

from search_ml_model import JobSearchML

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(_name_)

# Constants
JOBS_DATA_FILE = "data/jobs.json"

# Initialize FastAPI with metadata for Swagger
app = FastAPI(
    title="ML-Enhanced Job Search API",
    description="""
    This API provides machine learning enhanced job search capabilities.
    
    It uses a neural network model trained on TF-IDF vectors to match job seekers' 
    search queries with job listings, providing more accurate and semantic matching 
    than traditional keyword-based approaches.
    """,
    version="1.0.0",
    docs_url=None,  # We'll customize the docs URL
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML model
search_model = JobSearchML()

# Load the model on startup
@app.on_event("startup")
async def startup_event():
    # Load the trained model and vectorizer
    model_loaded = search_model.load_model()
    vectorizer_loaded = search_model.load_vectorizer()
    
    if not model_loaded or not vectorizer_loaded:
        logger.warning("Model or vectorizer not found. Training new model...")
        search_model.train(epochs=10, batch_size=32)
    
    logger.info("ML model loaded and ready")

# Pydantic models for request and response
class SearchRequest(BaseModel):
    query: str = Field(
        ..., 
        description="Search query (skills, job title, etc.)",
        example="Java developer with Spring Boot and AWS experience"
    )
    experience_level: Optional[str] = Field(
        None, 
        description="Filter by experience level (e.g., Junior, Mid-level, Senior, Lead)",
        example="Senior"
    )
    location: Optional[str] = Field(
        None, 
        description="Filter by location (e.g., Remote, New York)",
        example="Remote"
    )
    job_type: Optional[str] = Field(
        None, 
        description="Filter by job type (e.g., Full-time, Part-time, Contract)",
        example="Full-time"
    )
    limit: int = Field(
        10, 
        description="Maximum number of results to return",
        ge=1, 
        le=100,
        example=10
    )

class JobResult(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    requirements: str
    experience_level: str
    job_type: str
    salary_range: Optional[str] = None
    posted_date: Optional[str] = None
    similarity_score: float
    match_percentage: int
    extracted_skills: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "id": "job1",
                "title": "Senior Java Developer",
                "company": "TechCorp Solutions",
                "location": "Remote",
                "description": "We are looking for a Senior Java Developer to join our team and help build scalable microservices.",
                "requirements": "Strong coding skills in Core Java 8.0, Expertise in J2EE, Spring Boot, Spring Data JPA, and Hibernate...",
                "experience_level": "Senior",
                "job_type": "Full-time",
                "salary_range": "$120,000 - $150,000",
                "posted_date": "2023-04-10",
                "similarity_score": 0.92,
                "match_percentage": 92,
                "extracted_skills": ["Java", "Spring Boot", "Hibernate", "JPA", "REST API", "Microservices", "PostgreSQL", "AWS"]
            }
        }

class SearchResponse(BaseModel):
    jobs: List[JobResult]
    total_matches: int
    query_time_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "jobs": [
                    {
                        "id": "job1",
                        "title": "Senior  [
                    {
                        "id": "job1",
                        "title": "Senior Java Developer",
                        "company": "TechCorp Solutions",
                        "location": "Remote",
                        "description": "We are looking for a Senior Java Developer to join our team and help build scalable microservices.",
                        "requirements": "Strong coding skills in Core Java 8.0, Expertise in J2EE, Spring Boot, Spring Data JPA, and Hibernate...",
                        "experience_level": "Senior",
                        "job_type": "Full-time",
                        "salary_range": "$120,000 - $150,000",
                        "posted_date": "2023-04-10",
                        "similarity_score": 0.92,
                        "match_percentage": 92,
                        "extracted_skills": ["Java", "Spring Boot", "Hibernate", "JPA", "REST API", "Microservices", "PostgreSQL", "AWS"]
                    }
                ],
                "total_matches": 1,
                "query_time_ms": 125.45
            }
        }

class APIStatus(BaseModel):
    status: str
    version: str
    model_loaded: bool
    uptime: str
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "model_loaded": True,
                "uptime": "1 day, 2 hours, 35 minutes"
            }
        }

# Load jobs data
def load_jobs_data():
    try:
        with open(JOBS_DATA_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading jobs data: {str(e)}")
        return search_model.create_sample_jobs()

# Start time for uptime calculation
start_time = datetime.now()

# Custom Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )

# Custom OpenAPI schema
@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

@app.post("/api/search/", response_model=SearchResponse, tags=["Search"])
async def search_jobs(request: SearchRequest):
    """
    Search for jobs using ML-based matching.
    
    This endpoint uses a trained neural network model to find jobs
    that match the provided search query, with better semantic understanding
    than traditional keyword-based search.
    
    The model considers:
    - Semantic relationships between terms
    - Context in which terms are used
    - Technical skills mentioned in the query
    
    Returns a list of matching jobs with detailed match information.
    """
    logger.info(f"Search request: {request}")
    start_time_ms = time.time() * 1000
    
    # Load jobs data
    jobs = load_jobs_data()
    if not jobs:
        return SearchResponse(
            jobs=[],
            total_matches=0,
            query_time_ms=0
        )
    
    # Apply filters before search
    filtered_jobs = jobs
    if request.experience_level:
        filtered_jobs = [job for job in filtered_jobs 
                        if job.get('experience_level') and 
                        request.experience_level.lower() in job['experience_level'].lower()]
    
    if request.location:
        filtered_jobs = [job for job in filtered_jobs 
                        if job.get('location') and 
                        request.location.lower() in job['location'].lower()]
    
    if request.job_type:
        filtered_jobs = [job for job in filtered_jobs 
                        if job.get('job_type') and 
                        request.job_type.lower() in job['job_type'].lower()]
    
    # Search using ML model
    results = search_model.search(request.query, filtered_jobs, request.limit)
    
    # Calculate query time
    query_time = time.time() * 1000 - start_time_ms
    
    logger.info(f"Found {len(results)} matching jobs in {query_time:.2f}ms")
    return SearchResponse(
        jobs=results,
        total_matches=len(results),
        query_time_ms=round(query_time, 2)
    )

@app.get("/api/extract-skills/", tags=["Skills"])
async def extract_skills(text: str = Query(..., description="Text to extract skills from")):
    """
    Extract technical skills from text.
    
    This endpoint analyzes the provided text and identifies technical skills.
    
    Returns a list of extracted skills.
    """
    # Extract skills from text
    skills = search_model.extract_skills(text)
    
    return {"skills": skills}

@app.get("/", response_model=APIStatus, tags=["Status"])
async def root():
    """
    Get API status.
    
    Returns information about the API status, version, and model.
    """
    # Calculate uptime
    uptime = datetime.now() - start_time
    days, remainder = divmod(uptime.total_seconds(), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    uptime_str = f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes"
    if days < 1:
        uptime_str = f"{int(hours)} hours, {int(minutes)} minutes"
    if hours < 1:
        uptime_str = f"{int(minutes)} minutes, {int(seconds)} seconds"
    
    return APIStatus(
        status="ok",
        version="1.0.0",
        model_loaded=search_model.model is not None,
        uptime=uptime_str
    )

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
