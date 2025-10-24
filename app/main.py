from fastapi import FastAPI,Request
import requests
from elasticsearch import Elasticsearch
from datetime import datetime
import math
import time
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import http.client
import os
from dotenv import load_dotenv

app = FastAPI(title="Job Sync API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ou ["*"] pour tout autoriser (moins sécurisé)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Charger le fichier .env
load_dotenv()








es = Elasticsearch(
    os.getenv("ELASTIC_URL"),
    api_key=os.getenv("ELASTIC_API_KEY_ID")
)
model = SentenceTransformer("all-MiniLM-L6-v2")


# Configuration Adzuna
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
API_KEY = os.getenv("API_KEY")

RESULTS_PER_PAGE = 1000

# Elasticsearch

genai.configure(api_key=API_KEY)

model_gemini = genai.GenerativeModel("gemini-2.5-flash")


def fetch_all_jobs(delay=1):
    """Récupère toutes les offres d’Adzuna avec pagination sécurisée"""
    all_jobs = []
    url_template = "https://api.adzuna.com/v1/api/jobs/gb/search/{page}"
    total_pages = 200  # tu peux augmenter si besoin

    for page in range(1, total_pages + 1):
        url = url_template.format(page=page)
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "results_per_page": RESULTS_PER_PAGE,
            "content-type": "application/json"
        }

        try:
            print(f"➡️  Récupération de la page {page}/{total_pages}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # lève une exception si erreur HTTP
            data = response.json()

            jobs = data.get("results", [])
            if not jobs:
                print(f"⚠️  Aucune offre trouvée à la page {page}, arrêt anticipé.")
                break

            all_jobs.extend(jobs)
            print(f"✅ {len(jobs)} offres récupérées (total: {len(all_jobs)})")

            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur lors de la récupération de la page {page}: {e}")
            break

    print(f"🎯 Total d'offres collectées : {len(all_jobs)}")
    return all_jobs

'''
def fetch_all_jobs2(query="developer jobs in chicago", country="us", max_pages=5, delay=1):
    """Récupère toutes les offres via JSearch API (RapidAPI) avec pagination"""
    all_jobs = []

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
    'x-rapidapi-key': "38bd1a1df8mshe1547c0eb91c83fp19b06bjsn106f3b1c2c53",
    'x-rapidapi-host': "jsearch.p.rapidapi.com"
}


    for page in range(1, max_pages + 1):
        params = {
            "query": query,
            "page": page,
            "num_pages": 1,          # 1 page à la fois
            "country": country,
            "date_posted": "all"
        }

        try:
            print(f"➡️  Récupération de la page {page}/{max_pages}...")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            jobs = data.get("data", [])
            if not jobs:
                print(f"⚠️  Aucune offre trouvée à la page {page}, arrêt anticipé.")
                break

            all_jobs.extend(jobs)
            print(f"✅ {len(jobs)} offres récupérées (total: {len(all_jobs)})")

            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur lors de la récupération de la page {page}: {e}")
            break

    print(f"🎯 Total d'offres collectées : {len(all_jobs)}")
    return all_jobs
'''

def index_jobs(jobs):
    """Indexe les offres dans Elasticsearch"""
    for job in jobs:
        text_to_embed = f"{job.get('title', '')}. {job.get('description', '')}. {job.get('company', '')}. {job.get('location', '')}. salary min:{job.get('salary_min','')}. salary_max:{job.get('salary_max','')}. post date: ${job.get('created','')}"
        embedding = model.encode(text_to_embed)

        doc = {
            "id": job.get("id"),
            "title": job.get("title"),
            "company": job.get("company", {}).get("display_name"),
            "location": job.get("location", {}).get("display_name"),
            "description": job.get("description"),
            "source": "Adzuna",
            "url": job.get("redirect_url"),
            "posted_at": datetime.strptime(job["created"], "%Y-%m-%dT%H:%M:%SZ"),
            "salary_min": job.get('salary_min',''),
            "salary_max": job.get('salary_max',''),
            "contract_type": job.get('contract_type',''),
            "contract_time": job.get('contract_time',''),
            "embedding":embedding
        }
        es.index(index="jobs2", document=doc)

'''
def index_jobs2(jobs):
    """Indexe les offres dans Elasticsearch"""
    for job in jobs:
        text_to_embed = f"{job.get('job_title', '')}. {job.get('job_description', '')}. {job.get('company_object', {}).get('name', '')}. {job.get('location', [{}])[0].get('display_name', '')}. salary min:{job.get('job_min_salary','')}. salary_max:{job.get('job_max_salary','')}. post date: ${job.get('job_posted_at_datetime_utc','')}"
        embedding = model.encode(text_to_embed)

        doc = {
            "id": job.get("job_id"),
            "title": job.get("job_title"),
            "company": job.get("company_object", {}).get("name", ""),
            "location": job.get("locations", [{}])[0].get("display_name", ""),
            "description": job.get("job_description"),
            "source": "JSearch",
            "url": job.get("job_apply_link"),
            "posted_at": datetime.strptime(job['job_posted_at_datetime_utc'], "%Y-%m-%dT%H:%M:%S.%fZ") 
             if job.get('job_posted_at_datetime_utc') else None,
            "salary_min": job.get('job_min_salary',''),
            "salary_max": job.get('job_max_salary',''),
            "contract_type": job.get('job_employment_type',''),
            "remote": job.get('job_is_remote',''),
            "embedding":embedding
        }
        es.index(index="jobs3", document=doc)

'''        

def semantic_search(query, top_k=5):

    #prompt = f"Écris une phrase sur le coucher du soleil. '${query}'"
    #query = model.generate_content(prompt)

    query_vector =  model.encode(query)


    response = es.search(
        index="jobs2",
        body={
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }
    )
    results = [
        {
            "id": hit["_source"]["id"],
            "title": hit["_source"]["title"],
            "company": hit["_source"].get("company"),
            "location":hit["_source"].get("location"),
            "description":hit["_source"].get("description"),
            "score": hit["_score"],
            "url": hit["_source"]["url"]
        }
        for hit in response["hits"]["hits"]
    ]

    return results


@app.on_event("startup")
def create_index():
    """Créer l'index Elasticsearch avec le bon mapping si inexistant"""
    index_name = "jobs2"
    if not es.indices.exists(index=index_name):
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "company": {"type": "keyword"},
                        "location": {"type": "keyword"},
                        "description": {"type": "text"},
                        "source": {"type": "keyword"},
                        "url": {"type": "keyword"},
                        "salary_min": {"type": "keyword"},
                        "salary_max": {"type": "keyword"},
                        "remote": {"type": "keyword"},
                        "contract_time": {"type": "keyword"},
                        "posted_at": {"type": "date"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 384,  # modèle MiniLM-L6-v2
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
        )
        print("✅ Index 'jobs' créé avec mapping dense_vector.")
    else:
        print("ℹ️ Index 'jobs' déjà existant.")

@app.get("/sync_all")
def sync_all():
    """Récupère toutes les offres Adzuna et les envoie dans Elasticsearch"""
    print("hi")
    jobs = fetch_all_jobs()
    index_jobs(jobs)
    return {"status": "ok", "indexed": len(jobs)}

'''
@app.get("/sync_all2")
def sync_all2():
    """Récupère toutes les offres Adzuna et les envoie dans Elasticsearch"""
    print("hi")
    jobs = fetch_all_jobs2()
    index_jobs2(jobs)
    return {"status": "ok", "indexed": len(jobs)}

'''

@app.post("/search")
async def search_jobs(request: Request, top_k: int = 5):
    data = await request.json()            # ✅ lire le corps JSON
    prompt = data.get("prompt")             # ✅ extraire le champ "prompt"
    
    if not prompt:
        return {"error": "Le champ 'prompt' est requis."}
    
    results = semantic_search(prompt, top_k)
    return {"query": prompt, "results": results}


@app.get("/job/{job_id}")
def get_job_by_id(job_id: str):
    """Retourne une offre à partir de son ID Elasticsearch"""
    try:
        response = es.search(
            index="jobs2",
            body={
                "query": {
                    "term": {
                        "id": job_id
                    }
                }
            }
        )

        hits = response["hits"]["hits"]
        if not hits:
            return {"error": f"Aucun job trouvé avec l'id {job_id}"}

        job = hits[0]["_source"]
        return job

    except Exception as e:
        return {"error": f"Erreur lors de la recherche du job {job_id}: {str(e)}"}


@app.get("/")
def root():
    return {"message": "🚀 FastAPI backend is running successfully on Render!"}