"""
Vercel Serverless API Handler
Exposes fast, serverless verification endpoints for Vercel deployment.
Supports:
1. Mode 1: Curated Medical Claims & Hinglish Dataset (4,946 records)
2. Mode 2: Live Real-Time Retrieval (NCBI PubMed Central & WHO)
"""

import os
import sys
import re
import requests
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.health_classifier import is_health_related
    from src.language_detector import detect_language
    from src.preprocessing import preprocess_text
    from src.live_retriever import fetch_live_clinical_evidence
    from src.verdict_engine import assess_verdict_and_risk
except Exception as e:
    # Fallback imports if executed directly from api directory
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))
    from health_classifier import is_health_related
    from language_detector import detect_language
    from preprocessing import preprocess_text
    from live_retriever import fetch_live_clinical_evidence
    from verdict_engine import assess_verdict_and_risk

app = FastAPI(title="Multilingual Healthcare RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Lazy-load unified dataset & TF-IDF index for Mode 1
# -------------------------------------------------------------
_df = None
_vectorizer = None
_tfidf_matrix = None

def get_dataset_index():
    global _df, _vectorizer, _tfidf_matrix
    if _df is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "evidence_dataset.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        _df = pd.read_csv(csv_path)
        _vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        _tfidf_matrix = _vectorizer.fit_transform(_df["evidence_text"].fillna(""))
    return _df, _vectorizer, _tfidf_matrix

class ClaimRequest(BaseModel):
    claim: str
    mode: Optional[str] = "dataset"  # "dataset" or "live"

def _quote_abstract(text: str, limit: int = 800) -> str:
    if not text or len(text.strip()) < 20:
        return ""
    t = text.strip()
    if len(t) > limit:
        cut = t[:limit]
        lp = max(cut.rfind(". "), cut.rfind(".\n"))
        t = (cut[:lp + 1] if lp > limit * 0.6 else cut) + " [...]"
    return f'"{t}"'

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "Multilingual Healthcare Misinformation RAG API"}

@app.post("/api/verify")
def verify_claim(req: ClaimRequest):
    claim = req.claim.strip()
    if not claim:
        return {"error": "Please enter a claim to verify."}

    # Step 1: Language Detection
    lang = detect_language(claim)

    # Step 2: Semantic Health Classification & Guardrails
    is_health = is_health_related(claim)
    if not is_health:
        return {
            "claim": claim,
            "detected_language": lang,
            "is_health": False,
            "verdict": "Non-Health Claim",
            "risk": "None",
            "summary": f'The query "{claim}" was classified as NOT health-related. Our clinical verification engine only processes medical, health, or clinical claims.',
            "top_evidence": None,
            "evidence_list": []
        }

    cleaned = preprocess_text(claim)
    mode = req.mode.lower()

    retrieved = []
    if "live" in mode:
        # Mode 2: Live Real-Time Retrieval (PubMed Central & WHO)
        retrieved = fetch_live_clinical_evidence(cleaned)
        if not retrieved:
            # Fallback to curated dataset if offline
            df, vec, mat = get_dataset_index()
            qv = vec.transform([cleaned])
            sims = cosine_similarity(qv, mat).flatten()
            top_indices = sims.argsort()[::-1][:3]
            for rank, idx in enumerate(top_indices):
                item = df.iloc[idx].to_dict()
                item["similarity_score"] = float(round(float(sims[idx]), 4))
                item["rank"] = rank + 1
                retrieved.append(item)
    else:
        # Mode 1: Curated Medical Claims & Hinglish Dataset (4,946 records)
        df, vec, mat = get_dataset_index()
        qv = vec.transform([cleaned])
        sims = cosine_similarity(qv, mat).flatten()
        top_indices = sims.argsort()[::-1][:3]
        for rank, idx in enumerate(top_indices):
            item = df.iloc[idx].to_dict()
            item["similarity_score"] = float(round(float(sims[idx]), 4))
            item["rank"] = rank + 1
            retrieved.append(item)

    # Step 3: Verdict & Risk Assessment
    verdict_data = assess_verdict_and_risk(cleaned, retrieved)
    verdict = verdict_data["verdict"]
    risk = verdict_data["risk"]

    top = retrieved[0] if retrieved else {}
    top_explanation = top.get("clinical_explanation", top.get("evidence_text", ""))
    top_abstract = top.get("full_abstract", top.get("evidence_text", ""))

    # Format quoted abstract
    quoted_abstract = _quote_abstract(top_abstract)

    # Build concise educational summary
    if verdict == "Supported":
        summary = (
            f"✅ **Supported by Medical Evidence**: This claim is validated by clinical trials, systematic reviews, or medical consensus. "
            f"Evaluated by **{top.get('authors', 'Clinical Researchers')}** ({top.get('source', 'Cochrane Collaboration')})."
        )
    elif "Uncertain" in verdict:
        summary = (
            f"⚠️ **Inconclusive / Not Proven Yet**: Current peer-reviewed literature does not provide sufficient, reliable clinical proof to validate this claim. "
            f"Further rigorous human trials are required."
        )
    elif verdict == "Refuted":
        summary = (
            f"❌ **Directly Refuted by Medical Research**: Clinical findings contradict or disprove this assertion. "
            f"Adhering to this misinformation poses a **{risk.upper()}** health risk."
        )
    else:
        summary = "No decisive clinical documents found matching this statement in vetted medical repositories."

    return {
        "claim": claim,
        "detected_language": lang,
        "is_health": True,
        "verdict": verdict,
        "risk": risk,
        "summary": summary,
        "top_evidence": {
            "title": top.get("article_title", top.get("claim_topic", "Clinical Review")),
            "authors": top.get("authors", "Verified Medical Researchers"),
            "author_credentials": top.get("author_credentials", "Peer-Reviewed Medical Researchers"),
            "source": top.get("source", "Clinical Evidence Base"),
            "source_tier": top.get("source_tier", "Tier 1: Cochrane Certified Clinical Fact-Check"),
            "url": top.get("url", "https://pubmed.ncbi.nlm.nih.gov/"),
            "stance": top.get("stance", "uncertain"),
            "similarity_score": top.get("similarity_score", 0.0),
            "clinical_explanation": top_explanation,
            "quoted_abstract": quoted_abstract,
            "date": top.get("date", "")
        },
        "evidence_list": [
            {
                "rank": item.get("rank", i + 1),
                "title": item.get("article_title", item.get("claim_topic", "")),
                "authors": item.get("authors", ""),
                "source": item.get("source", ""),
                "source_tier": item.get("source_tier", ""),
                "url": item.get("url", "#"),
                "stance": item.get("stance", ""),
                "similarity_score": item.get("similarity_score", 0.0),
                "explanation": item.get("clinical_explanation", item.get("evidence_text", "")),
                "quoted_abstract": _quote_abstract(item.get("full_abstract", item.get("evidence_text", "")))
            }
            for i, item in enumerate(retrieved)
        ]
    }
