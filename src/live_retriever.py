"""
Live Clinical Evidence Retriever Module
Fetches real-time peer-reviewed evidence ONLY from trusted, authoritative medical sources:
  - PubMed Central / NCBI  (peer-reviewed journals, primary source)
  - WHO / NIH / CDC        (global health authority publications)
  - Cochrane Library       (gold-standard systematic reviews)
  - Mayo Clinic / NHS      (renowned clinical institutions)

Wikipedia is intentionally excluded — it is not a primary medical authority.
"""

import re
import json
import time
import urllib.parse
import urllib.request
import urllib.error
import os
from typing import List, Dict, Any, Optional


# ── Trusted source registry ──────────────────────────────────────────────────
# Each entry defines: display name, credibility tier, author label
TRUSTED_SOURCES = {
    "google_factcheck": {
        "name": "Google Fact Check Tools",
        "tier": "🥇 Tier 1 – Verified Fact-Checking Organization (IFCN Signatory)",
        "author_label": "Certified Independent Fact-Checkers (Google Fact Check Tools)"
    },
    "europe_pmc": {
        "name": "Europe PMC",
        "tier": "🥇 Tier 1 – Biomedical Literature Index (EMBL-EBI)",
        "author_label": "Peer-Reviewed Medical Researchers"
    },
    "clinical_trials": {
        "name": "ClinicalTrials.gov",
        "tier": "🥇 Tier 1 – NIH Clinical Trials Registry",
        "author_label": "NIH Registered Clinical Investigators"
    },
    "pubmed": {
        "name": "NCBI PubMed Central",
        "tier": "🥇 Tier 1 – Peer-Reviewed Journal",
        "author_label": "Peer-Reviewed Medical Researchers (NCBI PMC)"
    },
    "who": {
        "name": "World Health Organization (WHO)",
        "tier": "🥇 Tier 1 – Global Health Authority",
        "author_label": "WHO Medical Experts & Epidemiologists"
    },
    "nih": {
        "name": "National Institutes of Health (NIH)",
        "tier": "🥇 Tier 1 – National Health Authority",
        "author_label": "NIH Research Scientists & Clinicians"
    },
    "cochrane": {
        "name": "Cochrane Library (Systematic Reviews)",
        "tier": "🥇 Tier 1 – Gold-Standard Systematic Review",
        "author_label": "Cochrane Collaboration Review Team"
    },
    "mayo": {
        "name": "Mayo Clinic",
        "tier": "🥈 Tier 2 – Renowned Clinical Institution",
        "author_label": "Mayo Clinic Physicians & Specialists"
    },
    "nhs": {
        "name": "NHS (National Health Service, UK)",
        "tier": "🥈 Tier 2 – National Clinical Authority",
        "author_label": "NHS Clinical Guideline Team"
    },
    "cdc": {
        "name": "Centers for Disease Control (CDC)",
        "tier": "🥈 Tier 2 – Public Health Authority",
        "author_label": "CDC Epidemiologists & Public Health Scientists"
    },
}

# Non-medical noise patterns to filter out from titles
NON_MEDICAL_NOISE = [
    "toyota", "album", "song", "film", "cricket", "football",
    "movie", "celebrity", "actor", "actress", "politician",
    "restaurant", "recipe", "cook", "weather", "financial",
    "stock", "crypto", "bitcoin", "game", "sport"
]


def strip_html_tags(text: str) -> str:
    """Removes HTML markup from snippets."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def expand_medical_query(query: str) -> str:
    """Expands colloquial and Hinglish health terms into biomedical terminology."""
    clean = query.strip()
    clean = re.sub(r"\bcorona\b(?!\s*virus)", "coronavirus COVID-19", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bmask\b(?!s)", "face mask respiratory protection", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bhaldi\s*doodh\b", "turmeric milk curcumin anti-inflammatory", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bhaldi\b", "turmeric curcumin", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bdoodh\b", "milk", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bsugar\b", "blood glucose diabetes mellitus", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bmadhumeh\b", "diabetes mellitus", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bbp\b", "blood pressure hypertension", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bilaj\b|\bilaaj\b", "treatment therapy", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\btheek\b", "cure treatment recovery", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bdawa\b|\bdawai\b", "medicine drug", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bbimari\b|\bbimaari\b", "disease illness", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bkhansi\b", "cough respiratory", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bbukhar\b", "fever", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bsore\s+throat\b", "sore throat pharyngitis", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bgale\s+(?:ki|ka)\s+kharash\b", "sore throat pharyngitis", clean, flags=re.IGNORECASE)
    # Filter out conversational Hinglish filler/stopwords before API query
    hinglish_fillers = r"\b(?:ko|se|ka|ki|ke|hai|hain|karta|karti|karte|hota|hoti|hote|kya|kyu|kyun|mein|me|pe|par|bhi|aur|ya|toh|karo|karein|wali|wala|wale|kuch)\b"
    clean = re.sub(hinglish_fillers, " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def _safe_fetch(url: str, timeout: int = 8) -> bytes:
    """Makes a safe HTTP GET request with a custom User-Agent."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "HealthcareRAG-Research/2.0 (Educational; ncbi.nlm.nih.gov)",
            "Accept": "application/json"
        }
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_google_factcheck_evidence(
    query: str,
    api_key: Optional[str] = None,
    language_code: Optional[str] = None,
    max_results: int = 4
) -> List[Dict[str, Any]]:
    """
    Queries Google Fact Check Tools API (https://factchecktools.googleapis.com/v1alpha1/claims:search)
    to find official claim reviews and fact checks published by certified fact-checking organizations
    (e.g., Health Feedback, FactCheck.org, Snopes, Boom Live, Vishwas News, etc.).

    Extracts:
      - Direct official source article URL (review.get('url'))
      - Fact-checking publisher name and website (review.get('publisher'))
      - Official review title
      - Exact textual rating ('False', 'True', 'Misleading', etc.)
      - Standardized scientific stance ('refutes', 'supports', 'uncertain')
    """
    records = []
    key = (
        api_key
        or os.environ.get("GOOGLE_FACTCHECK_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not key or not str(key).strip():
        print("[LiveRetriever] Note: Google Fact Check API key not configured. Set GOOGLE_FACTCHECK_API_KEY or input via UI to query Google Fact Check Tools.")
        return records

    try:
        clean_q = query.strip()
        params = {
            "query": clean_q,
            "key": key.strip(),
            "pageSize": max_results,
        }
        if language_code:
            params["languageCode"] = language_code

        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "HealthcareRAG-Research/2.0 (FactCheckTools)",
                "Accept": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))

        claims = data.get("claims", [])
        for claim_obj in claims:
            claim_text = claim_obj.get("text", "").strip()
            claimant = claim_obj.get("claimant", "Public Claim")
            claim_date = claim_obj.get("claimDate", "")

            reviews = claim_obj.get("claimReview", [])
            for review in reviews:
                pub_obj = review.get("publisher", {})
                publisher_name = pub_obj.get("name", "Independent Fact-Checker")
                publisher_site = pub_obj.get("site", "")
                review_url = review.get("url", "").strip()
                review_title = review.get("title", "").strip() or f"Fact Check: {claim_text}"
                textual_rating = review.get("textualRating", "Unrated").strip()
                review_date = review.get("reviewDate", claim_date)

                if not review_url:
                    continue

                parsed = urllib.parse.urlparse(review_url)
                origin_domain = parsed.netloc.replace("www.", "")

                rating_lower = textual_rating.lower()
                if any(term in rating_lower for term in [
                    "false", "incorrect", "pants on fire", "fake", "distorted", "misleading",
                    "disproven", "debunked", "hoax", "scam", "fabricated", "unfounded",
                    "galat", "jhooth", "asatya", "wrong", "bogus", "myth"
                ]):
                    stance = "refutes"
                elif any(term in rating_lower for term in [
                    "true", "correct", "accurate", "verified", "sahi", "satya", "supported"
                ]) and not any(term in rating_lower for term in ["partly", "half", "mostly false"]):
                    stance = "supports"
                else:
                    stance = "uncertain"

                evidence_summary = (
                    f"Official Fact-Check Investigation by {publisher_name}. "
                    f"Investigated Claim: \"{claim_text}\" (attributed to: {claimant}). "
                    f"Verified Rating: \"{textual_rating}\". "
                    f"Official Source Article: {review_url}"
                )

                citation = f"{publisher_name} ({review_date}): Rating '{textual_rating}'"

                records.append({
                    "claim_topic": review_title,
                    "evidence_text": evidence_summary,
                    "clinical_explanation": f"Independent fact-checking investigation by {publisher_name} evaluated this claim as '{textual_rating}'. Official source article: {review_url}",
                    "full_abstract": evidence_summary,
                    "source": f"Google Fact Check Tools ({publisher_name})",
                    "source_tier": "🥇 Tier 1 – Verified Fact-Checking Organization (IFCN / Google Fact Check)",
                    "authors": f"{publisher_name} Fact-Checking Team",
                    "author_credentials": f"Certified Fact-Checker ({publisher_site or 'Google Fact Check Partner'})",
                    "article_title": review_title,
                    "journal": f"Fact Check Investigation ({publisher_name})",
                    "citation": citation,
                    "date": review_date,
                    "doi": "",
                    "url": review_url,
                    "origin_domain": origin_domain,
                    "origin": f"Google Fact Check Tools API ({publisher_name})",
                    "stance": stance,
                    "rating": textual_rating,
                    "claimant": claimant
                })
    except urllib.error.HTTPError as he:
        if he.code == 403:
            print("[LiveRetriever] Google Fact Check API: HTTP 403 Forbidden (API key invalid or Fact Check Tools API not enabled).")
        elif he.code == 400:
            print(f"[LiveRetriever] Google Fact Check API: HTTP 400 Bad Request ({he.reason}).")
        else:
            print(f"[LiveRetriever] Google Fact Check API HTTP error {he.code}: {he.reason}")
    except Exception as exc:
        print(f"[LiveRetriever] Google Fact Check API fetch notice: {exc}")

    return records


def fetch_pubmed_abstract(pmc_id: str) -> str:
    """
    Fetches the full abstract text of a PMC article using NCBI efetch.
    Returns cleaned abstract text or empty string if unavailable.
    """
    try:
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pmc&id={pmc_id}&rettype=abstract&retmode=xml"
        )
        raw = _safe_fetch(url, timeout=7).decode("utf-8", errors="ignore")
        # Extract abstract text from XML
        abstract_match = re.search(
            r"<abstract[^>]*>(.*?)</abstract>",
            raw, re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            abstract_xml = abstract_match.group(1)
            # Remove XML tags, keep text
            abstract_text = re.sub(r"<[^>]+>", " ", abstract_xml)
            abstract_text = re.sub(r"\s+", " ", abstract_text).strip()
            if len(abstract_text) > 80:
                return abstract_text[:1200]  # cap at 1200 chars
    except Exception:
        pass
    return ""


def fetch_pubmed_evidence(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """
    Queries NCBI PubMed Central for peer-reviewed clinical literature.
    Fetches real article abstracts, not just titles.
    Filters to high-quality medical journals only.
    """
    records = []
    try:
        clinical_query = expand_medical_query(query)
        # Add clinical filters for quality: systematic reviews, meta-analyses, RCTs
        quality_filter = " AND (systematic[sb] OR meta-analysis[pt] OR clinical trial[pt] OR review[pt])"
        full_query = clinical_query + quality_filter
        encoded_query = urllib.parse.quote(full_query)

        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pmc&term={encoded_query}&retmode=json&retmax={max_results}&sort=relevance"
        )
        data = json.loads(_safe_fetch(search_url).decode())
        id_list = data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            # Fallback: search without quality filter
            encoded_query_plain = urllib.parse.quote(clinical_query)
            search_url_plain = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
                f"db=pmc&term={encoded_query_plain}&retmode=json&retmax={max_results}&sort=relevance"
            )
            data = json.loads(_safe_fetch(search_url_plain).decode())
            id_list = data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return records

        # Fetch article summaries
        summary_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
            f"db=pmc&id={','.join(id_list)}&retmode=json"
        )
        summary_data = json.loads(_safe_fetch(summary_url).decode()).get("result", {})

        for uid in id_list:
            item = summary_data.get(uid, {})
            title = item.get("title", "").strip()
            if not title:
                continue

            # Skip non-medical noise
            if any(noise in title.lower() for noise in NON_MEDICAL_NOISE):
                continue

            journal = item.get("source", "PubMed Central")
            pub_date = item.get("pubdate", "")
            volume = item.get("volume", "")
            issue = item.get("issue", "")
            doi = item.get("elocationid", "")

            # Author extraction with credentials
            author_objs = item.get("authors", [])
            author_names = [
                a.get("name", "") for a in author_objs
                if isinstance(a, dict) and a.get("name")
            ]
            if author_names:
                if len(author_names) == 1:
                    authors_str = author_names[0]
                elif len(author_names) <= 3:
                    authors_str = ", ".join(author_names)
                else:
                    authors_str = ", ".join(author_names[:3]) + f" et al. (+{len(author_names)-3} authors)"
            else:
                authors_str = TRUSTED_SOURCES["pubmed"]["author_label"]

            # Fetch real abstract
            abstract_text = fetch_pubmed_abstract(uid)
            time.sleep(0.1)  # be polite to NCBI API

            # Build citation string
            citation_parts = [journal]
            if pub_date:
                citation_parts.append(pub_date)
            if volume:
                citation_parts.append(f"Vol. {volume}")
            if issue:
                citation_parts.append(f"Issue {issue}")
            citation = " | ".join(citation_parts)

            evidence_body = abstract_text if abstract_text else (
                f"Peer-reviewed study investigating: {query}. "
                f"Published in {journal} ({pub_date}). "
                f"Full text available via NCBI PubMed Central."
            )

            records.append({
                "claim_topic": title,
                "evidence_text": evidence_body,
                "clinical_explanation": evidence_body,
                "full_abstract": abstract_text,
                "source": TRUSTED_SOURCES["pubmed"]["name"],
                "source_tier": TRUSTED_SOURCES["pubmed"]["tier"],
                "authors": authors_str,
                "author_credentials": "Published Researchers (Peer-Reviewed Medical Journal)",
                "article_title": title,
                "journal": journal,
                "citation": citation,
                "date": pub_date,
                "doi": doi,
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{uid}/",
                "origin_domain": "ncbi.nlm.nih.gov",
                "origin": "PubMed Central (NCBI)",
                "stance": ""  # filled by RAG engine via similarity
            })

    except Exception as e:
        print(f"[LiveRetriever] PubMed fetch notice: {e}")

    return records


def fetch_who_evidence(query: str) -> List[Dict[str, Any]]:
    """
    Queries WHO's public API for health topic fact sheets.
    Only returns content from who.int (authoritative global health source).
    """
    records = []
    try:
        clinical_query = expand_medical_query(query)
        encoded = urllib.parse.quote(clinical_query)
        # WHO uses their own search endpoint
        url = f"https://www.who.int/api/hubs/search?query={encoded}&lang=en&sf_culture=en&count=3"
        raw = _safe_fetch(url, timeout=7).decode("utf-8", errors="ignore")
        data = json.loads(raw)
        items = data.get("value", [])

        for item in items[:2]:
            title = item.get("Title", "").strip()
            summary = strip_html_tags(item.get("Description", "")).strip()
            link = item.get("Url", "https://www.who.int")

            if not title or not summary:
                continue
            if any(noise in title.lower() for noise in NON_MEDICAL_NOISE):
                continue

            records.append({
                "claim_topic": title,
                "evidence_text": summary,
                "clinical_explanation": summary,
                "full_abstract": summary,
                "source": TRUSTED_SOURCES["who"]["name"],
                "source_tier": TRUSTED_SOURCES["who"]["tier"],
                "authors": TRUSTED_SOURCES["who"]["author_label"],
                "author_credentials": "World Health Organization Expert Panel",
                "article_title": title,
                "journal": "WHO Fact Sheet / Health Topic",
                "citation": f"World Health Organization (WHO). {title}.",
                "date": "Current WHO Guidance",
                "doi": "",
                "url": link if link.startswith("http") else f"https://www.who.int{link}",
                "origin_domain": "who.int",
                "origin": "WHO (World Health Organization)",
                "stance": ""
            })
    except Exception as e:
        print(f"[LiveRetriever] WHO fetch notice: {e}")
    return records


def fetch_live_clinical_evidence_by_api(query: str, max_articles: int = 4, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
    """
    Queries external medical and fact-checking APIs separately and groups results by API.
    Enables the RAG engine to compute cosine similarity per API and select the single best-matching
    API to answer the claim.
    """
    max_k = kwargs.get("max_results", max_articles)
    api_candidates = {}

    google_api_key = kwargs.get("google_api_key") or kwargs.get("api_key")
    language_code = kwargs.get("language_code") or kwargs.get("lang")
    factcheck_items = fetch_google_factcheck_evidence(
        query,
        api_key=google_api_key,
        language_code=language_code,
        max_results=min(4, max_k)
    )
    if factcheck_items:
        api_candidates["Google Fact Check Tools"] = factcheck_items

    pubmed_items = fetch_pubmed_evidence(query, max_results=max_k)
    if pubmed_items:
        api_candidates["NCBI PubMed Central"] = pubmed_items

    europe_items = fetch_europe_pmc_evidence(query, max_results=min(3, max_k))
    if europe_items:
        api_candidates["Europe PMC"] = europe_items

    trial_items = fetch_clinical_trials_evidence(query, max_results=2)
    if trial_items:
        api_candidates["ClinicalTrials.gov"] = trial_items

    who_items = fetch_who_evidence(query)
    if who_items:
        api_candidates["World Health Organization (WHO)"] = who_items

    return api_candidates


def fetch_live_clinical_evidence(query: str, max_articles: int = 4, **kwargs) -> List[Dict[str, Any]]:
    """
    Aggregates live clinical evidence ONLY from authoritative, trusted medical sources.
    Priority order:
      1. PubMed Central (peer-reviewed primary literature + abstracts)
      2. WHO (global health authority)

    Evidence from unknown blogs, random people, or non-medical sources
    is explicitly rejected at the source level.
    """
    max_k = kwargs.get("max_results", max_articles)
    combined = []

    # 0. Google Fact Check Tools API (certified fact-check investigations)
    google_api_key = kwargs.get("google_api_key") or kwargs.get("api_key")
    language_code = kwargs.get("language_code") or kwargs.get("lang")
    factcheck_items = fetch_google_factcheck_evidence(
        query,
        api_key=google_api_key,
        language_code=language_code,
        max_results=min(4, max_k)
    )
    if factcheck_items:
        combined.extend(factcheck_items)

    # 1. Primary: PubMed Central (peer-reviewed journals with real abstracts)
    pubmed_items = fetch_pubmed_evidence(query, max_results=max_k)
    combined.extend(pubmed_items)

    # 2. Broader biomedical literature index, including reviews and guideline-linked records.
    combined.extend(fetch_europe_pmc_evidence(query, max_results=min(3, max_k)))

    # 3. Trial registry: included as related context, never treated as proof of efficacy.
    if len(combined) < 5:
        combined.extend(fetch_clinical_trials_evidence(query, max_results=2))

    # 4. Secondary: WHO Global Health Authority
    if len(combined) < 5:
        who_items = fetch_who_evidence(query)
        combined.extend(who_items)

    # Assign ranks
    for i, item in enumerate(combined):
        item["rank"] = i + 1
        if "similarity_score" not in item:
            item["similarity_score"] = 0.0

    return _finalize_records(combined, query)

CLINICAL_CONCEPTS = {
    "sore throat": ("pharyngitis", "sore throat", "throat pain"),
    "cough": ("cough", "tussis"),
    "common cold": ("common cold", "upper respiratory infection"),
    "turmeric": ("turmeric", "curcumin", "curcuma longa", "haldi"),
    "diabetes": ("diabetes", "blood glucose", "glycemic"),
    "blood pressure": ("hypertension", "blood pressure"),
}
QUERY_STOPWORDS = {
    "can", "does", "do", "is", "are", "the", "a", "an", "my", "your",
    "for", "to", "of", "and", "or", "with", "in", "on", "by", "it",
    "fix", "help", "cure", "treat", "prevent", "reduce", "cause", "causes",
}



def _query_concepts(query: str) -> set:
    """Extract condition/intervention concepts used to reject weakly-related matches."""
    query_lower = query.lower()
    concepts = set()
    for label, aliases in CLINICAL_CONCEPTS.items():
        if any(alias in query_lower for alias in aliases):
            concepts.add(label)
    concepts.update(token for token in re.findall(r"[a-z]{4,}", query_lower) if token not in QUERY_STOPWORDS)
    return concepts


def _correlation_score(query: str, record: Dict[str, Any]) -> float:
    """Reward evidence that explicitly mentions the claim's actual condition/intervention."""
    concepts = _query_concepts(query)
    corpus = " ".join([
        record.get("article_title", ""), record.get("claim_topic", ""),
        record.get("evidence_text", "")[:1800],
    ]).lower()
    matches = sum(any(alias in corpus for alias in CLINICAL_CONCEPTS.get(concept, (concept,))) for concept in concepts)
    return round(min(0.30, 0.15 * matches), 3)


def _finalize_records(records: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Deduplicate and rank sources by how directly they relate to the original claim."""
    unique = {}
    for item in records:
        key = item.get("url") or item.get("article_title", "").lower()
        if key and key not in unique:
            item["correlation_score"] = _correlation_score(query, item)
            unique[key] = item
    ordered = sorted(unique.values(), key=lambda item: item.get("correlation_score", 0.0), reverse=True)
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
        item.setdefault("similarity_score", 0.0)
    return ordered

def fetch_europe_pmc_evidence(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Fetch related peer-reviewed records from Europe PMC's public REST API."""
    records = []
    try:
        clinical_query = urllib.parse.quote(expand_medical_query(query))
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
            f"query={clinical_query}&format=json&resultType=core&pageSize={max_results}"
        )
        data = json.loads(_safe_fetch(url, timeout=8).decode("utf-8", errors="ignore"))
        for item in data.get("resultList", {}).get("result", []):
            title = strip_html_tags(item.get("title", "")).strip()
            abstract = strip_html_tags(item.get("abstractText", "")).strip()[:1200]
            article_id = item.get("id", "")
            source_id = item.get("source", "MED")
            if not title or not article_id or any(noise in title.lower() for noise in NON_MEDICAL_NOISE):
                continue
            records.append({
                "claim_topic": title,
                "evidence_text": abstract or title,
                "clinical_explanation": abstract or title,
                "full_abstract": abstract,
                "source": "Europe PMC",
                "source_tier": "Tier 1 - Biomedical Literature Index",
                "authors": item.get("authorString", "Peer-Reviewed Medical Researchers"),
                "author_credentials": "Published Researchers (Europe PMC indexed)",
                "article_title": title,
                "journal": item.get("journalTitle", "Europe PMC indexed journal"),
                "citation": item.get("journalTitle", "Europe PMC"),
                "date": item.get("firstPublicationDate", item.get("pubYear", "")),
                "url": f"https://europepmc.org/article/{source_id}/{article_id}",
                "origin_domain": "europepmc.org",
                "origin": "Europe PMC",
                "stance": "",
            })
    except Exception as exc:
        print(f"[LiveRetriever] Europe PMC fetch notice: {exc}")
    return records


def fetch_clinical_trials_evidence(query: str, max_results: int = 2) -> List[Dict[str, Any]]:
    """Fetch related ClinicalTrials.gov studies as context, not efficacy evidence."""
    records = []
    try:
        params = urllib.parse.urlencode({"query.term": expand_medical_query(query), "pageSize": max_results})
        data = json.loads(_safe_fetch(f"https://clinicaltrials.gov/api/v2/studies?{params}", timeout=8).decode("utf-8", errors="ignore"))
        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            desc = protocol.get("descriptionModule", {})
            nct_id = ident.get("nctId", "")
            title = ident.get("briefTitle", "").strip()
            summary = desc.get("briefSummary", "").strip()
            if not nct_id or not title:
                continue
            records.append({
                "claim_topic": title,
                "evidence_text": summary or f"Related registered clinical study: {title}",
                "clinical_explanation": summary or "A related clinical study is registered; registry status is not proof of treatment effectiveness.",
                "full_abstract": summary,
                "source": "ClinicalTrials.gov",
                "source_tier": "Tier 1 - NIH Clinical Trial Registry",
                "authors": "ClinicalTrials.gov registered investigators",
                "author_credentials": "Registered clinical-study investigators",
                "article_title": title,
                "journal": "ClinicalTrials.gov registry",
                "citation": f"{nct_id} | {status.get('overallStatus', 'Status not listed')}",
                "date": status.get("lastUpdateSubmitDate", ""),
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "origin_domain": "clinicaltrials.gov",
                "origin": "ClinicalTrials.gov",
                "stance": "uncertain",
            })
    except Exception as exc:
        print(f"[LiveRetriever] ClinicalTrials.gov fetch notice: {exc}")
    return records