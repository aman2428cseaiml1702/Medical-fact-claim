import os
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from src.health_classifier import is_health_related
from src.live_retriever import fetch_live_clinical_evidence
from src.verdict_engine import assess_verdict_and_risk, generate_explanation

test_queries = [
    # 1. Scientific claim (Supported)
    "Drinking enough water helps maintain kidney function and hydration",
    # 2. Refuted claim
    "Drinking bleach cures viral infections and diseases",
    # 3. Uncertain claim (Not proven yet)
    "Eating dark chocolate every morning prevents Alzheimer disease",
    # 4. Non-health claim (should be rejected)
    "can eating sugar causes nasa to run"
]

print("=" * 60)
print("TESTING FULL END-TO-END CLINICAL VERIFICATION")
print("=" * 60)

for q in test_queries:
    is_health = is_health_related(q)
    print(f"\nClaim: {q}")
    print(f"Health Classified: {is_health}")
    
    if not is_health:
        print("-> Rejected as non-health (Expected).")
        continue

    print("Fetching live evidence from NCBI PubMed PMC & WHO...")
    evidence = fetch_live_clinical_evidence(q, max_results=3)
    print(f"Retrieved {len(evidence)} clinical articles.")
    
    for i, ev in enumerate(evidence):
        print(f"\n--- [Article {i+1}] ---")
        print(f"Title: {ev.get('article_title')}")
        print(f"Authors: {ev.get('authors')}")
        print(f"Credentials: {ev.get('author_credentials')}")
        print(f"Tier: {ev.get('source_tier')}")
        print(f"Source: {ev.get('source')}")
        print(f"Link: {ev.get('url')}")
        print(f"Abstract preview in quotes: \"{ev.get('full_abstract', '')[:180]}...\"")

    verdict_data = assess_verdict_and_risk(q, evidence)
    print(f"\nVerdict: {verdict_data['verdict']}")
    print(f"Risk: {verdict_data['risk']}")

print("\nAll tests completed!")
