# 🩺 Multilingual Healthcare Misinformation RAG (Dual-Mode: Local & Live)

A clinical misinformation verification system utilizing **Retrieval-Augmented Generation (RAG)** across **English**, **Hindi (हिंदी)**, and **Hinglish**.

This repository is the **Dual-Mode Live edition**, featuring both:
1. **Mode 1: Curated Clinical Dataset (8,694 Records)** — Comprehensive multi-source clinical benchmark combining *Medizin Transparent* / Cochrane trials (750 records), *HealthVer* peer-reviewed evidence (5,000 records), *MEDIQ* clinical hospital Q&A (2,000 records), and *Hinglish Health Knowledgebase* (944 records) for deterministic, fast (<5ms) evaluations.
2. **Mode 2: Live Real-Time Retrieval (PubMed Central & Medical Consensus APIs)** — Real-time dynamic fetch of biomedical literature directly from the US National Library of Medicine (PubMed/NCBI) and Wikipedia Medical Consensus APIs, vectorized on-the-fly into an ephemeral in-memory FAISS index.

---

## 🌟 Project Highlights

- **Dual-Mode Architecture**: Seamlessly switch between the 8,694-record curated clinical dataset and live real-time web retrieval via PubMed Central.
- **On-The-Fly FAISS Indexing**: Dynamically downloads open-access PMC study abstracts, generates 384-D dense embeddings via `paraphrase-multilingual-MiniLM-L12-v2`, and builds an ephemeral inner-product FAISS index in under 1ms.
- **Cross-Lingual Dense Semantic Retrieval**: Aligns Hinglish (Romanized Hindi) and Devanagari Hindi queries with English medical literature.
- **Data Preprocessing Pipeline**: NFKC Unicode normalization, Devanagari script preservation (`\u0900-\u097F`), and Hinglish phonetic variant normalization (`thik` → `theek`, `dawa` → `dawai`).
- **Domain Guardrail**: Early rule-based classification that filters out non-health claims (e.g., *"India won the cricket match"*) before vector retrieval.
- **Explainable Verdict & Risk Assessment**: Maps claims to `Supported`, `Refuted`, `Uncertain`, or `No relevant evidence`, with risk levels (`Low`, `Medium`, `High`, `Very High`).
- **Zero Hallucination & Authentic Citations**: Backed by clickable URLs directly to `ncbi.nlm.nih.gov` or official Cochrane trial pages.
- **Interactive Gradio Web UI**: Features a radio selector to switch between Local and Live Real-Time modes.

---

## 📁 Repository Structure

```
multilingual-healthcare-rag-live/
├── README.md                                          # Documentation & Usage Guide
├── requirements.txt                                  # Dependency specifications
├── data/
│   ├── Datensatz_raw.csv                              # Original Datensatz source dataset (750 records)
│   ├── eda_summary_plots.png                          # 3-panel EDA statistical visualization
│   └── evidence_dataset.csv                           # 8,694 standardized clinical evidence records with URLs
├── src/
│   ├── __init__.py
│   ├── eda.py                                        # Exploratory Data Analysis & statistical plotting script
│   ├── live_retriever.py                              # Live PubMed Central (NCBI) & Wikipedia API connector
│   ├── preprocessing.py                              # Text cleaning & Hinglish normalization
│   ├── language_detector.py                          # Hindi (Devanagari) / Hinglish / English detector
│   ├── health_classifier.py                          # Rule-based health claim filter
│   ├── rag_engine.py                                 # SentenceTransformer, FAISS & dynamic live indexer
│   ├── verdict_engine.py                             # Verdict evaluation, risk rating, & explanation
│   └── app.py                                        # Gradio web application with Dual-Mode selector
├── scripts/
│   └── generate_notebook.py                          # Colab notebook generator script
└── tests/
    └── test_pipeline.py                              # Automated unit and integration test suite
```

---

## 💻 Running the Live Project Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Gradio Web Application
```bash
python3 src/app.py
```
Open your browser at: **`http://127.0.0.1:7860`**

### 3. Using the Dual-Mode Selector in the UI:
- **Curated Dataset (8,694 Clinical Records)**: Fast, deterministic verification against the 8,694 multi-source clinical database.
- **Live Real-Time Retrieval (PubMed Central & Medical APIs)**: Queries PubMed Central in real-time, vectorizes articles on-the-fly, and verifies any novel health claim with live `ncbi.nlm.nih.gov` links!

---

## 🧪 Evaluation Test Claims

| # | Demo Claim | Mode | Expected Verdict | Risk |
| :- | :--- | :--- | :--- | :--- |
| 1 | `Can masks reduce corona infections when worn by a large proportion of the population?` | Curated Dataset | **Supported** | Low |
| 2 | `Does arthroscopy help with osteoarthritis of the knee joint?` | Curated Dataset | **Refuted** | High |
| 3 | `Haldi doodh cancer ko theek karta hai` | Live PubMed API | **Uncertain / Refuted** | High |
| 4 | `can the preventive in take of lysine prevent cold sores` | Curated Dataset | **Uncertain** | Medium |
| 5 | `India won the cricket match` | Guardrail Filter | **Non-Health Claim** | None |
