# 🩺 Multilingual Healthcare Misinformation RAG (Dual-Mode: Local & Live)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/Vector_Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-MiniLM--L12--v2-orange.svg)](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
[![Gradio UI](https://img.shields.io/badge/Interface-Gradio-red.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clinical misinformation verification system utilizing **Retrieval-Augmented Generation (RAG)** across **English**, **Hindi (हिंदी)**, and **Hinglish** (Romanized Hindi).

This repository is the **Dual-Mode Live Edition**, integrating curated multi-source clinical evidence with real-time global medical APIs.

---

## 🎯 Dual-Mode Architectural Overview

| Feature | 🗂️ Mode 1: Curated Clinical Dataset | 🌐 Mode 2: Live Real-Time Multi-API Retrieval |
| :--- | :--- | :--- |
| **Data Source** | 8,694 unified peer-reviewed records (HealthVer, MEDIQ, Hinglish, HealthFC) | Live external APIs (Google Fact Check, PubMed, Europe PMC, ClinicalTrials.gov, WHO) |
| **Local Dataset Access** | Full dense semantic search over 8,694 records | **100% Bypassed (Zero local dataset used)** |
| **Latency** | Instantaneous (< 5ms via precomputed 384-D FAISS index) | Real-time HTTP API retrieval (~1.5s - 2.5s) |
| **API Key Required** | None (Fully offline capable) | Optional Google Fact Check API key (PubMed, Europe PMC, ClinicalTrials are keyless) |
| **Tournament Selection** | Pre-indexed vector scoring with correlation weighting | **Real-time cosine similarity tournament across all APIs** |
| **Use Case** | Known clinical questions, established trials, rapid evaluation | Emerging epidemics, novel treatments, trending viral claims |

---

## 🌟 Core System Features & Innovations

### 1. 🌐 Real-Time Multi-API Tournament Selection (Mode 2)
Mode 2 connects concurrently to Tier 1 medical repositories:
- **Google Fact Check Tools API**: Fetches certified claim reviews from IFCN signatories (Snopes, Boom Live, FactCheck.org, Health Feedback, Vishwas News).
- **NCBI PubMed Central**: Open-access biomedical articles filtered to systematic reviews, meta-analyses, and RCTs with direct XML abstract extraction.
- **Europe PMC**: Comprehensive biomedical indexing by EMBL-EBI.
- **ClinicalTrials.gov**: NIH clinical trial protocols and registered outcomes.
- **WHO Global Health Guidelines**: Official international advisories.

**The Tournament Engine**: Encodes article abstracts dynamically and measures the pure cosine similarity to the claim:
$$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
The API producing the highest cosine similarity wins the tournament, and its top evidence is used to formulate the scientific verdict.

### 2. 🗂️ Expanded 8,694 Multi-Source Clinical Dataset (Mode 1)
Mode 1 features an expanded, standardized clinical knowledge base:
- **HealthVer (5,000 records)**: Peer-reviewed claim-evidence pairs from PubMed with ground-truth stance classifications.
- **MEDIQ Clinical Knowledge (2,000 records)**: Real-world clinical hospital Q&A from King George's Medical University (KGMU) faculty.
- **Hinglish Health Knowledgebase (944 records)**: Authentic code-mixed health communication data.
- **Medizin Transparent / HealthFC (750 records)**: Fact-checked clinical trials affiliated with Cochrane Austria.

### 3. 🔬 Deep Sentence-Level Excerpt Mining
Rather than returning monolithic text blobs, the system splits candidate abstracts into individual sentences, vector-encodes each sentence, and extracts the top 3 most relevant sentences matching the query.

### 4. 🛡️ NLP Domain Guardrail Filter
An intelligent health classifier screens all queries prior to retrieval:
- Validates medical entity presence (symptoms, treatments, diseases, anatomy).
- Mode 2 employs an API probing fallback for ambiguous vernacular terms.
- Rejects non-health queries (e.g., *"India won the cricket match"*) with an educational explanation.

### 5. 🗣️ Code-Mixed Multilingual Preprocessing
- **Devanagari Preservation**: Maintains Unicode ranges (`\u0900`–`\u097F`).
- **Hinglish Expansion**: Normalizes phonetic spelling variants (`thik` $\rightarrow$ `theek`, `dawa` $\rightarrow$ `dawai`) and expands colloquial terms to biomedical terminology (`haldi doodh` $\rightarrow$ `turmeric curcumin anti-inflammatory`).
- **Language-Tailored Explanations**: Generates explanations in English, Hindi, or Hinglish based on the detected query language.

---

## 📁 Repository Structure

```
multilingual-healthcare-rag-live/
├── README.md                                          # Documentation & Architecture Guide
├── requirements.txt                                  # Python dependencies
├── data/
│   ├── evidence_dataset.csv                           # 8,694 unified clinical records
│   ├── evidence_embeddings.npy                       # Precomputed 384-D FAISS embeddings
│   ├── Datensatz_raw.csv                              # Original raw clinical trials
│   └── eda_summary_plots.png                          # EDA statistical distribution visualization
├── hinglish dataset/
│   ├── hinglish_corpus_final.tsv                     # Code-mixed linguistic corpus
│   └── hinglish/                                      # Language split files
├── medical claim dataset/
│   └── medical_nlp_training_datasets.xlsx            # Source benchmarks (HealthVer, MEDIQ, HealthFC)
├── src/
│   ├── __init__.py
│   ├── app.py                                        # Gradio Web UI with Dual-Mode selector
│   ├── live_retriever.py                              # External API connectors (PubMed, WHO, Google Fact Check)
│   ├── rag_engine.py                                 # Vector embeddings, FAISS indexing & tournament engine
│   ├── verdict_engine.py                             # Stance arbitration, risk rating & explanation
│   ├── health_classifier.py                          # NLP domain guardrail filter
│   ├── language_detector.py                          # 3-way language classifier (EN / HI / Hinglish)
│   ├── preprocessing.py                              # NFKC Unicode cleaner & Hinglish normalizer
│   └── eda.py                                        # Statistical data analysis script
├── scripts/
│   ├── prepare_mode1_dataset.py                      # Multi-source dataset labeling & embedding builder
│   └── generate_notebook.py                          # Colab research notebook generator
└── tests/
    └── test_pipeline.py                              # Automated unit & integration tests
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/aman2428cseaiml1702/Medical-fact-claim-clone-1.git
cd Medical-fact-claim-clone-1
pip install -r requirements.txt
```

### 2. Launch the Web Interface
```bash
python src/app.py
```
Open your browser at: **`http://127.0.0.1:7860`**

### 3. Optional: Configure Google Fact Check API
For certified IFCN fact-checking in Mode 2, set your API key as an environment variable or enter it directly into the Gradio UI:
```bash
export GOOGLE_FACTCHECK_API_KEY="your-api-key-here"
```
*(PubMed Central, Europe PMC, and ClinicalTrials.gov work out of the box without any API key.)*

---

## 🧪 Benchmark Test Claims

| # | Test Claim | Language | Engine Mode | Expected Verdict | Risk Level |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Can masks reduce corona infections when worn by a large proportion of the population?` | English | Curated Dataset (8,694) | **Supported** | Low |
| 2 | `Does arthroscopy help with osteoarthritis of the knee joint?` | English | Curated Dataset (8,694) | **Refuted** | High |
| 3 | `sugar khane se diabetes hota hai` | Hinglish | Curated Dataset (8,694) | **Uncertain / Refuted** | Medium |
| 4 | `Haldi doodh cancer ko theek karta hai` | Hinglish | Live Multi-API Retrieval | **Uncertain / Refuted** | High |
| 5 | `Pineapple reduces acidity` | English | Live Multi-API Retrieval | **Refuted / Uncertain** | Medium |
| 6 | `India won the cricket match` | English | NLP Guardrail | **Non-Health Claim** | None |

---

## 📜 Scientific Methodology & Citations

All verdicts produced by this system cite authentic peer-reviewed literature with direct, clickable links to official sources:
- **US National Library of Medicine (NCBI PubMed)**: `https://pubmed.ncbi.nlm.nih.gov/`
- **Cochrane Library Systematic Reviews**: `https://www.cochranelibrary.com/`
- **Medizin Transparent Evidence**: `https://www.medizin-transparent.at/`
- **Google Fact Check Tools / IFCN**: `https://toolbox.google.com/factcheck/explorer`
- **World Health Organization (WHO)**: `https://www.who.int/`
