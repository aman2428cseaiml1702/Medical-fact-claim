"""
Gradio Web Application for Multilingual Healthcare Misinformation RAG
Allows users to enter health-related or general claims in English, Hindi, or Hinglish
and receive automated verification, evidence citations, and risk assessment.
"""

import os
import sys

# Prevent macOS OpenMP / Rust tokenizers thread conflict
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings

# Suppress urllib3 and SSL warnings on macOS
warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gradio as gr

try:
    from src.preprocessing import preprocess_text
    from src.language_detector import detect_language
    from src.health_classifier import is_health_related
    from src.rag_engine import HealthcareRAGEngine
    from src.verdict_engine import assess_verdict_and_risk, generate_explanation
except ModuleNotFoundError:
    from preprocessing import preprocess_text
    from language_detector import detect_language
    from health_classifier import is_health_related
    from rag_engine import HealthcareRAGEngine
    from verdict_engine import assess_verdict_and_risk, generate_explanation

# Default dataset path
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evidence_dataset.csv")

# Global RAG Engine instance (lazy-loaded on first use)
_rag_engine = None


def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        print("Initializing Multilingual RAG Engine...")
        _rag_engine = HealthcareRAGEngine(csv_path=DATASET_PATH)
        print("RAG Engine ready.")
    return _rag_engine


def generate_verdict_banner(verdict: str, risk: str, top_author: str = "", top_source: str = "", top_url: str = "", top_title: str = "", claim: str = "") -> str:
    """Generates a high-impact, visual status banner with direct article link buttons and claim details."""
    article_link_btn = ""
    if top_url and top_url != "#":
        article_link_btn = f"""
        <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.25);">
            <div style="font-size: 13px; opacity: 0.95; margin-bottom: 6px;">
                <strong>🔎 Investigated Claim:</strong> <em>"{claim}"</em>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 13px;">
                    <strong>📄 Official Article:</strong> <span>{top_title}</span>
                </div>
                <a href="{top_url}" target="_blank" style="background: white; color: #0f172a; padding: 6px 14px; border-radius: 6px; font-weight: 700; font-size: 13px; text-decoration: none; box-shadow: 0 2px 6px rgba(0,0,0,0.25);">
                    🌐 Open Official Article Page ↗
                </a>
            </div>
        </div>
        """

    if verdict == "Supported":
        return f"""
        <div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 8px 24px rgba(16, 185, 129, 0.35); border: 2px solid #34d399;">
            <div style="font-size: 22px; font-weight: 800; display: flex; align-items: center; gap: 10px;">
                <span>✅ SCIENTIFIC VERDICT: SUPPORTED</span>
            </div>
            <div style="font-size: 15px; margin-top: 8px; line-height: 1.5; opacity: 0.96;">
                <strong>Corroborated by Verified Clinical Evidence!</strong> This medical claim is substantiated by systematic reviews, peer-reviewed clinical trials, or medical consensus.
            </div>
            <div style="margin-top: 14px; display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; font-weight: 600; background: rgba(0,0,0,0.18); padding: 8px 14px; border-radius: 8px;">
                <span>🛡️ Risk Level: <strong>LOW</strong></span>
                <span>•</span>
                <span>📑 Evidence Found: <strong>YES (Clinically Validated)</strong></span>
                <span>•</span>
                <span>🧑‍⚕️ Evaluated by: <strong>{top_author}</strong></span>
                <span>•</span>
                <span>🏛️ RAG Source: <strong>{top_source}</strong></span>
            </div>
            {article_link_btn}
        </div>
        """
    elif verdict == "Refuted":
        return f"""
        <div style="background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%); color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 8px 24px rgba(239, 68, 68, 0.35); border: 2px solid #f87171;">
            <div style="font-size: 22px; font-weight: 800; display: flex; align-items: center; gap: 10px;">
                <span>❌ SCIENTIFIC VERDICT: REFUTED</span>
            </div>
            <div style="font-size: 15px; margin-top: 8px; line-height: 1.5; opacity: 0.96;">
                <strong>Contradicted by Medical Research!</strong> Clinical evidence directly contradicts or disproves this claim.
            </div>
            <div style="margin-top: 14px; display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; font-weight: 600; background: rgba(0,0,0,0.18); padding: 8px 14px; border-radius: 8px;">
                <span>⚠️ Risk Level: <strong>{risk.upper()}</strong></span>
                <span>•</span>
                <span>📑 Evidence Found: <strong>YES (Directly Disproves Claim)</strong></span>
                <span>•</span>
                <span>🧑‍⚕️ Evaluated by: <strong>{top_author}</strong></span>
            </div>
            {article_link_btn}
        </div>
        """
    elif "Uncertain" in verdict:
        return f"""
        <div style="background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 8px 24px rgba(245, 158, 11, 0.35); border: 2px solid #fbbf24;">
            <div style="font-size: 21px; font-weight: 800; display: flex; align-items: center; gap: 10px;">
                <span>⚠️ SCIENTIFIC VERDICT: UNCERTAIN / NOT PROVEN YET</span>
            </div>
            <div style="font-size: 15px; margin-top: 8px; line-height: 1.5; opacity: 0.96;">
                <strong>Scientific Proof is Missing or Inconclusive!</strong> Controlled clinical trials and systematic reviews do not substantiate this claim.
            </div>
            <div style="margin-top: 14px; display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; font-weight: 600; background: rgba(0,0,0,0.18); padding: 8px 14px; border-radius: 8px;">
                <span>⚡ Risk Level: <strong>{risk.upper()}</strong></span>
                <span>•</span>
                <span>📑 Evidence Status: <strong>NOT PROVEN YET</strong></span>
                <span>•</span>
                <span>🧑‍⚕️ Evaluated by: <strong>{top_author}</strong></span>
                <span>•</span>
                <span>🏛️ RAG Source: <strong>{top_source}</strong></span>
            </div>
            {article_link_btn}
        </div>
        """
    elif "Non-Health" in verdict:
        display_claim = claim if claim else "your query"
        return f"""
        <div style="background: linear-gradient(135deg, #334155 0%, #475569 100%); color: white; padding: 22px 26px; border-radius: 14px; margin-bottom: 16px; box-shadow: 0 6px 20px rgba(51, 65, 85, 0.4); border: 2px solid #94a3b8;">
            <div style="font-size: 22px; font-weight: 800; display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <span>🚫 NOT A HEALTH-RELATED QUERY</span>
            </div>
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; border-left: 4px solid #f59e0b;">
                <div style="font-size: 12px; opacity: 0.75; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px;">❌ Rejected Query</div>
                <div style="font-size: 15px; font-style: italic; font-weight: 600;">"{display_claim}"</div>
            </div>
            <div style="font-size: 14px; line-height: 1.6; opacity: 0.95; margin-bottom: 12px;">
                <strong>Why was this filtered?</strong> Our NLP domain guardrail analyzed your input and determined it does not contain a verifiable medical or health claim. 
                The system checks for recognized disease names, symptoms, treatments, medications, and health context — none were found for this query.
            </div>
            <div style="font-size: 13px; opacity: 0.85; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 10px;">
                <strong>💡 Try a health claim instead, for example:</strong><br/>
                &nbsp;&nbsp;• "Can eating sugar cause diabetes?" &nbsp;|&nbsp; "Does the COVID vaccine prevent infection?"<br/>
                &nbsp;&nbsp;• "Kya sugar se diabetes hota hai?" (Hinglish) &nbsp;|&nbsp; "क्या कोविड वैक्सीन सुरक्षित है?" (Hindi)
            </div>
        </div>
        """
    return ""


def verify_claim(claim: str, mode: str = "Curated Dataset (750 Clinical Records)"):
    """
    Orchestrates the 10-step verification workflow with dual-mode support:
    1. Local Curated Dataset (750 Clinical Records)
    2. Live Real-Time Retrieval (PubMed Central & Medical Consensus APIs)
    """
    if not claim or not claim.strip():
        placeholder_banner = """
        <div style="background: #f8fafc; color: #64748b; padding: 16px 20px; border-radius: 10px; border: 1px dashed #cbd5e1; font-size: 14px;">
            👉 Please enter a clinical or health claim to verify.
        </div>
        """
        return (
            placeholder_banner,
            "N/A",
            "N/A",
            "Please enter a claim to verify.",
            "N/A",
            "No claim submitted.",
            "",
            ""
        )

    # Step 4: Preprocess
    cleaned_claim = preprocess_text(claim)

    # Step 5: Detect Language
    lang = detect_language(claim)

    # Step 6: Health Classification
    health_flag = is_health_related(cleaned_claim)

    if not health_flag:
        verdict_data = {
            "verdict": "Non-Health Claim",
            "evidence_status": "Filtered before retrieval",
            "risk": "None",
            "confidence": 0.0,
            "stance": "none"
        }
        explanation = generate_explanation(claim, lang, False, verdict_data, [])
        banner_html = generate_verdict_banner("Non-Health Claim", "None", claim=claim)
        return (
            banner_html,
            lang,
            "No (Filtered Out)",
            "Non-Health Claim",
            "None",
            explanation["summary_text"],
            "No retrieval performed for non-health query.",
            explanation["disclaimer"]
        )

    # Step 7: Retrieve Evidence via FAISS / Dense Vector Index (Local vs. Live API)
    engine = get_rag_engine()
    if "Live" in mode:
        retrieved = engine.retrieve_live_evidence(cleaned_claim, top_k=3)
    else:
        retrieved = engine.retrieve_evidence(cleaned_claim, top_k=3)

    # Step 8: Verdict & Risk
    verdict_data = assess_verdict_and_risk(cleaned_claim, retrieved)

    # Step 9: Generate Explanation
    explanation = generate_explanation(claim, lang, True, verdict_data, retrieved)

    # Extract top evidence metadata
    top_item = retrieved[0] if retrieved else {}
    top_author = top_item.get("authors", "Medical Evaluators (Cochrane Collaboration)")
    top_source = top_item.get("source", "Clinical Evidence Base")
    top_source_tier = top_item.get("source_tier", "Verified Medical Source")
    top_credentials = top_item.get("author_credentials", "Peer-Reviewed Medical Researchers")
    top_url = top_item.get("url", "#")
    top_title = top_item.get("article_title", top_item.get("claim_topic", "Clinical Review"))
    top_journal = top_item.get("journal", "")
    top_citation = top_item.get("citation", "")
    top_date = top_item.get("date", "")
    top_explanation = top_item.get("clinical_explanation", top_item.get("evidence_text", ""))
    top_full_abstract = top_item.get("full_abstract", top_item.get("evidence_text", ""))

    def _quote_abstract(text: str, limit: int = 900) -> str:
        """Format the full article abstract in apostrophes, truncated cleanly."""
        if not text or len(text.strip()) < 20:
            return ""
        t = text.strip()
        if len(t) > limit:
            cut = t[:limit]
            lp = max(cut.rfind(". "), cut.rfind(".\n"))
            t = (cut[:lp + 1] if lp > limit * 0.6 else cut) + " [...]"
        return f'"{t}"'

    # Generate visual verdict banner with direct link button
    banner_html = generate_verdict_banner(
        verdict_data["verdict"],
        verdict_data["risk"],
        top_author=top_author,
        top_source=top_source,
        top_url=top_url,
        top_title=top_title,
        claim=claim
    )

    # ── Format Evidence Markdown with full abstracts in apostrophes ──────────
    evidence_md = ""

    # ── TOP RESULT HIGHLIGHT CARD ────────────────────────────────────────────
    abstract_quoted = _quote_abstract(top_full_abstract)

    if verdict_data["verdict"] == "Supported":
        evidence_md += (
            f"### 🟢 SCIENTIFICALLY SUPPORTED — CLINICAL EVIDENCE SUMMARY\n\n"
            f"> **✅ Verified Claim**: *\"{claim}\"*\n>\n"
            f"> **💡 How & When This Claim Is True**:\n"
            f"> {top_explanation}\n>\n"
        )
        if abstract_quoted:
            evidence_md += (
                f"> **📄 Full Article Abstract (from {top_source})**:\n"
                f"> {abstract_quoted}\n>\n"
            )
        evidence_md += (
            f"> **🧑‍⚕️ Authors / Researchers**: **{top_author}**\n"
            f"> **📋 Author Credentials**: {top_credentials}\n"
            f"> **{top_source_tier}**\n"
        )
        if top_journal:
            evidence_md += f"> **📰 Published In**: {top_journal}" + (f" ({top_date})" if top_date else "") + "\n"
        if top_citation:
            evidence_md += f"> **📚 Citation**: *{top_citation}*\n"
        evidence_md += (
            f"> **🔗 Official Article**: [{top_title}]({top_url})\n\n"
            f"---\n\n"
        )
    elif "Uncertain" in verdict_data["verdict"]:
        evidence_md += (
            f"### 🟡 NOT YET PROVEN — CLINICAL EVIDENCE INSUFFICIENT\n\n"
            f"> **⚠️ Investigated Claim**: *\"{claim}\"*\n>\n"
            f"> **🔬 Why This Is Uncertain / Not Proven Yet**:\n"
            f"> {top_explanation}\n>\n"
        )
        if abstract_quoted:
            evidence_md += (
                f"> **📄 Related Study Abstract (from {top_source})**:\n"
                f"> {abstract_quoted}\n>\n"
            )
        evidence_md += (
            f"> **🧑‍⚕️ Reviewed by**: **{top_author}**\n"
            f"> **📋 Credentials**: {top_credentials}\n"
            f"> **{top_source_tier}**\n"
        )
        if top_journal:
            evidence_md += f"> **📰 Published In**: {top_journal}" + (f" ({top_date})" if top_date else "") + "\n"
        if top_citation:
            evidence_md += f"> **📚 Citation**: *{top_citation}*\n"
        evidence_md += (
            f"> **🔗 Investigation Article**: [{top_title}]({top_url})\n\n"
            f"---\n\n"
        )
    elif "Refuted" in verdict_data["verdict"]:
        evidence_md += (
            f"### 🔴 REFUTED — DIRECTLY CONTRADICTED BY MEDICAL SCIENCE\n\n"
            f"> **❌ Disproved Claim**: *\"{claim}\"*\n>\n"
            f"> **🔬 Why This Claim Is Scientifically Refuted**:\n"
            f"> {top_explanation}\n>\n"
        )
        if abstract_quoted:
            evidence_md += (
                f"> **📄 Full Article Abstract (from {top_source})**:\n"
                f"> {abstract_quoted}\n>\n"
            )
        evidence_md += (
            f"> **🧑‍⚕️ Authors / Researchers**: **{top_author}**\n"
            f"> **📋 Credentials**: {top_credentials}\n"
            f"> **{top_source_tier}**\n"
        )
        if top_journal:
            evidence_md += f"> **📰 Published In**: {top_journal}" + (f" ({top_date})" if top_date else "") + "\n"
        if top_citation:
            evidence_md += f"> **📚 Citation**: *{top_citation}*\n"
        evidence_md += (
            f"> **🔗 Clinical Investigation**: [{top_title}]({top_url})\n\n"
            f"---\n\n"
        )

    # ── ALL RETRIEVED ARTICLES (ranked list with full abstracts) ─────────────
    evidence_md += f"### 📚 All Retrieved Clinical Articles [{mode}]\n\n"
    for item in retrieved:
        authors = item.get("authors", "Medical Research Team")
        credentials = item.get("author_credentials", "Peer-Reviewed Researchers")
        source_tier = item.get("source_tier", "Verified Medical Source")
        article_title = item.get("article_title", item.get("claim_topic", "Clinical Investigation"))
        journal = item.get("journal", item.get("source", ""))
        date = item.get("date", "")
        citation = item.get("citation", "")
        item_abstract = item.get("full_abstract", item.get("clinical_explanation", item.get("evidence_text", "")))
        abstract_q = _quote_abstract(item_abstract)

        evidence_md += f"#### 🏷️ Rank {item.get('rank', '?')}: [{article_title}]({item['url']})\n\n"
        evidence_md += f"| Field | Detail |\n|---|---|\n"
        evidence_md += f"| **Scientific Stance** | `{item.get('stance', 'N/A').upper()}` |\n"
        evidence_md += f"| **RAG Match Score** | `{item.get('similarity_score', 'N/A')}` |\n"
        evidence_md += f"| **🧑‍⚕️ Authors** | **{authors}** |\n"
        evidence_md += f"| **📋 Credentials** | {credentials} |\n"
        evidence_md += f"| **{source_tier}** | {item.get('source', '')} |\n"
        if journal:
            evidence_md += f"| **📰 Journal** | {journal}" + (f" ({date})" if date else "") + " |\n"
        if citation:
            evidence_md += f"| **📚 Citation** | *{citation}* |\n"
        evidence_md += f"| **🔗 Article Link** | [{article_title}]({item['url']}) |\n\n"

        if abstract_q:
            evidence_md += f"**📄 Full Article Abstract**:\n\n> {abstract_q}\n\n"
        evidence_md += "---\n"

    return (
        banner_html,
        lang,
        "Yes (Proceeded to Retrieval)",
        verdict_data["verdict"],
        verdict_data["risk"],
        explanation["summary_text"],
        evidence_md,
        explanation["disclaimer"]
    )


def create_demo():
    """Builds the Gradio interface."""
    custom_css = """
    .gradio-container { max-width: 1100px !important; margin: auto; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .gr-button-primary { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; color: white !important; font-weight: 600 !important; }
    """

    with gr.Blocks(title="Multilingual Healthcare Misinformation RAG (Live)", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🩺 Multilingual Healthcare Misinformation Verifier (Live & Local RAG)
            Verify medical claims across **English**, **Hindi (हिंदी)**, and **Hinglish** using cross-lingual dense semantic retrieval.
            Supports both **Curated Medical Claims & Hinglish Dataset (4,946 Clinical Records from HealthVer, HealthFC, MEDIQ, and Hinglish Corpus)** and **Live Real-Time Web Retrieval (NCBI PubMed Central & WHO APIs)**.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                mode_selector = gr.Radio(
                    choices=[
                        "Curated Medical & Hinglish Dataset (4,946 Clinical Records)",
                        "Live Real-Time Retrieval (PubMed Central & Medical APIs)"
                    ],
                    value="Curated Medical & Hinglish Dataset (4,946 Clinical Records)",
                    label="Select Retrieval Engine Mode"
                )
                claim_input = gr.Textbox(
                    label="Enter Health Claim",
                    placeholder="e.g., Can masks reduce corona infections / sugar khane se diabetes hota hai",
                    lines=3,
                )
                verify_btn = gr.Button("🔍 Verify Claim", variant="primary", size="lg")

                gr.Markdown("### 💡 Click Any Example Below to Verify Instantly")
                examples_ui = gr.Examples(
                    examples=[
                        ["Can masks reduce corona infections when worn by a large proportion of the population?", "Curated Medical & Hinglish Dataset (4,946 Clinical Records)"],
                        ["sugar khane se diabetes hota hai", "Curated Medical & Hinglish Dataset (4,946 Clinical Records)"],
                        ["kya turmeric se cancer theek ho sakta hai", "Curated Medical & Hinglish Dataset (4,946 Clinical Records)"],
                        ["Does paxlovid protect unvaccinated people with risk factors from severe or even fatal covid 19 disease?", "Curated Medical & Hinglish Dataset (4,946 Clinical Records)"],
                        ["Can vinegar help in preventing COVID-19?", "Curated Medical & Hinglish Dataset (4,946 Clinical Records)"],
                        ["Haldi doodh cancer ko theek karta hai", "Live Real-Time Retrieval (PubMed Central & Medical APIs)"],
                        ["can eating sugar causes nasa to run", "Curated Medical & Hinglish Dataset (4,946 Clinical Records)"],
                    ],
                    inputs=[claim_input, mode_selector],
                )

            with gr.Column(scale=3):
                verdict_banner = gr.HTML(
                    value="""
                    <div style="background: #f1f5f9; color: #475569; padding: 16px 20px; border-radius: 10px; border: 1px solid #cbd5e1; font-size: 14px; font-weight: 500;">
                        💡 Enter a claim on the left or click an example to run RAG verification.
                    </div>
                    """,
                    label="Scientific Verdict Status & Article Links"
                )
                with gr.Row():
                    lang_output = gr.Textbox(label="Detected Language", interactive=False)
                    health_output = gr.Textbox(label="Health-Related?", interactive=False)
                with gr.Row():
                    verdict_output = gr.Textbox(label="Scientific Verdict", interactive=False)
                    risk_output = gr.Textbox(label="Risk Assessment", interactive=False)

                explanation_output = gr.Markdown(label="Clinical Explanation & How/When True")
                evidence_output = gr.Markdown(label="Retrieved Evidence, Article Links & Citations")
                disclaimer_output = gr.Markdown(label="Disclaimer")

        # Hook up both button click and Enter key submit
        verify_inputs = [claim_input, mode_selector]
        verify_outputs = [
            verdict_banner,
            lang_output,
            health_output,
            verdict_output,
            risk_output,
            explanation_output,
            evidence_output,
            disclaimer_output,
        ]

        verify_btn.click(fn=verify_claim, inputs=verify_inputs, outputs=verify_outputs)
        claim_input.submit(fn=verify_claim, inputs=verify_inputs, outputs=verify_outputs)

    return demo


if __name__ == "__main__":
    print("Pre-loading Multilingual RAG Engine on main thread...")
    get_rag_engine()
    print("RAG Engine ready! Launching web interface...")
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
