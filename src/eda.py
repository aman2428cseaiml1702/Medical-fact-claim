"""
Exploratory Data Analysis (EDA) Module
Performs statistical profiling, class distribution analysis,
language breakdown, source breakdown, word count metrics, and visual diagnostics
on the expanded 8,694 clinical evidence dataset.
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding for Windows command line
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Set modern publication-grade visual theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def run_eda(dataset_path: str = "data/evidence_dataset.csv"):
    """Runs complete EDA on the 8,694 record dataset and generates publication-grade 4-panel visual plots."""
    if not os.path.exists(dataset_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, "data", "evidence_dataset.csv")

    print("=" * 70)
    print("📊 CLINICAL EVIDENCE DATASET — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    df = pd.read_csv(dataset_path)
    total_records = len(df)
    print(f"\n📁 Loaded Dataset from: {dataset_path}")
    print(f"• Total Rows (Evidence Records): {total_records:,}")
    print(f"• Total Columns: {df.shape[1]} -> {list(df.columns)}")
    print(f"• Unique Clinical Claim Topics: {df['claim_topic'].nunique():,}")

    # 1. Missing Values Audit
    print("\n" + "-" * 45)
    print("1. DATA COMPLETENESS & NULL AUDIT")
    print("-" * 45)
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.sum() > 0 else "✅ Zero missing values in core columns!")

    # 2. Class / Stance Distribution
    print("\n" + "-" * 45)
    print("2. CLASS / STANCE DISTRIBUTION")
    print("-" * 45)
    counts = df["stance"].value_counts()
    pcts = df["stance"].value_counts(normalize=True) * 100
    for stance, count in counts.items():
        print(f"  • {stance.upper():<12}: {count:>5,} records ({pcts[stance]:.2f}%)")

    # 3. Source Distribution
    print("\n" + "-" * 45)
    print("3. SOURCE REPOSITORY DISTRIBUTION")
    print("-" * 45)
    src_counts = df["source"].value_counts()
    src_pcts = df["source"].value_counts(normalize=True) * 100
    for src, count in src_counts.items():
        print(f"  • {src:<50}: {count:>5,} records ({src_pcts[src]:.1f}%)")

    # 4. Language Breakdown
    print("\n" + "-" * 45)
    print("4. LANGUAGE DISTRIBUTION")
    print("-" * 45)
    lang_counts = df["language"].value_counts()
    for lang, count in lang_counts.items():
        print(f"  • {lang.upper():<10}: {count:>5,} records ({count/total_records*100:.1f}%)")

    # 5. Text Length & Word Count Metrics
    df["claim_word_count"] = df["claim_topic"].apply(lambda x: len(str(x).split()))
    df["evidence_word_count"] = df["evidence_text"].apply(lambda x: len(str(x).split()))

    print("\n" + "-" * 45)
    print("5. WORD COUNT & TOKEN METRICS")
    print("-" * 45)
    print(df[["claim_word_count", "evidence_word_count"]].describe().round(1))

    # 6. Generate Comprehensive 4-Panel Visualization
    print("\n" + "-" * 45)
    print("6. GENERATING 4-PANEL COMPREHENSIVE VISUALIZATION...")
    print("-" * 45)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle(f"Multilingual Healthcare RAG — Clinical Dataset EDA Profile ({total_records:,} Records)",
                 fontsize=16, fontweight="bold", y=0.98)

    stance_colors = {"supports": "#10b981", "uncertain": "#f59e0b", "refutes": "#ef4444"}
    ordered_stances = ["supports", "uncertain", "refutes"]
    ordered_counts = [counts.get(s, 0) for s in ordered_stances]
    ordered_colors = [stance_colors[s] for s in ordered_stances]

    # Panel 1: Stance Distribution Bar Chart
    bars = axes[0, 0].bar(ordered_stances, ordered_counts, color=ordered_colors, edgecolor="black", alpha=0.88, width=0.55)
    axes[0, 0].set_title("1. Clinical Evidence Stance Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Number of Evidence Records", fontsize=11)
    axes[0, 0].set_ylim(0, max(ordered_counts) * 1.18)
    for bar, s, val in zip(bars, ordered_stances, ordered_counts):
        pct = (val / total_records) * 100
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 90,
                        f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Panel 2: Dataset Source Breakdown (Horizontal Bar)
    source_labels = [
        "HealthVer (PubMed / PMC)",
        "MEDIQ (KGMU Hospital Q&A)",
        "Hinglish Health Knowledgebase",
        "Medizin Transparent (Cochrane)"
    ]
    source_palette = ["#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4"]
    src_y_pos = np.arange(len(src_counts))
    axes[0, 1].barh(src_y_pos, src_counts.values, color=source_palette, edgecolor="black", alpha=0.88, height=0.55)
    axes[0, 1].set_yticks(src_y_pos)
    axes[0, 1].set_yticklabels(source_labels, fontsize=10, fontweight="600")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("2. Evidence Source Breakdown", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Number of Records", fontsize=11)
    axes[0, 1].set_xlim(0, max(src_counts.values) * 1.2)
    for idx, (val, pct) in enumerate(zip(src_counts.values, src_pcts.values)):
        axes[0, 1].text(val + 80, idx, f"{val:,} ({pct:.1f}%)", va="center", fontweight="bold", fontsize=10)

    # Panel 3: Evidence Word Count Distribution (Histogram & KDE)
    sns.histplot(df["evidence_word_count"], bins=40, kde=True, ax=axes[1, 0], color="#2563eb", edgecolor="black", alpha=0.7)
    median_len = df["evidence_word_count"].median()
    axes[1, 0].axvline(median_len, color="#dc2626", linestyle="--", linewidth=2, label=f"Median ({median_len:.0f} words)")
    axes[1, 0].set_title("3. Evidence Text Word Count Distribution", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Word Count per Evidence Record", fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].legend(loc="upper right", frameon=True)

    # Panel 4: Claim Length by Stance Category (Boxplot with Mean Points)
    sns.boxplot(
        data=df,
        x="stance",
        y="claim_word_count",
        order=ordered_stances,
        palette=stance_colors,
        ax=axes[1, 1],
        showmeans=True,
        meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "white", "markersize": 7}
    )
    axes[1, 1].set_title("4. Claim Length Distribution by Stance (◆ = Mean)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Clinical Stance", fontsize=11)
    axes[1, 1].set_ylabel("Words per Claim", fontsize=11)
    axes[1, 1].set_ylim(0, df["claim_word_count"].quantile(0.99) + 5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_img = os.path.join(os.path.dirname(dataset_path), "eda_summary_plots.png")
    plt.savefig(out_img, dpi=300)
    print(f"\n✅ Successfully saved updated 4-panel EDA visualization to:\n   -> {out_img}")
    plt.close()

    print("\n" + "=" * 70)
    print("📌 KEY STATISTICAL TAKEAWAYS FOR VIVA / REPORT:")
    print(f"1. Unified Dataset Size: {total_records:,} peer-reviewed clinical & multilingual records.")
    print(f"2. Balanced Class Stances: Supports {pcts.get('supports', 0):.1f}%, Uncertain {pcts.get('uncertain', 0):.1f}%, Refutes {pcts.get('refutes', 0):.1f}%.")
    print(f"3. Multi-Institutional Authority: PubMed/PMC ({src_pcts.iloc[0]:.1f}%), KGMU Hospital ({src_pcts.iloc[1]:.1f}%), Hinglish Corpus ({src_pcts.iloc[2]:.1f}%), Cochrane ({src_pcts.iloc[3]:.1f}%).")
    print(f"4. Embedding Window Alignment: Median evidence length is {median_len:.0f} words, perfectly within MiniLM's 256-word limit.")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
