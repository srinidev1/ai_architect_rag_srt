import streamlit as st
import pandas as pd
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

from evaluation.evaltest import evaluate_all_retrieval, evaluate_all_answers

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# ── Thresholds ─────────────────────────────────────────────────────────────────
MRR_GREEN      = 0.8;  MRR_AMBER      = 0.60
NDCG_GREEN     = 0.8;  NDCG_AMBER     = 0.60
COVERAGE_GREEN = 80.0; COVERAGE_AMBER = 60.0
ANSWER_GREEN   = 4.0;  ANSWER_AMBER   = 3.8


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_color(value: float, metric_type: str) -> str:
    thresholds = {
        "mrr":          (MRR_GREEN,      MRR_AMBER),
        "ndcg":         (NDCG_GREEN,     NDCG_AMBER),
        "coverage":     (COVERAGE_GREEN, COVERAGE_AMBER),
        "accuracy":     (ANSWER_GREEN,   ANSWER_AMBER),
        "completeness": (ANSWER_GREEN,   ANSWER_AMBER),
        "relevance":    (ANSWER_GREEN,   ANSWER_AMBER),
    }
    if metric_type not in thresholds:
        return "black"
    green, amber = thresholds[metric_type]
    if value >= green:
        return "#28a745"
    elif value >= amber:
        return "#fd7e14"
    return "#dc3545"


def metric_card(label: str, value: float, metric_type: str,
                is_percentage: bool = False, score_format: bool = False) -> str:
    color = get_color(value, metric_type)
    value_str = (f"{value:.1f}%" if is_percentage
                 else f"{value:.2f}/5" if score_format
                 else f"{value:.4f}")
    return f"""
    <div style="margin:8px 0;padding:15px;background:#f8f9fa;
                border-radius:8px;border-left:5px solid {color};">
        <div style="font-size:13px;color:#666;margin-bottom:4px;">{label}</div>
        <div style="font-size:26px;font-weight:700;color:{color};">{value_str}</div>
    </div>"""


def complete_banner(count: int) -> str:
    return f"""
    <div style="margin-top:16px;padding:10px;background:#d4edda;border-radius:5px;
                text-align:center;border:1px solid #c3e6cb;">
        <span style="font-size:14px;color:#155724;font-weight:bold;">
            ✓ Evaluation Complete — {count} tests
        </span>
    </div>"""


# ── Retrieval evaluation ───────────────────────────────────────────────────────
def run_retrieval_evaluation():
    total_mrr = total_ndcg = total_coverage = 0.0
    category_mrr: dict[str, list[float]] = defaultdict(list)
    count = 0

    progress_bar = st.progress(0, text="Starting retrieval evaluation…")
    metrics_slot = st.empty()

    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr      += result.mrr
        total_ndcg     += result.ndcg
        total_coverage += result.keyword_coverage
        category_mrr[test.category].append(result.mrr)
        progress_bar.progress(prog_value, text=f"Evaluating test {count}…")

    progress_bar.empty()

    avg_mrr      = total_mrr      / count
    avg_ndcg     = total_ndcg     / count
    avg_coverage = total_coverage / count

    html = (
        metric_card("Mean Reciprocal Rank (MRR)", avg_mrr,      "mrr")
      + metric_card("Normalized DCG (nDCG)",      avg_ndcg,     "ndcg")
      + metric_card("Keyword Coverage",           avg_coverage, "coverage", is_percentage=True)
      + complete_banner(count)
    )
    metrics_slot.markdown(html, unsafe_allow_html=True)

    # Bar chart
    df = pd.DataFrame([
        {"Category": cat, "Average MRR": sum(scores) / len(scores)}
        for cat, scores in category_mrr.items()
    ])
    st.bar_chart(df.set_index("Category"), y="Average MRR", y_label="Average MRR", x_label="Category")


# ── Answer evaluation ──────────────────────────────────────────────────────────
def run_answer_evaluation():
    total_accuracy = total_completeness = total_relevance = 0.0
    category_accuracy: dict[str, list[float]] = defaultdict(list)
    count = 0

    progress_bar = st.progress(0, text="Starting answer evaluation…")
    metrics_slot = st.empty()

    for test, result, prog_value in evaluate_all_answers():
        count += 1
        total_accuracy     += result.accuracy
        total_completeness += result.completeness
        total_relevance    += result.relevance
        category_accuracy[test.category].append(result.accuracy)
        progress_bar.progress(prog_value, text=f"Evaluating test {count}…")

    progress_bar.empty()

    avg_accuracy     = total_accuracy     / count
    avg_completeness = total_completeness / count
    avg_relevance    = total_relevance    / count

    html = (
        metric_card("Accuracy",     avg_accuracy,     "accuracy",     score_format=True)
      + metric_card("Completeness", avg_completeness, "completeness", score_format=True)
      + metric_card("Relevance",    avg_relevance,    "relevance",    score_format=True)
      + complete_banner(count)
    )
    metrics_slot.markdown(html, unsafe_allow_html=True)

    # Bar chart
    df = pd.DataFrame([
        {"Category": cat, "Average Accuracy": sum(scores) / len(scores)}
        for cat, scores in category_accuracy.items()
    ])
    st.bar_chart(df.set_index("Category"), y="Average Accuracy", y_label="Average Accuracy (1–5)", x_label="Category")


# ── Streamlit page entry point (called from main.py as evaluationview.render()) ──
def render():
    st.title("📊 RAG Evaluation Dashboard")

    # Guard: knowledge base must exist before we can chat
    if not Path(DB_NAME).exists():
        st.warning("No knowledge base found. Please run the Ingest process first.")
        return
    
    st.caption("Evaluate retrieval and answer quality for the Insurellm RAG system.")

    # ── Retrieval section ──────────────────────────────────────────────────────
    st.markdown("## 🔍 Retrieval Evaluation")
    if st.button("Run Retrieval Evaluation", key="retrieval_btn"):
        run_retrieval_evaluation()

    st.divider()

    # ── Answer section ─────────────────────────────────────────────────────────
    st.markdown("## 💬 Answer Evaluation")
    if st.button("Run Answer Evaluation",  key="answer_btn"):
        run_answer_evaluation()


#if __name__ == "__main__":
    #render()
