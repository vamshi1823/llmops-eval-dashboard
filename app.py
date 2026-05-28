import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="LLMOps Evaluation Dashboard", layout="wide", page_icon="🧪")

# ── Mock Data ────────────────────────────────────────────────────────────────

PROMPTS = [
    {"id": "v1", "name": "Prompt v1 – Basic",      "description": "Simple zero-shot prompt with no chain-of-thought"},
    {"id": "v2", "name": "Prompt v2 – CoT",         "description": "Chain-of-thought reasoning added"},
    {"id": "v3", "name": "Prompt v3 – Few-Shot",    "description": "3 few-shot examples + CoT"},
]

QUESTIONS = [
    {"id": "q1", "question": "What is retrieval-augmented generation (RAG)?",         "expected": "RAG combines retrieval of relevant documents with LLM generation to ground answers in factual context."},
    {"id": "q2", "question": "How do you evaluate LLM hallucination?",                "expected": "Hallucination is measured by comparing generated answers to ground-truth references using metrics like faithfulness and factual consistency."},
    {"id": "q3", "question": "What is prompt engineering?",                           "expected": "Prompt engineering is the practice of designing inputs to guide LLM outputs toward desired responses."},
    {"id": "q4", "question": "Explain the difference between fine-tuning and RAG.",   "expected": "Fine-tuning updates model weights with domain data; RAG retrieves external knowledge at inference time without changing weights."},
    {"id": "q5", "question": "What metrics matter most in LLMOps?",                  "expected": "Key metrics include faithfulness, answer relevancy, hallucination rate, latency, and cost per query."},
]

SAMPLE_ANSWERS = {
    "v1": [
        "RAG is a method that retrieves documents and uses them to generate answers.",
        "You can check hallucinations by comparing outputs to known facts.",
        "Prompt engineering means writing good prompts for AI models.",
        "Fine-tuning trains the model while RAG just looks up information.",
        "Latency and accuracy are important metrics.",
    ],
    "v2": [
        "Let me think step by step. RAG stands for Retrieval-Augmented Generation. It first retrieves relevant documents from a knowledge base, then uses an LLM to generate a grounded answer based on those documents.",
        "To evaluate hallucination, I'll reason through it: compare the generated answer to reference answers using faithfulness scores, then flag any claims not supported by retrieved context.",
        "Thinking carefully: prompt engineering is the systematic design of instructions given to an LLM to shape its behavior and output quality.",
        "Step by step: fine-tuning modifies model weights using training data, while RAG retrieves live external knowledge without touching the model weights.",
        "Key LLMOps metrics are: faithfulness (is the answer grounded?), relevancy (does it answer the question?), latency (how fast?), and cost.",
    ],
    "v3": [
        "Based on examples: RAG (Retrieval-Augmented Generation) retrieves semantically relevant documents from a vector store and feeds them as context to the LLM, grounding the response in real data.",
        "Following the pattern from examples: hallucination evaluation uses RAGAS-style metrics — faithfulness checks if each claim is supported by retrieved context, scored 0–1.",
        "As shown in the examples: prompt engineering is the disciplined craft of structuring LLM inputs — including system prompts, few-shot examples, and chain-of-thought instructions — to maximise output quality.",
        "Mirroring the example format: fine-tuning adapts model weights on domain-specific data (expensive, static); RAG retrieves fresh external knowledge at inference time (cheaper, dynamic, updatable).",
        "Drawing from examples: top LLMOps metrics are faithfulness, hallucination rate, answer relevancy, context precision, latency (p50/p95), and token cost per query.",
    ],
}

random.seed(42)
np.random.seed(42)

def generate_mock_results():
    results = {"prompts_tested": PROMPTS, "results": []}
    base = {"v1": (0.62, 0.28, 380), "v2": (0.78, 0.14, 520), "v3": (0.91, 0.06, 610)}
    for prompt in PROMPTS:
        pid = prompt["id"]
        b_faith, b_hall, b_lat = base[pid]
        for i, q in enumerate(QUESTIONS):
            faith  = min(1.0, max(0.0, b_faith + random.gauss(0, 0.05)))
            hall   = min(1.0, max(0.0, b_hall  + random.gauss(0, 0.03)))
            relev  = min(1.0, max(0.0, faith   + random.gauss(0.05, 0.04)))
            lat    = max(100, b_lat + random.gauss(0, 40))
            overall = round((faith * 0.4 + (1 - hall) * 0.3 + relev * 0.3), 3)
            results["results"].append({
                "prompt_id": pid, "prompt_name": prompt["name"],
                "question_id": q["id"], "question": q["question"],
                "generated_answer": SAMPLE_ANSWERS[pid][i],
                "expected_answer": q["expected"],
                "metrics": {
                    "faithfulness": round(faith, 3),
                    "hallucination_rate": round(hall, 3),
                    "answer_relevancy": round(relev, 3),
                    "overall_score": overall,
                    "latency_ms": round(lat, 1),
                },
            })
    return results

def get_comparison(results):
    summary = {}
    for r in results["results"]:
        pid = r["prompt_id"]
        if pid not in summary:
            summary[pid] = {"name": r["prompt_name"], "scores": [], "latencies": [], "hall": []}
        m = r["metrics"]
        summary[pid]["scores"].append(m["overall_score"])
        summary[pid]["latencies"].append(m["latency_ms"])
        summary[pid]["hall"].append(m["hallucination_rate"])
    out = {}
    for pid, d in summary.items():
        out[pid] = {
            "name": d["name"],
            "avg_overall_score": round(sum(d["scores"]) / len(d["scores"]), 3),
            "avg_latency_ms": round(sum(d["latencies"]) / len(d["latencies"]), 1),
            "avg_hallucination": round(sum(d["hall"]) / len(d["hall"]), 3),
            "num_tests": len(d["scores"]),
        }
    return out

# ── App ──────────────────────────────────────────────────────────────────────

results = generate_mock_results()
comparison = get_comparison(results)

with st.sidebar:
    st.image("https://img.shields.io/badge/LLMOps-Demo_Dashboard-1f4e79?style=for-the-badge")
    st.markdown("### 📌 About This Project")
    st.info(
        "This dashboard demonstrates LLMOps evaluation concepts: "
        "prompt versioning, hallucination tracking, faithfulness scoring, "
        "and latency monitoring across multiple prompt versions."
    )
    st.markdown("---")
    st.markdown("### 🔗 Project Links")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-View_Source-black?logo=github)](https://github.com/vamshi1823/llmops-eval-dashboard)")
    st.markdown("---")
    st.markdown("**Stack:** Python · Streamlit · Plotly · Pandas")
    st.markdown("**Concepts:** RAG · RAGAS metrics · Prompt versioning · LLMOps")
    st.caption("⚠️ Demo mode — mock data, no API key required")

st.title("🧪 LLMOps Evaluation & Monitoring Dashboard")
st.caption("Compare prompt versions across faithfulness, hallucination rate, answer relevancy, and latency")
st.markdown("---")

best_pid = max(comparison, key=lambda x: comparison[x]["avg_overall_score"])
best = comparison[best_pid]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Prompts Tested", len(PROMPTS))
k2.metric("Test Questions", len(QUESTIONS))
k3.metric("Best Overall Score", best["avg_overall_score"], f"→ {best['name']}")
k4.metric("Lowest Hallucination", min(v["avg_hallucination"] for v in comparison.values()), "Prompt v3")

st.markdown("---")
st.subheader("📊 Prompt Version Comparison")

rows = []
for pid, d in comparison.items():
    rows.append({
        "Prompt Version": d["name"],
        "Avg Overall Score": d["avg_overall_score"],
        "Avg Faithfulness": round(sum(r["metrics"]["faithfulness"] for r in results["results"] if r["prompt_id"] == pid) / d["num_tests"], 3),
        "Hallucination Rate": d["avg_hallucination"],
        "Avg Latency (ms)": d["avg_latency_ms"],
        "Tests Run": d["num_tests"],
    })
df_cmp = pd.DataFrame(rows)
st.dataframe(df_cmp.style.highlight_max(subset=["Avg Overall Score", "Avg Faithfulness"], color="#c6efce")
                         .highlight_min(subset=["Hallucination Rate", "Avg Latency (ms)"], color="#c6efce"),
             use_container_width=True)

st.markdown("---")
st.subheader("📈 Visual Analysis")
c1, c2 = st.columns(2)

with c1:
    fig_bar = px.bar(
        df_cmp, x="Prompt Version", y="Avg Overall Score",
        color="Prompt Version", title="Overall Score by Prompt Version",
        color_discrete_sequence=["#4472C4", "#ED7D31", "#70AD47"],
        text="Avg Overall Score"
    )
    fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_bar.update_layout(showlegend=False, yaxis_range=[0, 1.1])
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    fig_scatter = px.scatter(
        df_cmp, x="Avg Latency (ms)", y="Avg Overall Score",
        size=[30, 30, 30], color="Prompt Version",
        title="Score vs Latency Trade-off",
        color_discrete_sequence=["#4472C4", "#ED7D31", "#70AD47"],
        text="Prompt Version"
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    metrics_df = pd.DataFrame([
        {"Metric": "Faithfulness",      "v1 – Basic": 0.621, "v2 – CoT": 0.779, "v3 – Few-Shot": 0.912},
        {"Metric": "Answer Relevancy",  "v1 – Basic": 0.648, "v2 – CoT": 0.801, "v3 – Few-Shot": 0.934},
        {"Metric": "1 – Hallucination", "v1 – Basic": 0.720, "v2 – CoT": 0.860, "v3 – Few-Shot": 0.940},
    ])
    fig_grouped = go.Figure()
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    for i, col in enumerate(["v1 – Basic", "v2 – CoT", "v3 – Few-Shot"]):
        fig_grouped.add_trace(go.Bar(name=col, x=metrics_df["Metric"], y=metrics_df[col], marker_color=colors[i]))
    fig_grouped.update_layout(barmode="group", title="Metric Breakdown by Prompt Version", yaxis_range=[0, 1.1])
    st.plotly_chart(fig_grouped, use_container_width=True)

with c4:
    hall_data = [{"Prompt": r["prompt_name"], "Hallucination Rate": r["metrics"]["hallucination_rate"]} for r in results["results"]]
    fig_box = px.box(pd.DataFrame(hall_data), x="Prompt", y="Hallucination Rate",
                     color="Prompt", title="Hallucination Rate Distribution",
                     color_discrete_sequence=["#4472C4", "#ED7D31", "#70AD47"])
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")
st.subheader("🔍 Detailed Results by Prompt Version")
tabs = st.tabs([p["name"] for p in PROMPTS])
for tab, prompt in zip(tabs, PROMPTS):
    with tab:
        pid = prompt["id"]
        prompt_results = [r for r in results["results"] if r["prompt_id"] == pid]
        st.markdown(f"**Description:** {prompt['description']}")
        detail_rows = []
        for r in prompt_results:
            m = r["metrics"]
            detail_rows.append({
                "Question": r["question"][:55] + "...",
                "Faithfulness": m["faithfulness"],
                "Hallucination": m["hallucination_rate"],
                "Relevancy": m["answer_relevancy"],
                "Overall": m["overall_score"],
                "Latency (ms)": m["latency_ms"],
            })
        df_detail = pd.DataFrame(detail_rows)
        st.dataframe(df_detail.style.highlight_max(subset=["Overall", "Faithfulness"], color="#c6efce")
                                    .highlight_min(subset=["Hallucination"], color="#c6efce"),
                     use_container_width=True)
        with st.expander("📝 View Sample Answer"):
            sample = prompt_results[0]
            st.markdown(f"**❓ Question:** {sample['question']}")
            st.markdown(f"**🤖 Generated:** {sample['generated_answer']}")
            st.markdown(f"**✅ Expected:** {sample['expected_answer']}")

st.markdown("---")
st.caption("Built by Vasam Vamshi · [GitHub](https://github.com/vamshi1823/llmops-eval-dashboard) · LLMOps Portfolio Project")