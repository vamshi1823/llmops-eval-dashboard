# LLMOps Evaluation & Monitoring Dashboard

> Benchmark multiple LLM prompt versions across faithfulness, hallucination rate, answer relevancy, and latency.

## Overview

This project implements an end-to-end LLM evaluation framework that enables data-driven prompt optimization. It tests multiple prompt versions against a golden dataset of test questions and visualizes performance metrics in an interactive Streamlit dashboard.

**Key Features:**
- 📊 Compare 3+ prompt versions side-by-side
- 🧪 Automated evaluation against golden test cases
- 📈 Track faithfulness, hallucination rate, answer relevancy, and latency
- 🎨 Interactive Streamlit dashboard for visualization
- 💾 Persistent results stored as JSON for versioning

## Tech Stack

- **Python** — core language
- **OpenAI API** — LLM inference (gpt-3.5-turbo)
- **Streamlit** — interactive dashboard
- **Pandas** — data analysis
- **JSON** — prompt versioning and results storage

## Project Structure