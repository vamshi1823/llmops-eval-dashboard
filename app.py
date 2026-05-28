import json
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluator import (evaluate_all_prompts, get_prompt_comparison,
                       load_golden_dataset, load_prompts)

st.set_page_config(page_title="LLMOps Evaluation Dashboard", layout="wide")

st.title("🧪 LLMOps Evaluation & Monitoring Dashboard")
st.caption("Compare prompt versions across faithfulness, hallucination rate, and latency")

# Sidebar for OpenAI API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
    if api_key:
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✓ API key set")
    
    st.divider()
    st.subheader("📊 Dashboard Info")
    st.info(
        "This dashboard evaluates multiple prompt versions against a golden dataset of test questions. "
        "Metrics include faithfulness, hallucination rate, answer relevancy, and latency."
    )

# Check if results exist
results_path = Path("results/eval_results.json")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Run Evaluation")
    if st.button("Start Evaluation"):
        if not api_key:
            st.error("❌ Please enter your OpenAI API key in the sidebar")
        else:
            try:
                with st.spinner("Loading prompts and test cases..."):
                    prompts = load_prompts("prompts/prompt_registry.json")
                    test_cases = load_golden_dataset("data/golden_dataset.json")
                
                with st.spinner("Running evaluation... this may take a minute"):
                    results = evaluate_all_prompts(prompts, test_cases)
                
                st.success("✅ Evaluation complete!")
            except FileNotFoundError as e:
                st.error(f"❌ File not found: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

with col2:
    st.subheader("📈 Quick Stats")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        num_prompts = len(results['prompts_tested'])
        num_tests = len([r for r in results['results'] if 'metrics' in r])
        
        st.metric("Prompts Tested", num_prompts)
        st.metric("Total Tests Run", num_tests)

st.divider()

# Show results if they exist
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    
    # Get comparison summary
    comparison = get_prompt_comparison(results)
    
    st.subheader("📊 Prompt Version Comparison")
    
    # Create comparison table
    comparison_data = []
    for prompt_id, stats in comparison.items():
        comparison_data.append({
            "Prompt": stats['name'],
            "Avg Score": stats['avg_overall_score'],
            "Avg Latency (ms)": stats['avg_latency_ms'],
            "Tests Run": stats['num_tests']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    st.divider()
    
    # Detailed results by prompt
    st.subheader("🔍 Detailed Results")
    
    tabs = st.tabs([p['name'] for p in results['prompts_tested']])
    
    for tab, prompt_info in zip(tabs, results['prompts_tested']):
        with tab:
            prompt_id = prompt_info['id']
            prompt_results = [r for r in results['results'] if r.get('prompt_id') == prompt_id]
            
            st.write(f"**Prompt ID:** {prompt_id}")
            
            # Create detailed results dataframe
            detailed_data = []
            for result in prompt_results:
                if 'metrics' in result:
                    detailed_data.append({
                        "Question": result['question'][:50] + "...",
                        "Faithfulness": round(result['metrics']['faithfulness'], 3),
                        "Hallucination Rate": round(result['metrics']['hallucination_rate'], 3),
                        "Relevancy": round(result['metrics']['answer_relevancy'], 3),
                        "Overall Score": round(result['metrics']['overall_score'], 3),
                        "Latency (ms)": round(result['metrics']['latency_ms'], 2)
                    })
                else:
                    detailed_data.append({
                        "Question": result['question'][:50] + "...",
                        "Error": result.get('error', 'Unknown error')
                    })
            
            if detailed_data:
                df_detailed = pd.DataFrame(detailed_data)
                st.dataframe(df_detailed, use_container_width=True)
            
            # Show sample answer
            if prompt_results and 'generated_answer' in prompt_results[0]:
                with st.expander("📝 Sample Answer"):
                    st.write(f"**Question:** {prompt_results[0]['question']}")
                    st.write(f"**Generated Answer:** {prompt_results[0]['generated_answer']}")
                    st.write(f"**Expected Answer:** {prompt_results[0]['expected_answer']}")
    
    st.divider()
    
    st.subheader("📥 Raw Results JSON")
    if st.checkbox("Show raw results JSON"):
        st.json(results)

else:
    st.info("👆 Click 'Start Evaluation' above to run your first test")