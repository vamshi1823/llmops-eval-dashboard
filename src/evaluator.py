import json
import time
from typing import Dict, List

from openai import OpenAI

from metrics import evaluate_response

client = OpenAI(api_key="your_openai_api_key_here")  # You'll replace this

def load_prompts(filepath: str) -> Dict:
    """Load prompt versions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_golden_dataset(filepath: str) -> List[Dict]:
    """Load test questions from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['test_cases']

def run_evaluation(prompt_id: str, prompt_text: str, question: str) -> Dict:
    """
    Send question to OpenAI with specific prompt version.
    Record response, latency, and evaluate it.
    """
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        generated_answer = response.choices[0].message.content
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "answer": generated_answer,
            "latency_ms": latency_ms
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "latency_ms": (time.time() - start_time) * 1000
        }

def evaluate_all_prompts(
    prompts: Dict,
    test_cases: List[Dict],
    results_file: str = "results/eval_results.json"
) -> Dict:
    """
    Test all prompt versions against all test cases.
    Save results to JSON file.
    """
    
    all_results = {
        "timestamp": time.time(),
        "prompts_tested": [],
        "results": []
    }
    
    prompt_list = prompts['prompts']
    
    for prompt_obj in prompt_list:
        prompt_id = prompt_obj['id']
        prompt_name = prompt_obj['name']
        prompt_text = prompt_obj['system_prompt']
        
        all_results["prompts_tested"].append({
            "id": prompt_id,
            "name": prompt_name
        })
        
        print(f"\n🧪 Testing {prompt_name} ({prompt_id})...")
        
        for test_case in test_cases:
            question = test_case['question']
            expected_answer = test_case['expected_answer']
            
            # Get response from OpenAI
            response = run_evaluation(prompt_id, prompt_text, question)
            
            if response['status'] == 'success':
                # Evaluate the response
                metrics = evaluate_response(
                    question=question,
                    generated_answer=response['answer'],
                    expected_answer=expected_answer,
                    latency_ms=response['latency_ms']
                )
                
                result_entry = {
                    "prompt_id": prompt_id,
                    "prompt_name": prompt_name,
                    "question_id": test_case['id'],
                    "question": question,
                    "generated_answer": response['answer'],
                    "expected_answer": expected_answer,
                    "metrics": metrics
                }
            else:
                result_entry = {
                    "prompt_id": prompt_id,
                    "prompt_name": prompt_name,
                    "question_id": test_case['id'],
                    "question": question,
                    "error": response['error']
                }
            
            all_results["results"].append(result_entry)
            print(f"  ✓ Question {test_case['id']} evaluated")
    
    # Save results to file
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Evaluation complete! Results saved to {results_file}")
    
    return all_results

def get_prompt_comparison(results: Dict) -> Dict:
    """
    Summarize which prompt performed best overall.
    """
    prompt_scores = {}
    
    for result in results['results']:
        if 'metrics' in result:
            prompt_id = result['prompt_id']
            if prompt_id not in prompt_scores:
                prompt_scores[prompt_id] = {
                    "name": result['prompt_name'],
                    "scores": [],
                    "latencies": []
                }
            
            prompt_scores[prompt_id]['scores'].append(result['metrics']['overall_score'])
            prompt_scores[prompt_id]['latencies'].append(result['metrics']['latency_ms'])
    
    # Calculate averages
    comparison = {}
    for prompt_id, data in prompt_scores.items():
        avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
        avg_latency = sum(data['latencies']) / len(data['latencies']) if data['latencies'] else 0
        
        comparison[prompt_id] = {
            "name": data['name'],
            "avg_overall_score": round(avg_score, 3),
            "avg_latency_ms": round(avg_latency, 2),
            "num_tests": len(data['scores'])
        }
    
    return comparison