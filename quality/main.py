import requests
import json
import sys
from collections import defaultdict

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
INPUT_DATASET_PATH = "prompts_dataset.jsonl"
TIMEOUT = 240

MODELS_TO_EVALUATE = [
    "llama3.2:1b"
]

MODEL_CONFIG = {
    "stream": False,
    "options": {
        "temperature": 0.5,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_ctx": 4096 
    }
}

# --- Pipeline 1: Single Run ---
SINGLE_RUN_OUTPUT_PATH = "results_with_answers.jsonl"

# --- Pipeline 2: Batch Statistical Run ---
N_RUNS_PER_PROMPT = 5  
BATCH_RUN_OUTPUT_PATH = "raw_results.jsonl" 


def get_model_response(prompt, model_name, current_options=None):
    """Sends a prompt to the Ollama API and gets a response."""
    
    final_config = MODEL_CONFIG.copy()
    if current_options:
        final_config["options"].update(current_options)
        
    payload = {
        "model": model_name,
        "prompt": prompt,
        **final_config
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status() 
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

# ======================================================================
# --- PIPELINE 1: SINGLE-PASS 
# ======================================================================

def run_single_pass_evaluation(run_config=None, replica_id=None, output_path=None):
    print("Starting single-pass inference process...")
    
    try:
        with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f_in:
            dataset = [json.loads(line) for line in f_in]
    except FileNotFoundError:
        print(f"Error: Input dataset file not found at {INPUT_DATASET_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse {INPUT_DATASET_PATH}. Ensure it is a valid JSONL file.")
        return

    target_path = output_path if output_path else SINGLE_RUN_OUTPUT_PATH
    
    # Determine models and context from config if available
    models = [run_config["Model"]] if run_config and "Model" in run_config else MODELS_TO_EVALUATE
    options_override = {}
    if run_config and "Context_Int" in run_config:
        options_override["num_ctx"] = run_config["Context_Int"]

    # If run_config exists, append mode. Else write mode.
    file_mode = 'a' if run_config else 'w'
    if file_mode == 'w': open(target_path, 'w').close()

    for model_name in models:
        print(f"\n--- Processing model (single run): {model_name} ---")
        with open(target_path, 'a', encoding='utf-8') as f_out:
            for item in dataset:
                prompt = item.get("prompt")
                if not prompt:
                    continue

                print(f"  Running prompt: {item.get('id', 'N/A')}...")
                model_answer = get_model_response(prompt, model_name, options_override)
                
                if model_answer is not None:
                    result_item = item.copy()
                    result_item['model_answer'] = model_answer
                    result_item['model_name'] = model_name
                    
                    if run_config:
                        result_item['replica'] = replica_id
                        result_item['block'] = run_config.get('block_id')
                        for k, v in run_config.items():
                            if k not in ['block_id', 'Context_Int']:
                                result_item[f'factor_{k}'] = v

                    f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')

    print(f"\nSingle-pass inference complete. Results saved to {target_path}")

# ======================================================================
# --- PIPELINE 2: BATCH STATISTICAL 
# ======================================================================

def run_batch_statistical_evaluation():

    print(f"Starting batch statistical inference process ({N_RUNS_PER_PROMPT} runs per prompt)...")
    
    try:
        with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f_in:
            dataset = [json.loads(line) for line in f_in]
    except FileNotFoundError:
        print(f"Error: Input dataset file not found at {INPUT_DATASET_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse {INPUT_DATASET_PATH}. Ensure it is a valid JSONL file.")
        return

    batch_results = {}
    for item in dataset:
        prompt_id = item['id']
        batch_results[prompt_id] = {
            "task": item['task'],
            "reference_answer": item['reference_answer'],
            "models": {}
        }
        for model_name in MODELS_TO_EVALUATE:
            batch_results[prompt_id]["models"][model_name] = {
                "model_answers": []
            }
    
    total_runs = len(MODELS_TO_EVALUATE) * len(dataset) * N_RUNS_PER_PROMPT
    print(f"Total runs to execute: {total_runs}")
    current_run = 0

    for model_name in MODELS_TO_EVALUATE:
        print(f"\n--- Processing model (batch run): {model_name} ---")
        for item in dataset:
            prompt_id = item['id']
            prompt = item['prompt']
            print(f"  Running prompt: {prompt_id} ({N_RUNS_PER_PROMPT} times)")
            
            for i in range(N_RUNS_PER_PROMPT):
                current_run += 1
                print(f"    Run {i+1}/{N_RUNS_PER_PROMPT} (Total {current_run}/{total_runs})")
                
                model_answer = get_model_response(prompt, model_name)
                
                if model_answer is not None:
                    batch_results[prompt_id]["models"][model_name]["model_answers"].append(model_answer)

    print(f"\nBatch inference complete. Writing aggregated results to {BATCH_RUN_OUTPUT_PATH}")
    with open(BATCH_RUN_OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        for prompt_id, data in batch_results.items():
            line_data = {"prompt_id": prompt_id, **data}
            f_out.write(json.dumps(line_data, ensure_ascii=False) + '\n')

    print(f"Batch results saved to {BATCH_RUN_OUTPUT_PATH}")


if __name__ == "__main__":
    if "--batch" in sys.argv:
        run_batch_statistical_evaluation()
    else:
        run_single_pass_evaluation()