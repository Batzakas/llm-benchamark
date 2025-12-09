from collections import defaultdict
import json
import re
import pandas as pd
import evaluate
import sys
import numpy as np
import scipy.stats as st
import os

# --- Configuration ---
SINGLE_RUN_INPUT_PATH = "results_with_answers.jsonl"
SINGLE_RUN_REPORT_PATH = "quality_report.json"
BATCH_RUN_INPUT_PATH = "raw_results.jsonl"
BATCH_REPORT_PATH = "aggregated_report.json" 
CONFIDENCE_LEVEL = 0.95

# Defines the minimum score to be considered a "pass"
METRIC_THRESHOLDS = {
    "rouge1": 0.5,
    "rouge2": 0.4,
    "rougeL": 0.5,
    "rougeLsum": 0.5,
    "exact_match": 0.7,  
    "extraction_f1": 0.4, 
    "unknown_metric": 0.0
}

try:
    rouge_metric_loader = evaluate.load('rouge')
    exact_match_metric_loader = evaluate.load('exact_match')
except Exception as e:
    print(f"Warning: Could not load Hugging Face metrics. {e}")
    rouge_metric_loader = None
    exact_match_metric_loader = None

# ======================================================================
# --- CORE METRIC FUNCTIONS 
# ======================================================================

def calculate_rouge_scores(prediction, reference, rouge_metric_loader):
    try:
        results = rouge_metric_loader.compute(predictions=[str(prediction)], references=[str(reference)])
        return [
            ('rouge1', results.get('rouge1', 0.0)),
            ('rouge2', results.get('rouge2', 0.0)),
            ('rougeL', results.get('rougeL', 0.0)),
            ('rougeLsum', results.get('rougeLsum', 0.0))
        ]
    except Exception:
        return [('rouge1', 0.0), ('rouge2', 0.0), ('rougeL', 0.0), ('rougeLsum', 0.0)]

def calculate_exact_match(prediction, reference):
    try:
        def normalize(text):
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\b(a|an|the)\b', ' ', text)
            return ' '.join(text.split())
        
        norm_pred = normalize(prediction)
        norm_ref = normalize(reference)
        results = exact_match_metric_loader.compute(predictions=[norm_pred], references=[norm_ref])
        return [('exact_match', results.get('exact_match', 0.0))]
    except Exception:
        return [('exact_match', 0.0)]

def calculate_extraction_f1(pred_str, ref_dict):
    try:
        if not isinstance(ref_dict, dict): ref_dict = {}
        try:
            pred_dict = json.loads(pred_str)
            if not isinstance(pred_dict, dict): pred_dict = {}
        except (json.JSONDecodeError, TypeError):
            pred_dict = {} 

        tp, fp, fn = 0, 0, 0
        all_fields = set(ref_dict.keys())

        if not all_fields:
             f1 = 1.0 if not pred_dict else 0.0
             return [('extraction_f1', f1)]

        for field in all_fields:
            ref_value = ref_dict.get(field)
            pred_value = pred_dict.get(field)
            
            ref_str = str(ref_value).lower() if ref_value is not None else None
            pred_str = str(pred_value).lower() if pred_value is not None else None

            if ref_str is not None and ref_str == pred_str:
                tp += 1
            elif pred_str is not None and ref_str != pred_str:
                fp += 1
            elif ref_str is not None and pred_str is None:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if tp == 0 and fp == 0 and fn == 0:
             f1 = 1.0 if not all_fields else 0.0
             
        return [('extraction_f1', f1)]
    except Exception:
        return [('extraction_f1', 0.0)]

def get_metric_for_task(task, prediction, reference, rouge_metric_loader):
    if task == 'summary' or task == 'short_qa':
        return calculate_rouge_scores(prediction, reference, rouge_metric_loader)
    elif task == 'factual_qa' or task == 'code_qa':
        return calculate_exact_match(prediction, reference)
    elif task == 'extraction':
        return calculate_extraction_f1(prediction, reference)
    return [('unknown_metric', 0.0)]

# ======================================================================
# --- PIPELINE 1: SINGLE-PASS REPORT 
# ======================================================================

def run_single_pass_report(rouge_metric_loader, input_path=None):
    
    target_input = input_path if input_path else SINGLE_RUN_INPUT_PATH
    print(f"Starting consolidated evaluation process for {target_input}...")
    
    try:
        with open(target_input, 'r', encoding='utf-8') as f:
            full_results = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Input file not found at {target_input}.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse {target_input}.")
        return
        
    results_by_config = defaultdict(list)
    
    for item in full_results:
        config_keys = {}
        # Capture Block, Replica, and Factors
        if 'block' in item: config_keys['block'] = item['block']
        if 'replica' in item: config_keys['replica'] = item['replica']
        if 'replica_id' in item: config_keys['replica'] = item['replica_id']

        for k, v in item.items():
            if k.startswith('factor_'):
                config_keys[k] = v
        
        # Fallback if no specific config
        if not config_keys:
             config_keys['model_name'] = item.get('model_name', 'unknown')

        # Use sorted tuple for consistent dictionary key
        config_tuple = tuple(sorted(config_keys.items()))
        results_by_config[config_tuple].append(item)

    final_report_list = []

    for config_tuple, results in results_by_config.items():
        config_dict = dict(config_tuple)
        
        # Calculate metrics (Legacy logic preserved)
        model_report = {}
        tasks = defaultdict(list)
        for item in results:
            tasks[item.get("task")].append(item)

        for task_name, items in tasks.items():
            if not items: continue
            
            scores_by_metric = defaultdict(list)
            pass_counts_by_metric = defaultdict(int)
            task_total_count = len(items)
            
            for item in items:
                metric_tuples = get_metric_for_task(
                    item.get('task'), 
                    item.get('model_answer'), 
                    item.get('reference_answer'),
                    rouge_metric_loader
                )
                
                for metric_name, score in metric_tuples:
                    scores_by_metric[metric_name].append(score)
                    threshold = METRIC_THRESHOLDS.get(metric_name, 0.0)
                    if score >= threshold:
                        pass_counts_by_metric[metric_name] += 1
            
            task_report = {}
            for metric_name, scores in scores_by_metric.items():
                if scores:
                    mean_score = np.mean(scores)
                    pass_count = pass_counts_by_metric[metric_name]
                    pass_percentage = (pass_count / task_total_count) * 100
                    
                    task_report[metric_name] = {
                        "mean_score": mean_score,
                        "pass_rate_percent": pass_percentage,
                        "pass_threshold": METRIC_THRESHOLDS.get(metric_name, 0.0)
                    }
            
            task_report["total_prompts"] = task_total_count
            model_report[task_name] = task_report

        # Add to list structure instead of dict-by-string-key
        final_report_list.append({
            "configuration": config_dict,
            "metrics": model_report
        })

    with open(SINGLE_RUN_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_report_list, f, indent=4)

    print(f"\nEvaluation complete. Report saved as list to {SINGLE_RUN_REPORT_PATH}")


# ======================================================================
# --- PIPELINE 2: BATCH STATISTICAL REPORT 
# ======================================================================

def run_batch_statistical_report():
    print(f"Starting statistical evaluation for {BATCH_RUN_INPUT_PATH}...")
    flat_run_data = []
    try:
        with open(BATCH_RUN_INPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                prompt_data = json.loads(line)
                prompt_id = prompt_data['prompt_id']
                task = prompt_data['task']
                reference_answer = prompt_data['reference_answer']
                for model_name, model_data in prompt_data['models'].items():
                    for answer in model_data['model_answers']:
                        flat_run_data.append({
                            "prompt_id": prompt_id, "task": task,
                            "model_name": model_name, "model_answer": answer,
                            "reference_answer": reference_answer
                        })
    except Exception as e:
        print(f"Error loading batch data: {e}")
        return

    if not flat_run_data: return
    df = pd.DataFrame(flat_run_data)
    
    # Note: Passed rouge_metric_loader to fix potential scope issue in original code
    df['metric_tuples'] = df.apply(
        lambda row: get_metric_for_task(
            row.get('task'), row.get('model_answer'), row.get('reference_answer'), rouge_metric_loader
        ), axis=1
    )
    
    df = df.explode('metric_tuples')
    df[['metric_name', 'metric_score']] = pd.DataFrame(df['metric_tuples'].tolist(), index=df.index)
    df = df.drop(columns=['metric_tuples']) 

    grouped = df.groupby(['prompt_id', 'model_name', 'metric_name'])['metric_score']
    agg_df = grouped.agg(mean='mean', min='min', max='max').reset_index()

    final_report = {}
    for index, row in agg_df.iterrows():
        pid = row['prompt_id']
        mn = row['model_name']
        met = row['metric_name']
        if pid not in final_report: final_report[pid] = {}
        if mn not in final_report[pid]: final_report[pid][mn] = {}
        final_report[pid][mn][met] = {"mean": row['mean'], "min": row['min'], "max": row['max']}

    with open(BATCH_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4)
    print(f"Batch statistical report saved to {BATCH_REPORT_PATH}")

# ======================================================================
# --- MAIN EXECUTION ---
# ======================================================================

if __name__ == "__main__":
    if "--batch" in sys.argv:
        run_batch_statistical_report()
    else:
        input_file = None
        for arg in sys.argv[1:]:
            if not arg.startswith("-") and os.path.exists(arg):
                input_file = arg
                break
        
        run_single_pass_report(rouge_metric_loader, input_path=input_file)