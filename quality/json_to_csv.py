import json
import pandas as pd
import sys
import os

# Configuration
INPUT_JSON = "quality_report.json"
OUTPUT_CSV = "results_summary.csv"

FIXED_COLUMNS = ['Block', 'Replica', 'Model', 'Quant', 'Conc', 'Context', 'Hardware']

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return

    if not isinstance(data, list):
        print("Error: JSON format must be a list of objects.")
        return

    rows = []
    
    for entry in data:
        config = entry.get('configuration', {})
        metrics_data = entry.get('metrics', {})
        
        row = {}
        
        # Extract Factors (Handling 'factor_' prefix)
        for col in FIXED_COLUMNS:
            if col in config:
                row[col] = config[col]
            elif f"factor_{col}" in config:
                row[col] = config[f"factor_{col}"]
            elif col.lower() in config:
                row[col] = config[col.lower()]
            else:
                row[col] = None
        
        for task, task_metrics in metrics_data.items():
            for metric_name, stats in task_metrics.items():
                if isinstance(stats, dict) and 'mean_score' in stats:
                    row[metric_name] = stats['mean_score']

        rows.append(row)

    if not rows:
        print("No data found to process.")
        return

    new_df = pd.DataFrame(rows)

    # Append or Create
    if os.path.exists(OUTPUT_CSV):
        print(f"Appending to existing {OUTPUT_CSV}...")
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            
            if 'Run_ID' in existing_df.columns and 'Run_ID' not in new_df.columns:
                max_id = existing_df['Run_ID'].max()
                if pd.isna(max_id): max_id = 0
                new_df.insert(0, 'Run_ID', range(int(max_id) + 1, int(max_id) + 1 + len(new_df)))
            
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(OUTPUT_CSV, index=False)
            print(f"Successfully appended {len(new_df)} rows. Total rows: {len(combined_df)}")
            
        except Exception as e:
            print(f"Error appending to CSV: {e}")
    else:
        print(f"Creating new {OUTPUT_CSV}...")
        if 'Run_ID' not in new_df.columns:
            new_df.insert(0, 'Run_ID', range(1, 1 + len(new_df)))
            
        new_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Successfully created {OUTPUT_CSV} with {len(new_df)} rows.")

if __name__ == "__main__":
    main()