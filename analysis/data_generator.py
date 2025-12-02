import pandas as pd
import numpy as np
import itertools
from config import ENVIRONMENT_FACTORS, DESIGN_SPECS

def create_mock_csv(output_filename: str = "mock_data.csv"):
    all_runs_list = []
    np.random.seed(42)

    for block_name, spec in DESIGN_SPECS.items():
        factors_config = ENVIRONMENT_FACTORS[block_name]
        factor_names = list(factors_config.keys())
        levels_list = list(factors_config.values())
        
        all_combinations = list(itertools.product(*levels_list))
        full_df = pd.DataFrame(all_combinations, columns=factor_names)

        if spec["type"] == "taguchi":
            n_runs = min(spec["base_runs"], len(full_df))
            base_df = full_df.sample(n=n_runs, random_state=42).reset_index(drop=True)
        else:
            base_df = full_df.reset_index(drop=True)

        for r in range(1, spec["replicas"] + 1):
            replica_df = base_df.copy()
            replica_df["Block"] = block_name
            replica_df["Replica"] = r
            
            n = len(replica_df)
            
            noise = np.random.normal(0, 5, n)
            
            # Fake effects: 
            # 1. High Concurrency -> High Latency (p95)
            # 2. int4 Quant -> Higher Throughput
            
            conc_effect = replica_df["Conc"].astype(float) * 2
            quant_effect = np.where(replica_df["Quant"] == "int4", 20, 0)
            
            replica_df["p95_ms"] = 100 + conc_effect + noise + (r * 2)
            replica_df["Throughput"] = 50 + quant_effect - (conc_effect / 5) + noise
            replica_df["ROUGE_L"] = np.random.uniform(0.4, 0.9, n)
            replica_df["F1_Extracao"] = np.random.uniform(0.5, 0.95, n)
            replica_df["Wh_100_prompts"] = 30 + (conc_effect / 2) + noise
            
            # Fill missing columns for B3-specific Hardware if not in B1/B2
            if "Hardware" not in replica_df.columns:
                replica_df["Hardware"] = "N/A"

            all_runs_list.append(replica_df)

    final_df = pd.concat(all_runs_list, ignore_index=True)
    final_df["Run_ID"] = range(1, len(final_df) + 1)
    
    cols = ["Run_ID", "Block", "Replica", "Model", "Quant", "Conc", "Context", "Hardware"]
    metric_cols = [c for c in final_df.columns if c not in cols]
    final_df = final_df[cols + metric_cols]

    final_df.to_csv(output_filename, index=False)
    print(f"Created {output_filename} with {len(final_df)} rows.")

if __name__ == "__main__":
    output_filename = "mock_data.csv"  
    create_mock_csv(output_filename)