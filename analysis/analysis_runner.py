import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import warnings

from assumptions import check_assumptions

warnings.simplefilter('ignore')

def get_formula(block_name: str) -> str:
    # Standard formula
    base = "C(Model) + C(Quant) + C(Conc) + C(Context) + C(Replica)"
    
    #Critical interactions
    interactions = " + C(Model):C(Quant) + C(Model):C(Conc)"
    
    formula = f"{base}{interactions}"
    
    if block_name == "B3":
        formula = f"{formula} + C(Hardware)"
        
    return formula

def run_analysis(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    results = []

    factors_and_ids = ["Run_ID", "Block", "Replica", "Model", "Quant", "Conc", "Context", "Hardware"]
    metrics = [c for c in df.columns if c not in factors_and_ids]

    for block in df["Block"].unique():
        print(f"--- Analyzing Block: {block} ---")
        block_df = df[df["Block"] == block].copy()
        
        formula_rhs = get_formula(block)

        for metric in metrics:
            formula = f"{metric} ~ {formula_rhs}"
            
            try:
                model = ols(formula, data=block_df).fit()
                
                try:
                    check_assumptions(model, metric, block)
                except Exception as diag_e:
                    print(f"    Diagnostic check failed: {diag_e}")
                    
                anova_table = sm.stats.anova_lm(model, typ=3)
                
                for source, row in anova_table.iterrows():
                    if source == "Intercept":
                        continue
                        
                    results.append({
                        "Block": block,
                        "Metric": metric,
                        "Source": source,
                        "F_Stat": round(row['F'], 4),
                        "p_value": round(row['PR(>F)'], 6),
                        "Significance": "*" if row['PR(>F)'] < 0.05 else ""
                    })

            except Exception as e:
                print(f"    Error analyzing {metric}: {e}")
                results.append({
                    "Block": block,
                    "Metric": metric,
                    "Source": "Analysis Failed",
                    "F_Stat": "",
                    "p_value": "",
                    "Significance": "Error"
                })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_file, index=False)
        print(f"\nAnalysis complete. Results saved to {output_file}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    input_file = "mock_data.csv"
    output_file = "analysis_results.csv"
    run_analysis(input_file, output_file)