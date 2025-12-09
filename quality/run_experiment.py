import argparse
import config
import quality.main as main
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("block", type=str)
    parser.add_argument("--output", type=str, default="results_experiment.jsonl")
    args = parser.parse_args()
    
    block_id = args.block.upper()
    runs = config.get_design_matrix(block_id)
    replicas = config.DESIGN_SPECS[block_id]["replicas"]
    
    open(args.output, 'w').close()
    
    print(f"Running Block {block_id} | Configs: {len(runs)} | Replicas: {replicas}")
    
    for r in range(1, replicas + 1):
        for cfg in runs:
            cfg['block_id'] = block_id
            main.run_single_pass_evaluation(cfg, replica_id=r, output_path=args.output)

if __name__ == "__main__":
    main()