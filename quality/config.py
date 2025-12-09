from typing import Dict, List, Any
import itertools

# --- Factor Definitions ---
ENVIRONMENT_FACTORS: Dict[str, Dict[str, List[Any]]] = {
    "B1": {
        "Model": ["llama3.2:1b", "llama3.2:3b", "phi3:mini"],  # 3 Levels (indices 0,1,2)
        "Quant": ["int4", "int8"],                             # 2 Levels (0,1)
        "Conc": [1, 10],                                       # 2 Levels (0,1)
        "Context": ["4k", "8k"],                               # 2 Levels (0,1)
    },
    "B2": {
        "Model": ["llama3.2:3b", "mistral", "llama3.1:8b"],
        "Quant": ["int4", "int8"],
        "Conc": [1, 10, 50],
        "Context": ["4k", "8k"],
    },
    "B3": {
        "Model": ["mistral", "llama3.1:13b", "deepseek-coder:1.3b"], # 3 Levels
        "Quant": ["int4", "int8", "fp16"],                           # 3 Levels
        "Conc": [1, 10, 50],                                         # 3 Levels
        "Context": ["4k", "8k", "32k"],                              # 3 Levels
        "Hardware": ["CPU", "GPU"],                                  # 2 Levels
    },
}

DESIGN_SPECS = {
    "B1": {"type": "taguchi", "table": "L12", "replicas": 3},
    "B2": {"type": "full_factorial", "replicas": 3},
    "B3": {"type": "taguchi", "table": "L18", "replicas": 2},
}

METRICS = ["p95_ms", "Throughput", "ROUGE_L", "F1_Extracao", "Wh_100_prompts"]

# --- Taguchi Tables (Indices) ---
# Rows = Runs, Columns = Factors in order of definition in ENVIRONMENT_FACTORS

# L12
# Design for 12 runs fitting B1 (Model, Quant, Conc, Context).
# Columns: [Model(0-2), Quant(0-1), Conc(0-1), Context(0-1)]
TAGUCHI_L12_B1_INDICES = [
    [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1],
    [1, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 0],
    [2, 0, 1, 1], [2, 1, 0, 0], [2, 0, 0, 1],
    [0, 1, 0, 1], [1, 0, 0, 0], [2, 1, 1, 0] 
]

# L18 (Standard Taguchi L18: 1 factor @ 2 levels, 7 factors @ 3 levels)
# B3 Factors: Model(3), Quant(3), Conc(3), Context(3), Hardware(2)
# Columns mapping below: [Hardware(0-1), Model(0-2), Quant(0-2), Conc(0-2), Context(0-2)]
TAGUCHI_L18_INDICES = [
    [0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 2, 2, 2],
    [0, 1, 0, 1, 1], [0, 1, 1, 2, 2], [0, 1, 2, 0, 0],
    [0, 2, 0, 2, 1], [0, 2, 1, 0, 2], [0, 2, 2, 1, 0],
    [1, 0, 0, 2, 2], [1, 0, 1, 0, 0], [1, 0, 2, 1, 1],
    [1, 1, 0, 0, 1], [1, 1, 1, 1, 2], [1, 1, 2, 2, 0],
    [1, 2, 0, 1, 2], [1, 2, 1, 2, 0], [1, 2, 2, 0, 1]
]

def parse_context(ctx_str: str) -> int:
    """Converts strings like '4k' to integer 4096."""
    if isinstance(ctx_str, int): return ctx_str
    clean = ctx_str.lower().replace("k", "")
    try:
        return int(clean) * 1024
    except ValueError:
        return 2048

def get_design_matrix(block_name: str) -> List[Dict[str, Any]]:
    """Returns the list of resolved configuration dictionaries for the block."""
    if block_name not in ENVIRONMENT_FACTORS:
        raise ValueError(f"Block {block_name} unknown.")

    factors_dict = ENVIRONMENT_FACTORS[block_name]
    factor_names = list(factors_dict.keys())
    factor_values = list(factors_dict.values())
    spec = DESIGN_SPECS.get(block_name, {})
    
    runs = []

    if spec.get("type") == "full_factorial":
        import itertools
        combinations = list(itertools.product(*factor_values))
        for comb in combinations:
            runs.append(dict(zip(factor_names, comb)))

    elif spec.get("type") == "taguchi":
        if block_name == "B1":
            # Map TAGUCHI_L12_B1_INDICES to values
            # Indices cols correspond to factor_names order: Model, Quant, Conc, Context
            table = TAGUCHI_L12_B1_INDICES
            for row_indices in table:
                run_config = {}
                for i, key in enumerate(factor_names):
                    val_idx = row_indices[i] % len(factor_values[i]) 
                    run_config[key] = factor_values[i][val_idx]
                runs.append(run_config)

        elif block_name == "B3":
            # Map TAGUCHI_L18_INDICES to values
            # Our L18 table has Hardware (2-lvl) first.
            # B3 Factors order in dict: Model, Quant, Conc, Context, Hardware
            # We must reorder mapping to match the table: Hardware first, then others.
            table = TAGUCHI_L18_INDICES
            for row_indices in table:
                run_config = {}
                
                run_config["Hardware"] = factors_dict["Hardware"][row_indices[0] % 2]
                run_config["Model"] = factors_dict["Model"][row_indices[1] % 3]
                run_config["Quant"] = factors_dict["Quant"][row_indices[2] % 3]
                run_config["Conc"] = factors_dict["Conc"][row_indices[3] % 3]
                run_config["Context"] = factors_dict["Context"][row_indices[4] % 3]
                
                runs.append(run_config)

    for run in runs:
        if "Context" in run:
            run["Context_Int"] = parse_context(run["Context"])

    return runs