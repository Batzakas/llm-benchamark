from typing import Dict, List, Any

ENVIRONMENT_FACTORS: Dict[str, Dict[str, List[Any]]] = {
    "B1": {
        "Model": ["Llama-3.2-1B", "Llama-3.2-3B", "Phi-3-mini"],
        "Quant": ["int4", "int8"],
        "Conc": [1, 10],
        "Context": ["4k", "8k"],
    },
    "B2": {
        "Model": ["Llama-3.2-3B", "Mistral-7B", "Llama-3.1-8B"],
        "Quant": ["int4", "int8"],
        "Conc": [1, 10, 50],
        "Context": ["4k", "8k"],
    },
    "B3": {
        "Model": ["Mistral-7B", "Llama-3.1-13B", "DeepSeek-13B"],
        "Quant": ["int4", "int8", "fp16"],
        "Conc": [1, 10, 50],
        "Context": ["4k", "8k", "32k"],
        "Hardware": ["CPU", "GPU"],
    },
}

DESIGN_SPECS = {
    "B1": {"type": "taguchi", "base_runs": 12, "replicas": 3},
    "B2": {"type": "full_factorial", "base_runs": 36, "replicas": 3},
    "B3": {"type": "taguchi", "base_runs": 18, "replicas": 2},
}

METRICS = ["p95_ms", "Throughput", "ROUGE_L", "F1_Extracao", "Wh_100_prompts"]