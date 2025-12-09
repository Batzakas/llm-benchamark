import json
import csv
import math
from collections import defaultdict
from statistics import mean

INPUT_JSONL = "output_dataset.jsonl"
OUTPUT_CSV = "analysis_per_prompt.csv"

METRICS_KEYS = [
    "prompt_tokens",
    "response_tokens",
    "total_tokens",
    "model_load_time_seconds",
    "prompt_eval_time_seconds",
    "response_time_seconds",
    "total_time_seconds",
    "prompt_tokens_per_second",
    "response_tokens_per_second",
    "total_tokens_per_second"
]

def calculate_p95(values):
    if not values:
        return 0.0
    values = sorted(values)
    k = math.ceil(0.95 * len(values)) - 1
    return values[max(0, min(k, len(values) - 1))]

def main():
    agrupado = defaultdict(lambda: defaultdict(list))

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = json.loads(line)

                model = row["model"]
                prompt_id = row["id"]
                metrics = row["metrics"]

                key = (model, prompt_id)

                for m in METRICS_KEYS:
                    agrupado[key][m].append(metrics[m])

            except Exception as e:
                print(f"Erro na linha {line_num}: {e}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        header = ["model", "prompt_id"]
        for m in METRICS_KEYS:
            header.append(f"{m}_mean")
            header.append(f"{m}_p95")

        writer.writerow(header)

        for (model, prompt_id), metrics_dict in agrupado.items():
            row = [model, prompt_id]

            for m in METRICS_KEYS:
                media = mean(metrics_dict[m])
                p95 = calculate_p95(metrics_dict[m])

                row.append(round(media, 6))
                row.append(round(p95, 6))

            writer.writerow(row)

    print(f"\n Gerado em: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
