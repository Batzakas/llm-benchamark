import psutil
import time
import json
import subprocess
from datetime import datetime, UTC
import threading
import matplotlib.pyplot as plt
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

monitoring = False
current_phase = None
data = {"init": [], "inference": [], "idle": []}
meta = {}
cpu_start_energy = None


def read_rapl_energy():
    """Lê energia da CPU via RAPL (Wh)"""
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as f:
            microjoules = int(f.read().strip())
        return microjoules / 1e6 / 3600  # µJ → Wh
    except FileNotFoundError:
        return None


def read_gpu_power():
    """Lê consumo instantâneo da GPU (W) via nvidia-smi"""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True
        )
        watts = float(output.strip().split("\n")[0])
        return watts
    except Exception:
        return None


def collect_metrics(interval=1):
    global monitoring, current_phase, data, cpu_start_energy

    print("[+] Iniciando coleta contínua de métricas...")
    cpu_start_energy = read_rapl_energy()

    while monitoring:
        if current_phase is not None:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            gpu_power = read_gpu_power()

            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "cpu_percent": cpu_percent,
                "mem_percent": mem.percent,
                "gpu_power_w": gpu_power
            }

            data[current_phase].append(record)

        time.sleep(interval)

    print("[✓] Coleta finalizada.")

def plot_phase_data(phase_name, records, run_id):
    if not records:
        print(f"[!] Nenhum dado disponível para {phase_name}.")
        return

    timestamps = [datetime.fromisoformat(r["timestamp"]) for r in records]
    cpu = [r["cpu_percent"] for r in records]
    mem = [r["mem_percent"] for r in records]
    gpu = [r["gpu_power_w"] for r in records if r["gpu_power_w"] is not None]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, cpu, label="CPU (%)")
    plt.plot(timestamps, mem, label="RAM (%)")
    if gpu:
        plt.plot(timestamps[:len(gpu)], gpu, label="GPU (W)")

    plt.title(f"Uso de Recursos - {phase_name} ({run_id})")
    plt.xlabel("Tempo")
    plt.ylabel("Uso (%) / Potência (W)")
    plt.legend()
    plt.tight_layout()

    os.makedirs("graphs", exist_ok=True)
    plt.savefig(f"graphs/{run_id}_{phase_name}.png")
    plt.close()


def generate_graphs(run_id):
    for phase, records in data.items():
        plot_phase_data(phase, records, run_id)
    print("[✓] Gráficos salvos na pasta ./graphs/")


def summarize_phase(phase_data):
    if not phase_data:
        return {}

    avg_cpu = sum(d["cpu_percent"] for d in phase_data) / len(phase_data)
    avg_mem = sum(d["mem_percent"] for d in phase_data) / len(phase_data)
    gpu_values = [d["gpu_power_w"] for d in phase_data if d["gpu_power_w"] is not None]
    avg_gpu = sum(gpu_values) / len(gpu_values) if gpu_values else None

    return {
        "samples": len(phase_data),
        "avg_cpu_percent": round(avg_cpu, 2),
        "avg_mem_percent": round(avg_mem, 2),
        "avg_gpu_power_w": round(avg_gpu, 2) if avg_gpu else None
    }

@app.route("/start", methods=["POST"])
def start_monitor():
    global monitoring, data, meta

    if monitoring:
        return jsonify({"error": "monitoring already active"}), 400

    req = request.get_json()
    meta = req or {}
    meta["start_time"] = datetime.now(UTC).isoformat()

    for key in data.keys():
        data[key] = []

    monitoring = True
    threading.Thread(target=collect_metrics, args=(1,), daemon=True).start()

    return jsonify({"status": "started", "meta": meta})


@app.route("/phase", methods=["POST"])
def set_phase():
    global current_phase
    req = request.get_json()
    phase = req.get("phase")
    if phase not in data:
        return jsonify({"error": "invalid phase"}), 400

    current_phase = phase
    print(f"[→] Fase atual: {phase}")
    return jsonify({"status": "ok", "current_phase": phase})


@app.route("/stop", methods=["POST"])
def stop_monitor():
    global monitoring, cpu_start_energy, meta

    monitoring = False
    time.sleep(2)  # Espera finalização da thread

    cpu_end_energy = read_rapl_energy()
    meta["end_time"] = datetime.now(UTC).isoformat()
    meta["energy_wh_cpu"] = (
        cpu_end_energy - cpu_start_energy
        if cpu_start_energy and cpu_end_energy
        else None
    )

    summary = {phase: summarize_phase(records) for phase, records in data.items()}
    result = {"meta": meta, "summary": summary, "samples": data}

    run_id = meta.get("run_id", "default")
    output_file = f"resources_{run_id}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    generate_graphs(run_id)

    print(f"[✓] Resultados salvos em {output_file}")
    return jsonify({"status": "stopped", "summary": summary})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
