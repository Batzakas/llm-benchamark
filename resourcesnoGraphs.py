import psutil
import time
import json
import subprocess
from datetime import datetime, timezone
import threading
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
        return float(output.strip().split("\n")[0])
    except Exception:
        return None

def collect_metrics(interval=1):
    global monitoring, current_phase, data, cpu_start_energy
    print("[+] Iniciando coleta contínua de métricas...")
    cpu_start_energy = read_rapl_energy()

    while monitoring:
        if current_phase is not None:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_percent": psutil.virtual_memory().percent,
                "gpu_power_w": read_gpu_power()
            }
            data[current_phase].append(record)
        time.sleep(interval)

    print("[✓] Coleta finalizada.")

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

    req = request.get_json() or {}
    meta = req
    meta["start_time"] = datetime.now(timezone.utc).isoformat()

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
    time.sleep(2)

    cpu_end_energy = read_rapl_energy()
    meta["end_time"] = datetime.now(timezone.utc).isoformat()
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

    print(f"[✓] Resultados salvos em {output_file}")
    return jsonify({"status": "stopped", "summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

