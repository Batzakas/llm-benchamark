from flask import Flask, jsonify, request, render_template
from datetime import datetime
import psutil, subprocess, threading, time

app = Flask(__name__, static_folder="static", template_folder="templates")

collecting = False
current_phase = None
data = {"init": [], "inference": [], "idle": []}
experiment_info = {}
collector = None


def read_rapl_energy():
    try:
        with open("/sys/class/powercap/intel-rapl:0/energy_uj","r") as f:
            micro = int(f.read().strip())
        return micro/1e6/3600
    except:
        return None

def read_gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip().split(", ")
        return {
            "gpu_util": float(out[0]),
            "gpu_mem_mb": float(out[1]),
            "gpu_power_w": float(out[2])
        }
    except:
        return {"gpu_util": None, "gpu_mem_mb": None, "gpu_power_w": None}


def collect_loop():
    while collecting:
        if current_phase:
            ts = datetime.utcnow().isoformat()
            cpu = psutil.cpu_percent(interval=None)
            cpu_times = psutil.cpu_times_percent()
            mem = psutil.virtual_memory()
            disk = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            load1, load5, load15 = psutil.getloadavg() if hasattr(psutil,'getloadavg') else (0,0,0)

            temps = psutil.sensors_temperatures() if hasattr(psutil,'sensors_temperatures') else {}
            cpu_temp = None
            if "coretemp" in temps and temps["coretemp"]:
                cpu_temp = temps["coretemp"][0].current

            gpu = read_gpu_stats()
            energy = read_rapl_energy()

            record = {
                "ts": ts,
                "cpu_percent": cpu,
                "cpu_user": cpu_times.user,
                "cpu_system": cpu_times.system,
                "load1": load1,
                "mem_percent": mem.percent,
                "mem_used_mb": mem.used / 1024 / 1024,
                "swap_percent": psutil.swap_memory().percent,
                "disk_read_mb": disk.read_bytes / 1024 / 1024,
                "disk_write_mb": disk.write_bytes / 1024 / 1024,
                "net_sent_mb": net.bytes_sent / 1024 / 1024,
                "net_recv_mb": net.bytes_recv / 1024 / 1024,
                "cpu_temp": cpu_temp,
                "gpu_util": gpu["gpu_util"],
                "gpu_mem_mb": gpu["gpu_mem_mb"],
                "gpu_power_w": gpu["gpu_power_w"],
                "rapl_wh": energy
            }

            data[current_phase].append(record)

        time.sleep(1)



def summarize_phase(phase_data):
    if not phase_data:
        return {"samples": 0}

    def avg(k):
        vals = [d[k] for d in phase_data if d.get(k) is not None]
        return sum(vals) / len(vals) if vals else None

    def mx(k):
        vals = [d[k] for d in phase_data if d.get(k) is not None]
        return max(vals) if vals else None

    start = phase_data[0]["ts"]
    end = phase_data[-1]["ts"]

    return {
        "samples": len(phase_data),
        "start": start,
        "end": end,
        "duration_s": (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(),
        "cpu_avg": avg("cpu_percent"),
        "cpu_max": mx("cpu_percent"),
        "mem_avg": avg("mem_percent"),
        "mem_max": mx("mem_percent"),

        "gpu_power_avg": avg("gpu_power_w"),
        "gpu_power_max": mx("gpu_power_w"),
        "gpu_util_avg": avg("gpu_util"),
        "gpu_mem_avg_mb": avg("gpu_mem_mb"),

        "disk_read_mb": avg("disk_read_mb"),
        "disk_write_mb": avg("disk_write_mb"),
        "net_sent_mb": avg("net_sent_mb"),
        "net_recv_mb": avg("net_recv_mb"),

        "cpu_temp_avg": avg("cpu_temp"),
        "rapl_wh_delta": None
    }


def global_summary():
    merged = data["init"] + data["inference"] + data["idle"]

    if not merged:
        return {}

    def avg(k):
        vals = [d[k] for d in merged if d.get(k) is not None]
        return sum(vals)/len(vals) if vals else None

    return {
        "cpu_avg": avg("cpu_percent"),
        "mem_avg": avg("mem_percent"),
        "gpu_util_avg": avg("gpu_util"),
        "gpu_power_avg": avg("gpu_power_w"),
        "gpu_mem_avg_mb": avg("gpu_mem_mb"),
        "net_recv_avg": avg("net_recv_mb"),
        "net_sent_avg": avg("net_sent_mb"),
        "disk_read_avg": avg("disk_read_mb"),
        "disk_write_avg": avg("disk_write_mb"),
        "cpu_temp_avg": avg("cpu_temp"),
    }


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/start", methods=["POST"])
def start():
    global collecting, current_phase, data, experiment_info, collector, rapl_start

    experiment_info = request.json or {}
    data = {"init": [], "inference": [], "idle": []}

    collecting = True
    current_phase = "init"

    rapl_start = read_rapl_energy()

    collector = threading.Thread(target=collect_loop, daemon=True)
    collector.start()

    return jsonify({"status": "started"})


@app.route("/set_phase", methods=["POST"])
def set_phase():
    global current_phase

    phase = request.json.get("phase")
    if phase not in data:
        return jsonify({"error": "invalid phase"}), 400

    current_phase = phase
    return jsonify({"status": "phase_set", "phase": phase})


@app.route("/stop", methods=["POST"])
def stop():
    global collecting, current_phase

    collecting = False
    current_phase = None

    rapl_end = read_rapl_energy()

    summary = {ph: summarize_phase(data[ph]) for ph in data}

    total_wh = rapl_end - rapl_start if rapl_start and rapl_end else None

    gsum = global_summary()

    result = {
        "experiment": experiment_info,
        "summary": summary,
        "global_summary": gsum,
        "samples": data,
        "total_cpu_wh": total_wh
    }

    return jsonify(result)


@app.route("/data")
def get_data():
    return jsonify({
        "experiment": experiment_info,
        "summary": {ph: summarize_phase(data[ph]) for ph in data},
        "global_summary": global_summary(),
        "samples": data
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)







