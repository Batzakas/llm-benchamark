async function fetchData() {
    const res = await fetch("/data");
    return await res.json();
}

function createChart(ctx, datasets) {
    return new Chart(ctx, {
        type: "line",
        data: { labels: [], datasets },
        options: {
            responsive: true,
            animation: { duration: 350 },
            scales: {
                x: { ticks: { color: "#ccc" } },
                y: { ticks: { color: "#ccc" } }
            },
            plugins: {
                legend: { labels: { color: "#fff" } }
            }
        }
    });
}

function addData(chart, label, values) {
    chart.data.labels.push(label);
    for (let i = 0; i < chart.data.datasets.length; i++) {
        chart.data.datasets[i].data.push(values[i]);
    }
    chart.update();
}

const neonColors = {
    init: "#7b00ff",
    inference: "#00eaff",
    idle: "#ff00c8"
};

let cpuChart, ramChart, gpuPowerChart, gpuUtilChart, gpuMemChart, netChart;

window.onload = () => {
    cpuChart = createChart(document.getElementById("cpuCmp"), [
        { label: "Init", borderColor: neonColors.init, data: [] },
        { label: "Inference", borderColor: neonColors.inference, data: [] },
        { label: "Idle", borderColor: neonColors.idle, data: [] }
    ]);

    ramChart = createChart(document.getElementById("ramCmp"), [
        { label: "Init", borderColor: neonColors.init, data: [] },
        { label: "Inference", borderColor: neonColors.inference, data: [] },
        { label: "Idle", borderColor: neonColors.idle, data: [] }
    ]);

    gpuPowerChart = createChart(document.getElementById("gpuPowerCmp"), [
        { label: "Init", borderColor: neonColors.init, data: [] },
        { label: "Inference", borderColor: neonColors.inference, data: [] },
        { label: "Idle", borderColor: neonColors.idle, data: [] }
    ]);

    gpuUtilChart = createChart(document.getElementById("gpuUtilCmp"), [
        { label: "Init", borderColor: neonColors.init, data: [] },
        { label: "Inference", borderColor: neonColors.inference, data: [] },
        { label: "Idle", borderColor: neonColors.idle, data: [] }
    ]);

    gpuMemChart = createChart(document.getElementById("gpuMemCmp"), [
        { label: "Init", borderColor: neonColors.init, data: [] },
        { label: "Inference", borderColor: neonColors.inference, data: [] },
        { label: "Idle", borderColor: neonColors.idle, data: [] }
    ]);

    netChart = createChart(document.getElementById("netCmp"), [
        { label: "Init RX", borderColor: neonColors.init, data: [] },
        { label: "Inference RX", borderColor: neonColors.inference, data: [] },
        { label: "Idle RX", borderColor: neonColors.idle, data: [] },
        { label: "Init TX", borderColor: neonColors.init, borderDash: [5, 4], data: [] },
        { label: "Inference TX", borderColor: neonColors.inference, borderDash: [5, 4], data: [] },
        { label: "Idle TX", borderColor: neonColors.idle, borderDash: [5, 4], data: [] }
    ]);

    setInterval(updateCharts, 1500);
};

async function updateCharts() {
    const json = await fetchData();
    const samples = json.samples;

    const phases = ["init", "inference", "idle"];
    const last = {};

    for (const p of phases) {
        if (samples[p].length > 0) {
            last[p] = samples[p][samples[p].length - 1];
        }
    }

    const label = new Date().toLocaleTimeString();

    addData(cpuChart, label, [
        last.init?.cpu_percent ?? 0,
        last.inference?.cpu_percent ?? 0,
        last.idle?.cpu_percent ?? 0
    ]);

    addData(ramChart, label, [
        last.init?.mem_percent ?? 0,
        last.inference?.mem_percent ?? 0,
        last.idle?.mem_percent ?? 0
    ]);

    addData(gpuPowerChart, label, [
        last.init?.gpu_power_w ?? 0,
        last.inference?.gpu_power_w ?? 0,
        last.idle?.gpu_power_w ?? 0
    ]);

    addData(gpuUtilChart, label, [
        last.init?.gpu_util ?? 0,
        last.inference?.gpu_util ?? 0,
        last.idle?.gpu_util ?? 0
    ]);

    addData(gpuMemChart, label, [
        last.init?.gpu_mem_mb ?? 0,
        last.inference?.gpu_mem_mb ?? 0,
        last.idle?.gpu_mem_mb ?? 0
    ]);

    addData(netChart, label, [
        last.init?.net_recv_mb ?? 0,
        last.inference?.net_recv_mb ?? 0,
        last.idle?.net_recv_mb ?? 0,
        last.init?.net_sent_mb ?? 0,
        last.inference?.net_sent_mb ?? 0,
        last.idle?.net_sent_mb ?? 0
    ]);

    updateExperimentPanel(json.experiment);
    updatePhasePanel(json.summary);
    updateSummaryPanel(json.summary);
}

function updateExperimentPanel(exp) {
    document.getElementById("expBox").innerHTML =
        `<b>RUN:</b> ${exp.run_id}<br>
         <b>MODEL:</b> ${exp.model}<br>
         <b>QUANT:</b> ${exp.quant}<br>
         <b>CTX:</b> ${exp.context}`;
}

function updatePhasePanel(sum) {
    document.getElementById("phaseBox").innerHTML =
        `<b>Init CPU:</b> ${sum.init.cpu_avg?.toFixed(1)}%<br>
         <b>Inference CPU:</b> ${sum.inference.cpu_avg?.toFixed(1)}%<br>
         <b>Idle CPU:</b> ${sum.idle.cpu_avg?.toFixed(1)}%`;
}

function updateSummaryPanel(sum) {
    const globalCpu =
        (sum.init.cpu_avg + sum.inference.cpu_avg + sum.idle.cpu_avg) / 3;

    const globalRam =
        (sum.init.mem_avg + sum.inference.mem_avg + sum.idle.mem_avg) / 3;

    const gpuUtil =
        (sum.init.gpu_util_avg + sum.inference.gpu_util_avg + sum.idle.gpu_util_avg) / 3;

    const gpuPower =
        (sum.init.gpu_power_avg + sum.inference.gpu_power_avg + sum.idle.gpu_power_avg) / 3;

    document.getElementById("summaryBox").innerHTML =
        `<b>CPU Média:</b> ${globalCpu.toFixed(1)}%<br>
         <b>RAM Média:</b> ${globalRam.toFixed(1)}%<br>
         <b>GPU Util:</b> ${gpuUtil.toFixed(1)}%<br>
         <b>GPU Power:</b> ${gpuPower.toFixed(1)} W<br>
         <b>GPU Mem:</b> ${(sum.init.gpu_mem_avg_mb + sum.inference.gpu_mem_avg_mb + sum.idle.gpu_mem_avg_mb)/3} MB<br>
         <b>RX Médio:</b> ${(sum.init.net_recv_mb + sum.inference.net_recv_mb + sum.idle.net_recv_mb).toFixed(2)} MB<br>
         <b>TX Médio:</b> ${(sum.init.net_sent_mb + sum.inference.net_sent_mb + sum.idle.net_sent_mb).toFixed(2)} MB<br>
         <b>Disk R Médio:</b> ${(sum.init.disk_read_mb + sum.inference.disk_read_mb + sum.idle.disk_read_mb).toFixed(2)} MB<br>
         <b>Disk W Médio:</b> ${(sum.init.disk_write_mb + sum.inference.disk_write_mb + sum.idle.disk_write_mb).toFixed(2)} MB<br>
         <b>Temp CPU:</b> ${sum.init.cpu_temp_avg?.toFixed(1)}°C`;
}


function openFullscreen(canvasId) {
    const elem = document.getElementById(canvasId);

    if (elem.requestFullscreen) elem.requestFullscreen();
    else if (elem.webkitRequestFullscreen) elem.webkitRequestFullscreen();
    else if (elem.msRequestFullscreen) elem.msRequestFullscreen();
}





