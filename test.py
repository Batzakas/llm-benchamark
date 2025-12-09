import requests, time, math
def heavy_cpu(s):
    end=time.time()+s
    x=0.0001
    while time.time()<end:
        x=math.sqrt(x*x+0.0001)
requests.post("http://localhost:5001/start",json={"run_id":"local_comp_a","model":"FAKE","quant":"none","context":"none"})
time.sleep(1)
requests.post("http://localhost:5001/set_phase",json={"phase":"init"})
heavy_cpu(4)
requests.post("http://localhost:5001/set_phase",json={"phase":"inference"})
heavy_cpu(10)
requests.post("http://localhost:5001/set_phase",json={"phase":"idle"})
time.sleep(3)
resp=requests.post("http://localhost:5001/stop")
print(resp.json())



