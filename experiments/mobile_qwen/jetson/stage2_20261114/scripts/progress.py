"""Read-only compact progress for the long Jetson accuracy pass."""
from pathlib import Path
import json,subprocess
S=Path(__file__).resolve().parents[1]
remote='''import json
from pathlib import Path
s=Path("/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage2_20261114")
p=s/"evaluations/jetson_2b_lora_nf4_512_noswap.metadata.jsonl"
x=[json.loads(a) for a in p.read_text().splitlines()] if p.exists() else []
m=x[-1]["metadata"]["memory"] if x else {}
f=x[0]["metadata"]["memory"] if x else {}
print(json.dumps({"queries":len(x),"model_swap":m.get("cgroup_swap_current"),"global_swap_in_delta":m.get("pswpin",0)-f.get("pswpin",0),"global_swap_out_delta":m.get("pswpout",0)-f.get("pswpout",0),"resources_complete":(s/"evaluations/jetson_2b_lora_nf4_512_noswap.resources.json").exists()}))'''
p=subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=5','csip@192.168.55.1','/home/csip/miniforge3/envs/ali-jetson/bin/python','-c',__import__('shlex').quote(remote)],capture_output=True,text=True)
result=json.loads(p.stdout) if p.returncode==0 else {'connection_error':p.stderr.strip()}
for size in ['2b','4b']:
 f=S/f'evaluations/pc_{size}_lora_nf4_512.metadata.jsonl'
 result['pc_'+size+'_queries']=sum(1 for _ in f.open()) if f.exists() else 0
result['report_ready']=(S/'REPORT.pdf').exists()
print(json.dumps(result,ensure_ascii=False))
