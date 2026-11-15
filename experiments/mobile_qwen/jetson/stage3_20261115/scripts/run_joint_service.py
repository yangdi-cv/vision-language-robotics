"""After the finite latency screen, test three PC-physics / Jetson-vision cases."""
from pathlib import Path
import subprocess,os,time,shlex,json,urllib.request,base64
S=Path(__file__).resolve().parents[1];ROOT=S.parents[3];HOST='csip@192.168.55.1';REMOTE='/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage3_20261115';P='/home/csip/ali-jetson/deployment_20261114/jetson_python.sh';unit='ali-vision-service-20261115'
assert os.environ.get('SSHPASS')
def ssh(command,**kw):return subprocess.run(['sshpass','-e','ssh','-oConnectTimeout=5',HOST,command],**kw)
def root(cmd):
 r=ssh('sudo -S bash -c '+shlex.quote(cmd),input=os.environ['SSHPASS']+'\n',text=True,capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr)
for _ in range(240):
 active=subprocess.run(['systemctl','--user','is-active','ali-jetson-latency-host-20261115'],capture_output=True,text=True).stdout.strip()=='active'
 if not active:break
 time.sleep(10)
else:raise TimeoutError('Prior latency screen did not end')
assert (S/'results/SUMMARY.json').exists(),'Prior screen incomplete'
assert json.loads((S/'results/SUMMARY.json').read_text())['valid_without_swap'],'Invalid prior probe'
job=tunnel=None;log=None
try:
 subprocess.run(['rsync','-a','--exclude=results','--exclude=service_run','-e','sshpass -e ssh',str(S)+'/',HOST+':'+REMOTE+'/'],check=True)
 assert ssh('test ! -e '+REMOTE+'/service_run').returncode==0
 log=(S/'service_driver.log').open('w');job=subprocess.Popen(['sshpass','-e','ssh',HOST,'systemd-run --user --wait --pipe --collect --unit '+unit+' -p MemoryHigh=5G -p MemoryMax=6G -p OOMPolicy=stop '+P+' '+REMOTE+'/scripts/vision_service.py'],stdout=log,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+600
 while ssh('test -f '+REMOTE+'/service_run/load.json').returncode:
  if job.poll() is not None:raise RuntimeError('Service load failed')
  if time.monotonic()>deadline:raise TimeoutError('Service loading')
  time.sleep(3)
 root('swapoff -a');assert ssh('test $(wc -l < /proc/swaps) -eq 1 && touch '+REMOTE+'/service_run/swap_ready').returncode==0
 tunnel=subprocess.Popen(['sshpass','-e','ssh','-N','-oExitOnForwardFailure=yes','-L','127.0.0.1:18889:127.0.0.1:18888',HOST],stdout=log,stderr=subprocess.STDOUT)
 for _ in range(60):
  try:
   with urllib.request.urlopen('http://127.0.0.1:18889',timeout=2) as r:assert json.load(r)['ready']
   break
  except Exception:
   if tunnel.poll() is not None:raise RuntimeError('Tunnel failed')
   time.sleep(1)
 else:raise TimeoutError('Service not ready')
 manifest=json.loads((S/'input_manifest.json').read_text());data=(S/'validation'/manifest['images'][0]['file']).read_bytes()
 request=urllib.request.Request('http://127.0.0.1:18889',data=json.dumps({'mode':'lora','phase':'warmup','image_base64':base64.b64encode(data).decode()}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=90) as response:warmup=json.load(response)
 assert warmup['ok'],warmup
 C=ROOT/'experiments/mobile_qwen/closed_loop_20261115';out=(S/'joint_cases.log').open('w')
 with out:
  subprocess.run(['/usr/bin/python3',str(C/'scripts/run_cases.py'),'--prefix','jetson_joint_v1','--modes','lora','--initials','2,3,4','--variants','nominal','--navigate','--worker-script','remote_grounding_worker.py'],check=True,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=1800)
finally:
 if job:
  ssh('systemctl --user kill --signal=SIGINT '+unit)
  try:job.wait(timeout=30)
  except subprocess.TimeoutExpired:ssh('systemctl --user stop '+unit);job.wait(timeout=15)
 if tunnel:tunnel.terminate();tunnel.wait(timeout=10)
 root('swapon -a; for d in /dev/zram[0-5]; do swapon --priority 5 "$d" 2>/dev/null || true; done')
 subprocess.run(['rsync','-a','-e','sshpass -e ssh',HOST+':'+REMOTE+'/service_run/',str(S/'service_run')+'/'],check=False)
 if log:log.close()
print('JOINT_SERVICE_FINISHED',flush=True)
