"""Sequential Jetson probe with a verified swap-off phase and guaranteed restoration."""
from pathlib import Path
import subprocess,os,time,shlex,json
S=Path(__file__).resolve().parents[1];HOST='csip@192.168.55.1';REMOTE='/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage3_20261115';PYTHON='/home/csip/ali-jetson/deployment_20261114/jetson_python.sh'
assert os.environ.get('SSHPASS')
def ssh(command,**kw):return subprocess.run(['sshpass','-e','ssh','-oConnectTimeout=5',HOST,command],**kw)
def root(command):
 r=ssh('sudo -S bash -c '+shlex.quote(command),input=os.environ['SSHPASS']+'\n',text=True,capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr)
try:
 subprocess.run(['rsync','-a','-e','sshpass -e ssh',str(S)+'/',HOST+':'+REMOTE+'/'],check=True)
 check=ssh('test ! -e '+REMOTE+'/results');assert check.returncode==0,'Unique result tag already exists'
 log=(S/'driver.log').open('w');job=subprocess.Popen(['sshpass','-e','ssh',HOST,'systemd-run --user --wait --pipe --collect --unit ali-latency-probe-20261115 -p MemoryHigh=5G -p MemoryMax=6G -p OOMPolicy=stop '+PYTHON+' '+REMOTE+'/scripts/latency_probe.py'],stdout=log,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+600
 while ssh('test -f '+REMOTE+'/results/load.json').returncode:
  if job.poll() is not None:raise RuntimeError('Jetson inference exited before load gate')
  if time.monotonic()>deadline:raise TimeoutError('Load gate')
  time.sleep(3)
 root('swapoff -a');r=ssh('test $(wc -l < /proc/swaps) -eq 1 && touch '+REMOTE+'/results/swap_ready');assert r.returncode==0
 if job.wait(timeout=1500):raise RuntimeError('Jetson probe failed')
finally:
 # Restore the exact previously used SD and zram swap topology.
 root('swapon -a; for d in /dev/zram[0-5]; do swapon --priority 5 "$d" 2>/dev/null || true; done')
 subprocess.run(['rsync','-a','-e','sshpass -e ssh',HOST+':'+REMOTE+'/results/',str(S/'results')+'/'],check=False)
 if 'log' in locals():log.close()
if (S/'results/SUMMARY.json').exists():
 x=json.loads((S/'results/SUMMARY.json').read_text());print('VALID_WITHOUT_SWAP',x['valid_without_swap']);print(json.dumps(x['latency'],indent=2),flush=True)
