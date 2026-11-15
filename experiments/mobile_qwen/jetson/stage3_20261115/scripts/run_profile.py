"""One bounded trace only, after the service test exits; restore swap afterward."""
from pathlib import Path
import subprocess,os,time,shlex
S=Path(__file__).resolve().parents[1];HOST='csip@192.168.55.1';R='/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage3_20261115';P='/home/csip/ali-jetson/deployment_20261114/jetson_python.sh'
def ssh(cmd,**kw):return subprocess.run(['sshpass','-e','ssh','-oConnectTimeout=5',HOST,cmd],**kw)
def root(cmd):
 r=ssh('sudo -S bash -c '+shlex.quote(cmd),input=os.environ['SSHPASS']+'\n',text=True,capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr)
for _ in range(240):
 if subprocess.run(['systemctl','--user','is-active','ali-jetson-joint-host-20261115'],capture_output=True,text=True).stdout.strip()!='active':break
 time.sleep(10)
else:raise TimeoutError('Prior service test exceeded limit')
subprocess.run(['rsync','-a','-e','sshpass -e ssh',str(S/'scripts/profile_one.py'),HOST+':'+R+'/scripts/'],check=True)
job=None
try:
 with (S/'profile_driver.log').open('w') as log:
  job=subprocess.Popen(['sshpass','-e','ssh',HOST,'systemd-run --user --wait --pipe --collect --unit ali-profile-one-20261115 -p MemoryHigh=5G -p MemoryMax=6G -p OOMPolicy=stop '+P+' '+R+'/scripts/profile_one.py'],stdout=log,stderr=subprocess.STDOUT)
  end=time.monotonic()+300
  while ssh('test -f '+R+'/profile_one/load.json').returncode:
   if job.poll() is not None:raise RuntimeError('Profiler load failed')
   if time.monotonic()>end:raise TimeoutError('Profile load')
   time.sleep(3)
  root('swapoff -a');assert ssh('test $(wc -l < /proc/swaps) -eq 1 && touch '+R+'/profile_one/swap_ready').returncode==0
  if job.wait(timeout=300):raise RuntimeError('Profiler failed; retained diagnostic')
finally:
 if job and job.poll() is None:ssh('systemctl --user stop ali-profile-one-20261115')
 root('swapon -a; for d in /dev/zram[0-5]; do swapon --priority 5 "$d" 2>/dev/null || true; done')
 subprocess.run(['rsync','-a','-e','sshpass -e ssh',HOST+':'+R+'/profile_one/',str(S/'profile_one')+'/'],check=False)
