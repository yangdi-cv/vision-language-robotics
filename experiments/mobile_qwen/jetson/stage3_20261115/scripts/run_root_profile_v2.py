"""Retry the one profile with administrative CUPTI access, without changing driver policy."""
from pathlib import Path
import subprocess,os,time,shlex
S=Path(__file__).resolve().parents[1];HOST='csip@192.168.55.1';R='/home/csip/ali-jetson/project/experiments/mobile_qwen/jetson/stage3_20261115';P='/home/csip/ali-jetson/deployment_20261114/jetson_python.sh'
def ssh(cmd,**kw):return subprocess.run(['sshpass','-e','ssh','-oConnectTimeout=5',HOST,cmd],**kw)
def root(cmd):
 r=ssh('sudo -S bash -c '+shlex.quote(cmd),input=os.environ['SSHPASS']+'\n',text=True,capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr)
subprocess.run(['rsync','-a','-e','sshpass -e ssh',str(S/'scripts/profile_one.py'),HOST+':'+R+'/scripts/'],check=True)
assert ssh('test ! -e '+R+'/profile_one_root_v2').returncode==0
job=None
try:
 with (S/'root_profile_v2_driver.log').open('w') as log:
  command='sudo -S systemd-run --wait --pipe --collect --unit ali-profile-root-v2-20261115 --setenv=ALI_PROFILE_TAG=profile_one_root_v2 -p MemoryHigh=5G -p MemoryMax=6G -p OOMPolicy=stop '+P+' '+R+'/scripts/profile_one.py'
  job=subprocess.Popen(['sshpass','-e','ssh',HOST,command],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True);job.stdin.write(os.environ['SSHPASS']+'\n');job.stdin.close()
  end=time.monotonic()+300
  while ssh('test -f '+R+'/profile_one_root_v2/load.json').returncode:
   if job.poll() is not None:raise RuntimeError('Root profile load failed')
   if time.monotonic()>end:raise TimeoutError('Root profile load')
   time.sleep(3)
  root('swapoff -a');root('test $(wc -l < /proc/swaps) -eq 1 && touch '+R+'/profile_one_root_v2/swap_ready')
  if job.wait(timeout=300):raise RuntimeError('Root profiling failed, retained result')
finally:
 if job and job.poll() is None:root('systemctl stop ali-profile-root-v2-20261115')
 root('swapon -a; for d in /dev/zram[0-5]; do swapon --priority 5 "$d" 2>/dev/null || true; done')
 root('chown -R csip:csip '+R+'/profile_one_root_v2')
 subprocess.run(['rsync','-a','-e','sshpass -e ssh',HOST+':'+R+'/profile_one_root_v2/',str(S/'profile_one_root_v2')+'/'],check=False)
