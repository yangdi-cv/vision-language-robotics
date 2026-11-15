"""First verify the ROS/physics joint execution chain without loading a VLM."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
D=Path(__file__).resolve().parents[1];ROOT=D.parents[2]
M=ROOT/'experiments/mobile_vla';V=ROOT/'experiments/vla_libero'
sys.path.insert(0,str(M/'scripts'));from rpc import call
p=__import__('argparse').ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
out=D/'results'/a.tag;sock=Path('/tmp/'+a.tag+'.sock');assert not out.exists() and not sock.exists()
logs=D/'logs'/a.tag;logs.mkdir()
env=os.environ.copy();env['MOBILE_VLA_DOMAIN']='116';env['MOBILE_TF_OVERLAY']=''
processes=[];streams=[]
def start(command,name,ros=False):
 f=(logs/(name+'.log')).open('w');streams.append(f)
 proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);processes.append(proc);return proc
try:
 sim=start([str(V/'scripts/python_shared'),str(D/'scripts/sim_precontact.py'),'--socket',str(sock),'--out',str(out),'--start-docked'],'sim')
 end=time.monotonic()+90
 while not sock.exists():
  if sim.poll() is not None:raise RuntimeError('Simulator exited')
  if time.monotonic()>end:raise TimeoutError('Simulator startup')
  time.sleep(.2)
 bridge=start([str(M/'scripts/ros-python'),str(D/'scripts/physics_bridge.py'),'--socket',str(sock),'--out',str(out)],'bridge')
 launch=start([str(M/'scripts/ros-python'),'/opt/ros/humble/bin/ros2','launch',str(D/'scripts/moveit.launch.py')],'moveit')
 client=start([str(M/'scripts/ros-python'),str(D/'scripts/plan_execute.py'),'--socket',str(sock),'--out',str(out),'--joint-smoke'],'client')
 client.wait(timeout=220)
 print('client exit',client.returncode)
 call(sock,'close_episode')
 if client.returncode:raise RuntimeError('Joint smoke did not pass')
finally:
 for proc in reversed(processes):
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGINT)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 for f in streams:f.close()
 sock.unlink(missing_ok=True)
