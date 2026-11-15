import argparse,subprocess,os,signal,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_extension'
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--with-act',action='store_true');a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
f=(out/'nav2_launch.log').open('w');launch=subprocess.Popen([str(TASK/'scripts/nav-launch')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
baglog=(out/'rosbag.log').open('w');bag=subprocess.Popen([str(TASK/'scripts/ros-python'),'-c',"import os;os.execvp('ros2',['ros2','bag','record','-o',"+repr(str(out/'sim_rosbag'))+",'/odom','/cmd_vel','/scan','/clock','/tf','/tf_static'])"],stdout=baglog,stderr=subprocess.STDOUT,start_new_session=True)
try:
 with (out/'goals.log').open('w') as goal_log:
  proc=subprocess.run([str(TASK/'scripts/ros-python'),str(TASK/'scripts/nav_goals.py'),'--out',str(out)]+(['--act-model',str(ROOT/'experiments/alibaba_baseline/data/official_act')] if a.with_act else []),stdout=goal_log,stderr=subprocess.STDOUT,timeout=350)
 (out/'client_exit.json').write_text(json.dumps({'exit':proc.returncode}));print('Nav client exit',proc.returncode,flush=True)
finally:
 if bag.poll() is None:
  os.killpg(bag.pid,signal.SIGINT)
  try:bag.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(bag.pid,signal.SIGKILL);bag.wait()
 baglog.close()
 if launch.poll() is None:
  os.killpg(launch.pid,signal.SIGINT)
  try:launch.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(launch.pid,signal.SIGKILL);launch.wait()
 f.close()
