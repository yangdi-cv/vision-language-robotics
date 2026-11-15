import argparse,subprocess,os,signal,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_extension'
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);f=(out/'launch.log').open('w')
cmd=[str(TASK/'scripts/moveit-python'),'-c',"import os; os.execvp('ros2',['ros2','launch',"+repr(str(TASK/'scripts/moveit_headless.launch.py'))+"])"]
launch=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
try:
 with (out/'goals.log').open('w') as log:result=subprocess.run([str(TASK/'scripts/moveit-python'),str(TASK/'scripts/moveit_goals.py'),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=200)
 (out/'exit.json').write_text(json.dumps({'client_exit':result.returncode}));print('MoveIt client exit',result.returncode,flush=True)
finally:
 if launch.poll() is None:
  os.killpg(launch.pid,signal.SIGINT)
  try:launch.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(launch.pid,signal.SIGKILL);launch.wait()
 f.close()
