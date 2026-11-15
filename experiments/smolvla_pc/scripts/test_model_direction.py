"""Operator-approved 1-degree shoulder/elbow direction probe, not a VLA rollout."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/so101_pickplace/scripts'))
from read_follower import PORT,read_register,serial,NAMES
out=ROOT/'experiments/smolvla_pc/runs'/('direction_test_'+str(time.time_ns()));out.mkdir(parents=True)
def emit(event,**kw):
 row={'event':event,'monotonic':time.monotonic(),**kw}
 with (out/'motion.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def read(b,i,a,n):return read_register(b,i,a,n)
def write(b,i,a,v,n):
 if i not in range(1,6) or a not in (40,42):raise ValueError('Write outside probe scope')
 body=bytes([i,n+3,3,a])+int(v).to_bytes(n,'little')
 b.reset_input_buffer();b.write(b'\xff\xff'+body+bytes([(~sum(body))&255]));r=b.read(6)
 if len(r)!=6 or r[:2]!=b'\xff\xff' or r[2]!=i or r[3]!=2 or r[4] or sum(r[2:])&255!=255:raise RuntimeError('Write acknowledgement failed')
with serial.Serial(PORT,1000000,timeout=.4,exclusive=True) as b:
 start=[read(b,i,56,2) for i in range(1,6)]
 if [read(b,i,40,1) for i in range(1,7)]!=[0]*6:raise RuntimeError('Expected torque off')
 expected=[2040,1935,2141,2065,976]
 if max(abs(a-c) for a,c in zip(start,expected))>6:raise RuntimeError('Pose changed; re-evaluate')
 for i,v in enumerate(start,1):
  if not read(b,i,9,2)<=v<=read(b,i,11,2):raise RuntimeError('Outside motor limits')
  if read(b,i,63,1)>=55:raise RuntimeError('Temperature limit')
 target=start.copy();target[1]+=11;target[2]+=11
 for i,v in enumerate(target,1):
  if not read(b,i,9,2)<=v<=read(b,i,11,2):raise RuntimeError('Target outside limits')
 emit('preflight',actual=start,target=target,gripper_driven=False)
 enabled=[]
 try:
  for i,v in enumerate(start,1):write(b,i,42,v,2)
  for i in range(1,6):
   enabled.append(i);write(b,i,40,1,1)
  time.sleep(.3)
  previous=start.copy()
  for k in range(1,12):
   actual=[read(b,i,56,2) for i in range(1,6)]
   if max(abs(a-c) for a,c in zip(actual,previous))>34:raise RuntimeError('Following error exceeds 3 degrees; holding')
   previous=start.copy();previous[1]+=k;previous[2]+=k
   write(b,2,42,previous[1],2);write(b,3,42,previous[2],2);time.sleep(.15)
  time.sleep(.4)
  emit('probe_completed_holding',actual=[read(b,i,56,2) for i in range(1,6)],goal=previous)
 except BaseException as e:emit('stopped_holding',error=str(e))
 finally:
  if enabled:
   while input('Holding. Support white arm then type OFF: ').strip()!='OFF':pass
   for i in enabled:write(b,i,40,0,1)
   emit('torque_off',state=[read(b,i,40,1) for i in range(1,7)])
