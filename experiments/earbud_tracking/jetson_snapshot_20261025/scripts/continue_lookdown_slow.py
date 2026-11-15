"""Operator-authorized local -11 tick elbow test, seeded at current supported pose."""
import json,time
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'so101_pickplace/scripts'))
from read_follower import PORT,read_register,serial
ROOT=Path(__file__).resolve().parents[1]
log=ROOT/'motion_logs'/('lookdown_slow_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.jsonl')
def emit(event,**kw):
 r={'utc':datetime.now(timezone.utc).isoformat(),'event':event,**kw}
 with log.open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True)
def read(bus,i,address,size):
 for attempt in range(3):
  try:return read_register(bus,i,address,size)
  except RuntimeError:
   if attempt==2:raise
   time.sleep(.03)
def write(bus,i,address,value,size):
 assert i in range(1,6) and address in (21,40,42)
 for attempt in range(3):
  body=bytes([i,size+3,3,address])+value.to_bytes(size,'little')
  bus.reset_input_buffer();bus.write(b'\xff\xff'+body+bytes([(~sum(body))&255]));r=bus.read(6)
  if len(r)==6 and r[:2]==b'\xff\xff' and r[2]==i and r[3]==2 and not r[4] and sum(r[2:])&255==255:return
  if attempt==2:raise RuntimeError(f'Write acknowledgement failed ID{i}: {r.hex()}')
  time.sleep(.03)
def position(bus):return [read(bus,i,56,2) for i in range(1,6)]
import numpy as np
config=json.loads((ROOT/'config/lookdown_prefix.json').read_text())
cal=json.loads((ROOT.parents[1]/'SO101/calibration/robots/so_follower/my_awesome_follower_arm.json').read_text())
names=config['joint_order'];q=config['sampled_arm_degrees']
path=np.array([[int(v*4095/360+(cal[n]['range_min']+cal[n]['range_max'])/2) for n,v in zip(names,row)] for row in q])
assert len(path)==161 and config['automatic_extension_beyond_last_frame'] is False
lo_saved=np.array([cal[n]['range_min'] for n in names]);hi_saved=np.array([cal[n]['range_max'] for n in names])
assert np.maximum(lo_saved-path,path-hi_saved).max()<=5,'Recorded prefix exceeds limits'
path=np.clip(path,lo_saved,hi_saved)
vision_path=ROOT/'dual_model_live/vision_latest.json'
status_path=ROOT/'dual_model_live/controller_latest.json'
def publish(state,**kwargs):
 row={'state':state,'host_monotonic':time.monotonic(),**kwargs}
 tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(status_path)
def check_vision():
 row=json.loads(vision_path.read_text())
 age=time.monotonic()-row['frame_received_monotonic']
 if not 0<=age<=.8:raise RuntimeError('Vision frame stale; stop lowering')
 if row['depth_finite_positive_fraction']<.95 or row['depth_advisory_stop']:
  raise RuntimeError('Depth advisory stop; hold')
 return row
with serial.Serial(PORT,1000000,timeout=.25,exclusive=True) as bus:
 assert [read(bus,i,40,1) for i in range(1,7)]==[1,1,1,1,1,0]
 start=position(bus)
 current_goals=np.array([read(bus,i,42,2) for i in range(1,6)])
 dense_t=np.arange(0,32.001,.02)
 dense=np.stack([np.interp(dense_t,np.arange(161)*.2,path[:,j]) for j in range(5)],axis=1)
 costs=np.max(np.abs(dense-current_goals),axis=1)
 best=int(np.argmin(costs));resume_t=float(dense_t[best]);source_index=resume_t/.2
 assert costs[best]<=12,'Current goal not on recorded prefix'
 assert max(abs(a-b) for a,b in zip(start,current_goals))<=68
 assert source_index<160
 for i,v in enumerate(start,1):
  lo=read(bus,i,9,2);hi=read(bus,i,11,2)
  assert lo-5<=v<=hi+5 and np.all(path[:,i-1]>=lo) and np.all(path[:,i-1]<=hi)
  assert lo<=current_goals[i-1]<=hi
  if v>hi:assert current_goals[i-1]<=v
  if v<lo:assert current_goals[i-1]>=v
  assert read(bus,i,33,1)==0 and read(bus,i,63,1)<60
 check_vision()
 emit('preflight',actual=start,start_target=path[0].tolist(),last_target=path[-1].tolist(),gripper_driven=False,cartesian_height_guard_validated=False)
 old_p={2:16,3:16};enabled=True;previous=current_goals.tolist()
 try:
  publish('bounded_lookdown_resume',torque_enabled=True,source_index=source_index)
  def send(command,index):
   global previous
   check_vision();actual=position(bus)
   if max(abs(a-b) for a,b in zip(actual,previous))>68:raise RuntimeError('Following error over 6 degrees')
   command=[round(float(v)) for v in command]
   assert max(abs(a-b) for a,b in zip(command,previous))<=24,'Command step too large'
   for i,(left,right) in enumerate(zip(previous,command),1):
    if left!=right:write(bus,i,42,right,2)
   previous=command
   if index%10==0:emit('progress',index=index,actual=actual,command=command)
  reason='recorded_prefix_endpoint'
  for k,elapsed in enumerate(np.arange(resume_t,32.001,.05)):
   tick=time.monotonic();vision=check_vision()
   if elapsed>=27 and vision['state']=='tracking' and vision.get('target') is not None:
    reason='target_acquired';emit('target_acquired',target=vision['target']);break
   command=[np.interp(elapsed,np.arange(161)*.2,path[:,j]) for j in range(5)]
   send(command,k)
   publish('bounded_lookdown',torque_enabled=True,elapsed_s=float(elapsed),prefix_endpoint_s=32)
   time.sleep(max(0,.1-(time.monotonic()-tick)))
  emit('lookdown_stopped_holding',reason=reason,actual=position(bus),goal=previous)
  publish('holding',torque_enabled=True,reason=reason)
 except BaseException as exc:
  emit('stopped',error=str(exc));publish('stopped_holding',torque_enabled=enabled,error=str(exc))
 finally:
  if enabled:
   while input('Holding. Support white arm, then type OFF: ').strip()!='OFF':pass
   for i in range(1,6):write(bus,i,40,0,1)
  for i,gain in old_p.items():write(bus,i,21,gain,1)
  state=[read(bus,i,40,1) for i in range(1,7)]
  emit('torque_disabled',state=state);publish('torque_off',torque_enabled=False)
