"""Bounded base-yaw visual tracking; all other joint goals remain fixed."""
import json,time
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'so101_pickplace/scripts'))
from read_follower import PORT,read_register,serial
ROOT=Path(__file__).resolve().parents[1]
log=ROOT/'motion_logs'/('yaw_tracking_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.jsonl')
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

import statistics
vision_path=ROOT/'dual_model_live/vision_latest.json'
status_path=ROOT/'dual_model_live/controller_latest.json'
def publish(state,**kw):
 row={'state':state,'host_monotonic':time.monotonic(),'lowering_enabled':False,**kw}
 tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(status_path)
def vision():
 r=json.loads(vision_path.read_text())
 if not 0<=time.monotonic()-r['frame_received_monotonic']<=.65:return None
 if r['state']!='tracking' or not r.get('target'):return None
 return r
with serial.Serial(PORT,1000000,timeout=.25,exclusive=True) as bus:
 assert [read(bus,i,40,1) for i in range(1,7)]==[1,1,1,1,1,0]
 goals=[read(bus,i,42,2) for i in range(1,6)]
 assert max(abs(a-b) for a,b in zip(position(bus),goals))<45
 assert all(read(bus,i,63,1)<60 for i in range(1,6))
 assert vision() is not None
 origin=goals[0];current=origin
 low=max(read(bus,1,9,2),origin-114);high=min(read(bus,1,11,2),origin+114)
 def guard():
  actual=position(bus)
  if max(abs(a-b) for a,b in zip(actual,[current]+goals[1:]))>45:raise RuntimeError('Joint following error; holding')
  if any(read(bus,i,42,2)!=goals[i-1] for i in range(2,6)):raise RuntimeError('Other joint goal changed')
 def move(target):
  global current
  assert low<=target<=high
  while current!=target:
   guard()
   if vision() is None:raise RuntimeError('Target lost during direction probe; holding')
   current+=max(-2,min(2,target-current));write(bus,1,42,current,2);time.sleep(.1)
 def sample():
  values=[];last=-1
  for _ in range(30):
   r=vision()
   if r is not None and r['frame']!=last:
    values.append(r['target']['center_px'][0]);last=r['frame']
   if len(values)>=7:break
   time.sleep(.06)
  if len(values)<5 or max(values)-min(values)>10:raise RuntimeError('Target unstable during direction probe; holding')
  return statistics.median(values)
 try:
  emit('preflight',goals=goals,pan_bounds=[low,high],other_joint_goals_fixed=True)
  publish('direction_probe',torque_enabled=True)
  x0=sample();move(origin+34);time.sleep(.5);x1=sample()
  move(origin);time.sleep(.5);x2=sample()
  slope=(x1-(x0+x2)/2)/34
  if abs(x2-x0)>10 or abs(slope)<.12 or abs(slope)>4:raise RuntimeError('Direction probe inconclusive; holding')
  emit('direction_calibrated',x0=x0,x1=x1,x2=x2,pixels_per_tick=slope)
  publish('yaw_tracking',torque_enabled=True,pan_bounds=[low,high],pixels_per_tick=slope)
  last_frame=-1;last_seen=time.monotonic();started=last_seen;last_log=0
  while time.monotonic()-started<300:
   guard();r=vision()
   if r is None:
    publish('target_lost_holding',torque_enabled=True)
    if time.monotonic()-last_seen>3:raise RuntimeError('Target absent for 3 seconds; holding')
    time.sleep(.1);continue
   if r['frame']==last_frame:time.sleep(.05);continue
   last_frame=r['frame'];last_seen=time.monotonic();error=r['target']['center_px'][0]-320
   step=0 if abs(error)<=35 else round(max(-2,min(2,-.2*error/slope)))
   target=max(low,min(high,current+step))
   if target!=current:write(bus,1,42,target,2);current=target
   publish('yaw_tracking',torque_enabled=True,pan_raw=current,error_px=error,pan_limit_reached=current in (low,high))
   if time.monotonic()-last_log>1:
    emit('tracking',pan_raw=current,error_px=error);last_log=time.monotonic()
   time.sleep(.1)
  emit('session_time_limit_holding')
 except BaseException as exc:
  emit('stopped_holding',error=str(exc))
 finally:
  publish('holding',torque_enabled=True)
  while input('Holding. Support white arm, then type OFF: ').strip()!='OFF':pass
  for i in range(1,6):write(bus,i,40,0,1)
  for i in (2,3):write(bus,i,21,16,1)
  emit('torque_disabled',state=[read(bus,i,40,1) for i in range(1,7)])
  publish('torque_off',torque_enabled=False)
