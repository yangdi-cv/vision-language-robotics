"""Resume the existing pi0.5 skill from the physically reached precontact pose.
No arm reset/unfold on handover. Official success is shadow-only during execution.
"""
from pathlib import Path
import argparse,json,socketserver,sys,traceback
import numpy as np
from sim_precontact import Engine as PrecontactEngine,D,ROOT
sys.path.insert(0,str(D.parent/'scripts'))
from durable_log import atomic_json

class Engine(PrecontactEngine):
 def request(self,q):
  if q['op']=='resume_policy':
   from sim_server_qwen import load_policy_bounded
   before=self.arm_state();self.phase='loading_policy'
   self.policy,info=load_policy_bounded()
   self.s.joint_mode=False;self.s.mode='manip';self.s.hold=self.s.data.qpos[self.s.joints].copy();self.s.robot.controller.reset_goal()
   self.predictions=[];self.applied=[];self.queue.clear();self.policy_steps=0;self.phase='manipulation'
   after=self.arm_state();assert np.array_equal(before['joint_positions'],after['joint_positions'])
   atomic_json(self.out/'policy_handover.json',{'before':before,'after':after,'no_state_reset':True,'bounded_restore':info})
   return self.state()
  if q['op']=='policy_tick':
   if self.phase!='manipulation':raise ValueError(self.phase)
   if not self.queue:
    import evaluate as ev
    inputs=ev.make_input(self.s.obs,self.instruction);directory=self.out/'policy_inputs';directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/f'query{len(self.predictions):04d}.npz',image=inputs['observation/image'],wrist=inputs['observation/wrist_image'],state=inputs['observation/state'],instruction=inputs['prompt'])
    actions=np.asarray(self.policy.infer(inputs,noise=self.rng.standard_normal((10,32)).astype(np.float32))['actions'])
    if actions.shape!=(10,7) or not np.isfinite(actions).all():raise ValueError('Invalid model actions')
    self.predictions.append(actions);self.queue.extend(actions[:5])
   action=self.queue.popleft();self.s.step(action);self.applied.append(action.copy());self.policy_steps+=1
   # Official predicates are recorded but never control policy stopping.
   self.journal.append('policy_step',policy_step=self.policy_steps,shadow_official_success=bool(self.s.e.check_success()),**self.arm_state())
   return self.state()
  return super().request(q)

def main():
 p=argparse.ArgumentParser();p.add_argument('--socket',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--task',type=int,default=3);p.add_argument('--initial',type=int,default=0);p.add_argument('--start-docked',action='store_true');p.add_argument('--variant',default='nominal');a=p.parse_args();assert not a.socket.exists();e=Engine(a)
 class Handler(socketserver.StreamRequestHandler):
  def handle(self):
   try:r=e.request(json.loads(self.rfile.readline(16*1024*1024)))
   except Exception:r={'ok':False,'error':traceback.format_exc()}
   self.wfile.write((json.dumps(r)+'\n').encode())
 try:
  with socketserver.UnixStreamServer(str(a.socket),Handler) as server:print('READY',flush=True);server.serve_forever()
 except KeyboardInterrupt:pass
 finally:e.writer.close();e.s.close();a.socket.unlink(missing_ok=True)
if __name__=='__main__':main()
