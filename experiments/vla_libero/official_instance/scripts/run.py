"""Fresh original-task pi0.5 instance. Native policy actions, no high-level overrides."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'skill_bootstrap/scripts')]
import evaluate as ev
from runtime import load_policy
from episode_journal import atomic_json,append_jsonl
class Journal:
 def __init__(self,p,out,info):self.p=p;self.out=out;self.info=info;self.index=0
 def infer(self,obs,*,noise):
  t=time.monotonic();r=self.p.infer(obs,noise=noise);self.out.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(self.out/f'query{self.index:03d}.npz',image=obs['observation/image'],wrist=obs['observation/wrist_image'],state=obs['observation/state'],noise=noise,actions=r['actions'])
  append_jsonl(self.out/'queries.jsonl',dict(query=self.index,prompt=obs['prompt'],seconds=time.monotonic()-t))
  Image.fromarray(obs['observation/image']).resize((672,672)).save(S/'latest.jpg')
  atomic_json(S/'progress.json',dict(state='running',**self.info,query=self.index,prompt=obs['prompt'],latest_action=np.asarray(r['actions'][0]).tolist(),time=time.time()))
  self.index+=1;return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',default='fresh_original');ap.add_argument('--tasks',type=int,nargs='+',default=[0,1,9]);ap.add_argument('--initials',type=int,nargs='+',default=[32,33]);a=ap.parse_args();assert a.tag.replace('_','').replace('-','').isalnum()
 out=S/'runs'/a.tag;out.mkdir(parents=True,exist_ok=True);assert not (out/'PROTOCOL.json').exists(),'Use a new tag'
 files=[Path(__file__),V/'scripts/evaluate.py',V/'skill_bootstrap/scripts/runtime.py',V/'skill_bootstrap/scripts/bounded_restore.py']
 atomic_json(out/'PROTOCOL.json',dict(suite='libero_10',tasks=a.tasks,initials=a.initials,checkpoint='gs://openpi-assets/checkpoints/pi05_libero',native_original_prompts=True,no_training=True,no_oracle_action=True,no_hardware=True,code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
 policy=load_policy();results=[]
 for task in a.tasks:
  for initial in a.initials:
   folder=out/f'task{task:02d}_init{initial:02d}';folder.mkdir()
   row=ev.trial(Journal(policy,folder/'query_journal',dict(task=task,initial=initial,tag=a.tag)),'libero_10',task,initial,'baseline',folder)
   results.append(row);atomic_json(out/'STATUS.json',dict(state='running',complete=len(results),expected=len(a.tasks)*len(a.initials),successes=sum(r['success'] for r in results)))
 atomic_json(out/'STATUS.json',dict(state='complete',complete=len(results),expected=len(a.tasks)*len(a.initials),successes=sum(r['success'] for r in results)))
 atomic_json(S/'progress.json',dict(state='complete',tag=a.tag,complete=len(results),successes=sum(r['success'] for r in results),time=time.time()))
if __name__=='__main__':main()
