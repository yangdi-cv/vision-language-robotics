"""Paired targets and reset-only position swaps; privileged teacher data, not policy scores."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from layout import S,V,swap_at_reset,reference
ROOT=V.parents[1]
sys.path[:0]=[str(V/'skill_bootstrap/scripts'),str(V/'scripts')]
import teacher
import evaluate as ev
from episode_journal import atomic_json
from libero.libero import benchmark,get_libero_path
import libero.libero.envs as envs
OriginalEnv=envs.OffScreenRenderEnv

class Capture:
 def __init__(self,task,layout,folder,*args,**kwargs):self.inner=OriginalEnv(*args,**kwargs);self.task=task;self.layout=layout;self.folder=folder;self.n=0
 def __getattr__(self,k):return getattr(self.inner,k)
 def set_init_state(self,initial):
  self.n=0;self.inner.set_init_state(initial);swap_at_reset(self.inner,self.task,self.layout)
  return self.inner.env._get_observations()
 def step(self,action):
  r=self.inner.step(action);self.n+=1
  if self.n==10:
   asset=reference(self.inner,r[0],self.task);np.savez_compressed(self.folder/'reference.npz',**asset)
  return r

def main():
 assert json.loads((S/'preflight/CHECK.json').read_text())['passed']
 files=[Path(__file__),S/'scripts/layout.py',V/'skill_bootstrap/scripts/teacher.py']
 spec=dict(new_train_initials=[36,37,38,39],new_validation_initials=[40],old_train_initials=list(range(20,26)),old_validation_initials=[26,27],held_out_initials=[41,42],layouts=['normal','swap'],tasks=[0,1],targets=[0,1],new_teacher_candidates=40,acceptance='both goals must pass original selective score within same layout; accepted pairs only',oracle_boxes=True,oracle_actions_only_teacher=True,swap_only_at_reset=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 assert not (S/'DATA_PROTOCOL.json').exists(),'Refuse overwrite';atomic_json(S/'DATA_PROTOCOL.json',spec)
 manifest=[];teacher.S=S
 for split,initials in [('train',[36,37,38,39]),('validation',[40])]:
  for task in [0,1]:
   for initial in initials:
    for layout in ['normal','swap']:
     tag=split+'_'+layout
     for target in [0,1]:
      folder=S/f'episodes/{tag}/task{task:02d}_target{target}_init{initial:02d}';folder.mkdir(parents=True)
      envs.OffScreenRenderEnv=lambda *a,**kw:Capture(task,layout,folder,*a,**kw)
      try:r=teacher.trial(task,initial,target,tag)
      finally:envs.OffScreenRenderEnv=OriginalEnv
      manifest.append(dict(split=split,task=task,initial=initial,layout=layout,target=target,accepted=r['selective_success'],trajectory=str((folder/'trajectory.npz').relative_to(ROOT)),reference=str((folder/'reference.npz').relative_to(ROOT)),origin='new_reset_swap_teacher'))
      atomic_json(S/'DATA_STATUS.json',dict(state='running',new_completed=len(manifest),expected=40));atomic_json(S/'MANIFEST_CANDIDATES.json',manifest)
 # Reuse accepted original trajectories, with separately checked initial reference images.
 old=json.loads((V/'skill_bootstrap/MANIFEST.json').read_text());refs=S/'old_references';refs.mkdir()
 for task,initial in sorted({(r['task'],r['initial']) for r in old if r['accepted']}):
  suite=benchmark.get_benchmark_dict()['libero_10']();t=suite.get_task(task);env=OriginalEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/t.problem_folder/t.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
  env.seed(7);env.reset();obs=env.set_init_state(np.asarray(ev.initial_states(suite,task)[initial]))
  for _ in range(10):obs,_,_,_=env.step(ev.DUMMY.tolist())
  asset=reference(env,obs,task);path=refs/f'task{task}_init{initial}.npz';np.savez_compressed(path,**asset);env.close()
  for r in old:
   if r['accepted'] and (r['task'],r['initial'])==(task,initial):
    with np.load(ROOT/r['trajectory']) as f:err=float(np.mean(abs(f['image'][0].astype(float)-asset['image'].astype(float))))
    assert err<3.,err
    manifest.append(dict(split=r['split'],task=task,initial=initial,layout='normal',target=r['target'],accepted=True,trajectory=r['trajectory'],reference=str(path.relative_to(ROOT)),origin='old_teacher',reference_alignment_mae=err))
 pairs=[]
 keys=sorted({(r['split'],r['task'],r['initial'],r['layout']) for r in manifest})
 for split,task,initial,layout in keys:
  rows=[r for r in manifest if (r['split'],r['task'],r['initial'],r['layout'])==(split,task,initial,layout)]
  if {r['target'] for r in rows if r['accepted']}!={0,1}:continue
  rows.sort(key=lambda r:r['target'])
  arrays=[]
  for r in rows:
   with np.load(ROOT/r['trajectory']) as f:arrays.append({k:f[k] for k in ['image','wrist','state','action']})
  n=min(len(x['action']) for x in arrays);diff=np.max(abs(arrays[0]['action'][:n]-arrays[1]['action'][:n]),axis=1);ii=np.flatnonzero(diff>1e-5);assert len(ii)
  branch=int(ii[0]);assert all(np.array_equal(arrays[0][k][branch],arrays[1][k][branch]) for k in ['image','wrist','state'])
  with np.load(ROOT/rows[0]['reference']) as a,np.load(ROOT/rows[1]['reference']) as b:assert np.array_equal(a['image'],b['image']) and np.array_equal(a['boxes'],b['boxes'])
  pairs.append(dict(split=split,task=task,initial=initial,layout=layout,branch=branch,rows=rows))
 assert all(any(p['split']==split and p['task']==task and p['layout']==layout for p in pairs) for split in ['train','validation'] for task in [0,1] for layout in ['normal','swap'])
 atomic_json(S/'MANIFEST_CANDIDATES.json',manifest);atomic_json(S/'PAIRS.json',pairs);atomic_json(S/'DATA_STATUS.json',dict(state='complete',new_completed=40,all_candidates=len(manifest),accepted_pairs=len(pairs),train_pairs=sum(p['split']=='train' for p in pairs),validation_pairs=sum(p['split']=='validation' for p in pairs)))
if __name__=='__main__':main()
