"""Verify actual inputs/actions and compare three frozen variants on paired layouts."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
from episode_journal import atomic_json
from protocol import score
from visual_prompt import GENERIC,mark
spec=importlib.util.spec_from_file_location('old_analysis',V/'instruction_response/scripts/analyze.py');diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 rows=[];total=0;probes=[]
 for variant in ['base','text','goal']:
  folder=S/'evaluation'/variant;protocol=json.loads((folder/'PROTOCOL.json').read_text());assert json.loads((folder/'STATUS.json').read_text())['state']=='complete'
  for n,h in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
  for layout in ['normal','swap']:
   for p in sorted((folder/layout/'episodes').glob('*/*/result.json')):
    r=json.loads(p.read_text());assert r['status']=='complete';target=int(r['mode'][-1]);origin=np.array(r['origins'])
    with np.load(p.parent/'trace.npz') as f:a={k:f[k] for k in f.files}
    with np.load(p.parent/'reference.npz') as f:goal=mark(f['image'],f['boxes'][target])
    rng=np.random.default_rng(r['noise_seed']);actions=[];input_hashes=[]
    queries=[json.loads(x) for x in (p.parent/'queries.jsonl').read_text().splitlines()];assert len(queries)==60
    for i,q in enumerate(queries):
     with np.load(p.parent/f'inputs/step{i*5:04d}.npz') as old,np.load(p.parent/f'actual_inputs/query{i:03d}.npz') as f:
      assert np.array_equal(f['noise'],rng.standard_normal((10,32)).astype(np.float32));np.testing.assert_array_equal(old['noise'],f['noise']);np.testing.assert_array_equal(old['actions'],f['actions'])
      for k in ['image','wrist','state']:np.testing.assert_array_equal(old[k],f[k])
      if variant=='goal':assert bool(f['goal_mask']) and str(f['prompt'])==GENERIC;np.testing.assert_array_equal(goal,f['goal_ref'])
      else:assert not bool(f['goal_mask']) and str(f['prompt'])==r['prompt']
      actions.extend(f['actions'][:5]);total+=1
      if i==0:input_hashes={k:sha(f[k]) for k in ['image','wrist','state','noise']}
    np.testing.assert_array_equal(actions,a['applied'][:300]);assert len(a['applied'])==325
    lifts=[diag.first_run((a['object_positions'][:300,i,2]>origin[i,2]+.02)&a['contact'][:300,i]&(a['applied'][:300,6]>0),3) for i in [0,1]]
    sc=score(dict(target_in=a['atoms'][:,target],target_free=~a['contact'][:,target],protected_displacement=np.linalg.norm(a['object_positions'][:,1-target]-origin[1-target],axis=1)),300,'budget');assert sc==r['scores'][r['mode']]
    rows.append(dict(variant=variant,task=r['task'],layout=layout,initial=r['initial'],target=target,first_lift=diag.earliest(lifts),first_lift_correct=diag.earliest(lifts)==target,success=sc['selective_success'],valid_window=sc['first_valid_step'] is not None,protected_ok=sc['protected_untouched'],target_stable=sc['target_final_stable'],settled_sha256=r['settled_sha256'],initial_input_hashes=input_hashes,source=str(p.relative_to(ROOT))))
    if variant=='goal':
     with np.load(p.parent/'box_counterfactual.npz') as f:probes.append(dict(task=r['task'],layout=layout,target=target,action_rms=float(np.sqrt(np.mean((f['requested']-f['other_box'])**2)))))
 assert len(rows)==24 and total==1440 and len(probes)==8
 groups={}
 for v in ['base','text','goal']:
  rr=[r for r in rows if r['variant']==v];groups[v]={k:sum(r[k] for r in rr) for k in ['first_lift_correct','success','valid_window','protected_ok','target_stable']};groups[v]['n']=len(rr);groups[v]['by_layout']={l:{k:sum(r[k] for r in rr if r['layout']==l) for k in ['first_lift_correct','success']} for l in ['normal','swap']}
 for t in [0,1]:
  for l in ['normal','swap']:
   for target in [0,1]:
    rr=[r for r in rows if (r['task'],r['layout'],r['target'])==(t,l,target)];assert len(rr)==3 and len({r['settled_sha256'] for r in rr})==1;assert all(r['initial_input_hashes']==rr[0]['initial_input_hashes'] for r in rr)
 atomic_json(S/'SUMMARY.json',dict(groups=groups,episodes=rows,box_only_probes=probes,audit=dict(episodes=24,queries=1440,extra_box_probe_queries=8,frozen_sources=True,identical_paired_initial_inputs=True,actual_action_chunks_verified=True,physical_rescore=False)))
 print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
