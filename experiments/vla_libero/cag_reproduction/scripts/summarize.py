"""Audit the executed CAG equation and report selection separately from success."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
from episode_journal import atomic_json
from protocol import score
from mixing import combine
spec=importlib.util.spec_from_file_location('old_analysis',V/'instruction_response/scripts/analyze.py');diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
sha=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main():
 protocol=json.loads((S/'PROTOCOL.json').read_text());control=json.loads((S/'CONTROL_CHECK.json').read_text());assert control['passed']
 for n,h in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 rows=[];allqueries=0;allcalls=0;timings={};max_formula_error=0.
 for variant,scale in [('base',1.),('cag15',1.5)]:
  timings[variant]=[]
  for p in sorted((S/'development'/variant/'episodes').glob('*/*/result.json')):
   r=json.loads(p.read_text());assert r['status']=='complete';target=int(r['mode'][-1]);folder=p.parent
   with np.load(folder/'trace.npz') as f:a={k:f[k] for k in f.files}
   raw=[json.loads(x) for x in (folder/'queries.jsonl').read_text().splitlines()];guided=[json.loads(x) for x in (folder/'guidance/queries.jsonl').read_text().splitlines()];assert len(raw)==len(guided)==60
   rng=np.random.default_rng(r['noise_seed']);applied=[];clip_count=0
   for i,(qr,qg) in enumerate(zip(raw,guided)):
    with np.load(folder/f'inputs/step{i*5:04d}.npz') as f,np.load(folder/f'guidance/query{i:03d}.npz') as g:
     assert str(f['prompt'])==qr['prompt']==r['prompt'];assert np.array_equal(f['noise'],rng.standard_normal((10,32)).astype(np.float32));np.testing.assert_array_equal(f['noise'],g['noise'])
     for n,k in [('image','observation/image'),('wrist','observation/wrist_image'),('state','observation/state')]:assert sha(f[n])==qr['hashes'][k]
     if scale==1:expected=g['conditional'];assert qg['branch_calls']==1
     else:
      expected=combine(g['conditional'],g['unconditional'],scale);assert qg['branch_calls']==2 and qg['nonlanguage_inputs_identical'];assert qg['noise_sha256']==sha(g['noise'])
     max_formula_error=max(max_formula_error,float(abs(expected-g['guided']).max()));np.testing.assert_array_equal(expected,g['guided']);np.testing.assert_array_equal(g['guided'],f['actions']);applied.extend(f['actions'][:5]);clip_count+=int((abs(f['actions'][:5])>1).sum())
    timings[variant].append(qg['seconds']);allqueries+=1;allcalls+=qg['branch_calls']
   np.testing.assert_array_equal(applied,a['applied'][:300]);assert len(a['applied'])==325
   origin=np.asarray(r['origins']);pos=a['object_positions'];lifts=[diag.first_run((pos[:300,i,2]>origin[i,2]+.02)&a['contact'][:300,i]&(a['applied'][:300,6]>0),3) for i in [0,1]]
   result=score(dict(target_in=a['atoms'][:,target],target_free=~a['contact'][:,target],protected_displacement=np.linalg.norm(pos[:,1-target]-origin[1-target],axis=1)),300,'budget');assert result==r['scores'][r['mode']]
   rows.append(dict(variant=variant,task=r['task'],initial=r['initial'],target=target,first_lift=diag.earliest(lifts),first_lift_correct=diag.earliest(lifts)==target,target_contact=bool(a['contact'][:300,target].any()),success=result['selective_success'],valid_window=result['first_valid_step'] is not None,protected_ok=result['protected_untouched'],target_stable=result['target_final_stable'],max_protected_m=result['max_protected_displacement_m'],initial_sha256=r['initial_sha256'],settled_sha256=r['settled_sha256'],outside_unit_executed_components=clip_count,source=str(p.relative_to(ROOT))))
 assert len(rows)==16 and allqueries==960 and allcalls==1440
 pairs=[]
 for t in [0,1]:
  for initial in [0,11]:
   for target in [0,1]:
    rr=[next(r for r in rows if (r['variant'],r['task'],r['initial'],r['target'])==(v,t,initial,target)) for v in ['base','cag15']];a,b=rr
    assert a['initial_sha256']==b['initial_sha256'] and a['settled_sha256']==b['settled_sha256']
    pairs.append(dict(task=t,initial=initial,target=target,base_first=a['first_lift'],cag_first=b['first_lift'],base_success=a['success'],cag_success=b['success']))
 groups={}
 for v in ['base','cag15']:
  rr=[r for r in rows if r['variant']==v];groups[v]={k:sum(r[k] for r in rr) for k in ['first_lift_correct','target_contact','success','valid_window','protected_ok','target_stable']};groups[v].update(n=len(rr),seconds_per_query_median=float(np.median(timings[v])),outside_unit_executed_components=sum(r['outside_unit_executed_components'] for r in rr))
 atomic_json(S/'SUMMARY.json',dict(groups=groups,pairs=pairs,episodes=rows,audit=dict(episodes=16,queries=allqueries,branch_calls=allcalls,control_branch_calls=4,max_formula_error=max_formula_error,source_hashes_verified=True,same_initial_states=True,physical_state_rescore=False),control=control,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
