"""Audit closed-loop actions and report selection separately from final success."""
import hashlib,json,sys,importlib.util,collections
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
from episode_journal import atomic_json
from protocol import score
spec=importlib.util.spec_from_file_location('diag_analysis',V/'instruction_response/scripts/analyze.py');diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
arrhash=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def main(evaluation_root='policy_eval',variants=('base','lora'),initials=(28,29),output='SUMMARY.json',training='lora_pilot'):
    episodes=[];queries=0
    for variant in variants:
        folder=S/evaluation_root/variant;cfg=json.loads((folder/'PROTOCOL.json').read_text())
        assert json.loads((folder/'STATUS.json').read_text())['state']=='complete'
        for p,d in cfg['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==d
        for p in sorted((folder/'episodes').glob('*/*/result.json')):
            r=json.loads(p.read_text());assert r['status']=='complete';assert r['initial'] in initials
            target=int(r['mode'][-1]);rng=np.random.default_rng(r['noise_seed']);actions=[]
            qrows=[json.loads(x) for x in (p.parent/'queries.jsonl').read_text().splitlines()]
            assert len(qrows)==60
            for q in qrows:
                with np.load(p.parent/f'inputs/step{q["step"]:04d}.npz') as f:
                    assert str(f['prompt'])==q['prompt']==r['prompt']
                    assert np.array_equal(f['noise'],rng.standard_normal((10,32)).astype(np.float32))
                    for name,key in [('image','observation/image'),('wrist','observation/wrist_image'),('state','observation/state')]:assert arrhash(f[name])==q['hashes'][key]
                    actions.extend(f['actions'][:5]);queries+=1
            with np.load(p.parent/'trace.npz') as f:a={k:f[k] for k in f.files}
            np.testing.assert_array_equal(actions,a['applied'][:300]);assert len(a['applied'])==325
            origin=np.array(r['origins']);pos=a['object_positions'];lifts=[];moves=[]
            for i in [0,1]:
                lifts.append(diag.first_run((pos[:300,i,2]>origin[i,2]+.02)&a['contact'][:300,i]&(a['applied'][:300,6]>0),3))
                moves.append(diag.first_run(np.linalg.norm(pos[:300,i]-origin[i],axis=1)>.03,5))
                recomputed=score(dict(target_in=a['atoms'][:,i],target_free=~a['contact'][:,i],protected_displacement=np.linalg.norm(pos[:,1-i]-origin[1-i],axis=1)),300,'budget');assert recomputed==r['scores'][f'target{i}']
            first=diag.earliest(lifts);result=r['scores'][r['mode']]
            episodes.append(dict(variant=variant,task=r['task'],initial=r['initial'],target=target,first_lift=first,first_lift_correct=first==target,first_move=diag.earliest(moves),lift_steps=lifts,success=result['selective_success'],valid_window=result['first_valid_step'] is not None,protected_ok=result['protected_untouched'],max_protected_m=result['max_protected_displacement_m'],target_stable=result['target_final_stable'],initial_sha256=r['initial_sha256'],settled_sha256=r['settled_sha256'],source=str(p.relative_to(ROOT))))
    assert len(episodes)==16 and queries==960
    groups={};pairs=[]
    for variant in variants:
        rows=[r for r in episodes if r['variant']==variant];groups[variant]={k:sum(r[k] for r in rows) for k in ['success','first_lift_correct','valid_window','protected_ok','target_stable']};groups[variant]['n']=len(rows)
        groups[variant]['by_task_target']={f'task{t}_target{g}':{k:sum(r[k] for r in rows if r['task']==t and r['target']==g) for k in ['success','first_lift_correct','valid_window']} for t in [0,1] for g in [0,1]}
    for t in [0,1]:
        for i in initials:
            for target in [0,1]:
                a=next(r for r in episodes if (r['variant'],r['task'],r['initial'],r['target'])==('base',t,i,target));b=next(r for r in episodes if (r['variant'],r['task'],r['initial'],r['target'])==(variants[1],t,i,target))
                assert a['initial_sha256']==b['initial_sha256'] and a['settled_sha256']==b['settled_sha256']
                pairs.append(dict(task=t,initial=i,target=target,base_success=a['success'],lora_success=b['success'],base_first=a['first_lift'],lora_first=b['first_lift']))
    probe_rows=[]
    for variant in variants:
        for p in (S/evaluation_root/variant/'probes').glob('*/*/result.json'):
            r=json.loads(p.read_text());q={x['mode']:x for x in r['queries']}
            assert all(x['raw_hashes']==q['target0']['raw_hashes'] and x['noise_sha256']==q['target0']['noise_sha256'] for x in q.values())
            assert q['target0']['transformed']['active_tokens']!=q['target1']['transformed']['active_tokens']
            with np.load(p.parent/'arrays.npz') as f:
                delta=f['target1']-f['target0'];repeat=f['repeat_target0']-f['target0'];assert np.max(abs(repeat))==0.
                probe_rows.append(dict(variant=variant,task=r['task'],initial=r['initial'],step=r['step'],rms=float(np.sqrt(np.mean(delta*delta))),repeat_max=0.))
    assert len(probe_rows)==48
    teacher=json.loads((S/'MANIFEST.json').read_text());data={}
    for split in ['train','validation']:
        rows=[r for r in teacher if r['split']==split];data[split]=dict(total=len(rows),accepted=sum(r['accepted'] for r in rows),cells={f't{t}_g{g}':sum(r['accepted'] for r in rows if r['task']==t and r['target']==g) for t in [0,1] for g in [0,1]})
    resources={}
    for p in (S/'resource_checks').glob('memory_*.jsonl'):
        rows=[json.loads(x) for x in p.read_text().splitlines()];resources[p.stem]=dict(observed_rss_peak_gib=max(r.get('rss_kib',0) for r in rows)/1024**2,observed_cgroup_peak_gib=max(int(r['MemoryCurrent']) if r.get('MemoryCurrent','').isdigit() else 0 for r in rows)/1024**3)
    atomic_json(S/output,dict(probes=probe_rows,groups=groups,pairs=pairs,episodes=episodes,data=data,resources=resources,paired_success_gains=sum(p['lora_success'] and not p['base_success'] for p in pairs),paired_success_losses=sum(p['base_success'] and not p['lora_success'] for p in pairs),audit=dict(episodes=16,policy_queries=queries,common_state_probes=len(probe_rows),extra_probe_inferences=len(probe_rows)*4,identical_paired_initial_states=True,actions_and_scores_checked=True,physical_state_rescore=False),training=json.loads((S/training/'TRAIN_STATUS.json').read_text()),zero_check={k:v for k,v in json.loads((S/'lora_pilot/ZERO_ADAPTER_CHECK.json').read_text()).items() if k not in ['actions','parameter_sha256']}))
    print(json.dumps(groups,indent=2))

if __name__=='__main__':
    main('policy_eval_paired',('base','paired'),(30,31),'SUMMARY_PAIRED.json','lora_paired') if '--paired' in sys.argv else main()
