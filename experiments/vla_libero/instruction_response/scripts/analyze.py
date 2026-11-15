"""Separate instruction sensitivity, behavior selection, and task completion."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'selective_execution/scripts'),str(V/'scripts')]
from protocol import score
from episode_journal import atomic_json
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def first_run(mask,length):
    if len(mask)<length:return None
    ids=np.flatnonzero(np.convolve(np.asarray(mask,int),np.ones(length,int),mode='valid')==length)
    return int(ids[0]+1) if len(ids) else None

def earliest(times):
    available=[(t,i) for i,t in enumerate(times) if t is not None]
    if not available:return None
    t=min(x[0] for x in available);ids=[i for k,i in available if k==t]
    return ids[0] if len(ids)==1 else 'tie'

def main():
    frozen=json.loads((S/'PROTOCOL.json').read_text())
    for path,h in frozen['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    probes=[];episodes=[];inputs=0
    for p in sorted((S/'probes').glob('*/*/result.json')):
        r=json.loads(p.read_text());q={x['mode']:x for x in r['queries']}
        assert len(q)==4
        for m,row in q.items():
            assert row['noise_sha256']==q['target0']['noise_sha256']
            assert row['raw_hashes']==q['target0']['raw_hashes']
            assert row['transformed']['nonlanguage_hashes']==q['target0']['transformed']['nonlanguage_hashes']
            assert len(row['transformed']['tokens'])==len(row['transformed']['mask'])==200
        assert q['target0']['transformed']['active_tokens']==q['repeat_target0']['transformed']['active_tokens']
        assert q['target0']['transformed']['active_tokens']!=q['target1']['transformed']['active_tokens']
        with np.load(p.parent/'arrays.npz') as f:
            assert sha(f['noise'])==q['target0']['noise_sha256']
            for name,key in [('image','observation/image'),('wrist','observation/wrist_image'),('state','observation/state')]:assert sha(f[name])==q['target0']['raw_hashes'][key]
            metrics={}
            for m in ['target1','original','repeat_target0']:
                d=f[m]-f['target0'];metrics[m]=dict(rms_all=float(np.sqrt(np.mean(d*d))),rms_translation_first5=float(np.sqrt(np.mean(d[:5,:3]**2))),max_all=float(np.max(abs(d))))
        probes.append(dict(task=r['task'],initial=r['initial'],step=r['step'],metrics=metrics,active_token_lengths={m:sum(row['transformed']['mask']) for m,row in q.items()}))
    for p in sorted((S/'episodes').glob('*/*/result.json')):
        r=json.loads(p.read_text());assert r['status']=='complete'
        with np.load(p.parent/'trace.npz') as f:a={k:f[k] for k in f.files}
        assert len(a['applied'])==325
        rng=np.random.default_rng(r['noise_seed']);actions=[]
        queries=[json.loads(x) for x in (p.parent/'queries.jsonl').read_text().splitlines()]
        assert len(queries)==60
        for q in queries:
            with np.load(p.parent/f'inputs/step{q["step"]:04d}.npz') as f:
                assert str(f['prompt'])==r['prompt']==q['prompt']
                assert np.array_equal(f['noise'],rng.standard_normal((10,32)).astype(np.float32))
                for name,key in [('image','observation/image'),('wrist','observation/wrist_image'),('state','observation/state')]:assert sha(f[name])==q['hashes'][key]
                actions.extend(f['actions'][:5]);inputs+=1
        np.testing.assert_array_equal(actions,a['applied'][:300])
        hold=np.zeros(7);hold[-1]=a['applied'][299,-1];np.testing.assert_array_equal(a['applied'][300:],np.tile(hold,(25,1)))
        positions=a['object_positions'];origins=np.asarray(r['origins']);distance=np.linalg.norm(positions[:300]-origins,axis=2)
        move=[first_run(distance[:,i]>.03,5) for i in [0,1]]
        lift=[first_run((positions[:300,i,2]-origins[i,2]>.02)&a['contact'][:300,i]&(a['applied'][:300,6]>0),3) for i in [0,1]]
        contact=[first_run(a['contact'][:300,i]&(a['applied'][:300,6]>0),3) for i in [0,1]]
        for i in [0,1]:
            recompute=score(dict(target_in=a['atoms'][:,i],target_free=~a['contact'][:,i],protected_displacement=np.linalg.norm(positions[:,1-i]-origins[1-i],axis=1)),300,'budget')
            assert recompute==r['scores'][f'target{i}']
        target=int(r['mode'][-1]) if r['mode'].startswith('target') else None
        episodes.append(dict(case=p.parent.parent.name,mode=r['mode'],task=r['task'],initial=r['initial'],initial_sha256=r['initial_sha256'],settled_sha256=r['settled_sha256'],first_move=earliest(move),first_lift=earliest(lift),first_contact=earliest(contact),move_steps=move,lift_steps=lift,contact_steps=contact,requested_target=target,requested_first_lift=(earliest(lift)==target if target is not None else None),scores=r['scores'],original_success_observed=r['original_success_observed'],source=str(p.relative_to(ROOT))))
    assert len(probes)==24 and len(episodes)==12
    pairs=[]
    for task in [0,1]:
        for init in [0,11]:
            group=[r for r in episodes if r['task']==task and r['initial']==init]
            assert len(group)==3 and len({r['settled_sha256'] for r in group})==len({r['initial_sha256'] for r in group})==1
            a=next(r for r in group if r['mode']=='target0');b=next(r for r in group if r['mode']=='target1');o=next(r for r in group if r['mode']=='original')
            pairs.append(dict(task=task,initial=init,first_lifts={r['mode']:r['first_lift'] for r in group},selection_changes=a['first_lift']!=b['first_lift'],both_requested_first=a['requested_first_lift'] and b['requested_first_lift'],original_first=o['first_lift']))
    modes={}
    for m in ['target0','target1','original']:
        group=[r for r in episodes if r['mode']==m];modes[m]=dict(n=len(group),first_lift_matches=(sum(r['requested_first_lift'] for r in group) if m!='original' else None),first_lift_absent=sum(r['first_lift'] is None for r in group),original_success=sum(r['original_success_observed'] for r in group),target0_valid_window=sum(r['scores']['target0']['first_valid_step'] is not None for r in group),target1_valid_window=sum(r['scores']['target1']['first_valid_step'] is not None for r in group),requested_final_success=(sum(r['scores'][m]['selective_success'] for r in group) if m!='original' else None))
    summary=dict(episodes=episodes,probes=probes,pairs=pairs,modes=modes,audit=dict(frozen_sources=len(frozen['sources']),episodes=12,policy_inputs=inputs,common_state_probes=len(probes),inference_requests_at_probes=len(probes)*4,all_same_observation_and_noise=True,all_distinct_target_tokens=True,self_repeat_max_difference=max(r['metrics']['repeat_target0']['max_all'] for r in probes),physical_state_rescore=False),initial_probe_target_swap_rms=[r['metrics']['target1']['rms_all'] for r in probes if r['step']==0],target_swap_rms_median=float(np.median([r['metrics']['target1']['rms_all'] for r in probes])))
    atomic_json(S/'SUMMARY.json',summary);print(json.dumps(dict(audit=summary['audit'],modes=modes,pairs=pairs,initial_probe_target_swap_rms=summary['initial_probe_target_swap_rms'],target_swap_rms_median=summary['target_swap_rms_median']),indent=2))

if __name__=='__main__':main()
