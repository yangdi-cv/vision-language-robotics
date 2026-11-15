"""Audit inputs/actions/stopping and independently restore simulator scoring states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import imageio_ffmpeg
from protocol import MODES, DISPLACEMENT_LIMIT, MAX_STEPS, SETTLE_STEPS, released
from control import StopRule

S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path.insert(0,str(V/'scripts'))
from episode_journal import atomic_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='eval');parser.add_argument('--physics',action='store_true');parser.add_argument('--sample',type=int);a=parser.parse_args()
    protocol=json.loads((S/'PROTOCOL.json').read_text())
    if a.phase=='eval':
        for name,digest in json.loads((S/'EVALUATION_PREPARED.json').read_text()).items():
            assert sha(S/name)==digest,name
    for name,digest in protocol['sources'].items():
        assert sha(ROOT/name)==digest and sha(S/'archive'/name)==digest,name
    resources=json.loads((S/'RESOURCES.json').read_text())
    assert sha(V/'checkpoints/pi05_libero/download_verified.json')==resources['checkpoint_manifest_sha256']
    request_count=0
    allowed={'select':{'op','image','instruction','objects'},'translate':{'op','image','instruction'},'check':{'op','image','subgoal'}}
    for log in (S/'logs').glob('requests_*.jsonl'):
        for line in log.read_text().splitlines():
            entry=json.loads(line)
            if entry['kind']!='request':continue
            payload=entry['payload'];assert set(payload)==allowed[payload['op']]
            if payload['op']=='select':assert all(len(o)==3 and all(isinstance(v,str) for v in o) for o in payload['objects'])
            request_count+=1
    paths=sorted((S/f'episodes/{a.phase}').glob('*/*/result.json'))
    if a.sample:paths=paths[:a.sample]
    counters=dict(episodes=0,policy_queries=0,decisions=0,video_frames=0,physics_states=0,original_success_continued=0)
    max_position_error=0.;predicate_disagreements=contact_disagreements=0
    final_score_disagreements=[]
    env=None;env_task=None
    try:
        for p in paths:
            r=json.loads(p.read_text())
            if r['status']!='complete':continue
            assert r['protocol_sha256']==sha(S/'PROTOCOL.json')
            root=p.parent;c=r['case'];n=r['decision_step']
            assert r['goal_atoms'][c['target_index']][1]==c['target']
            assert r['goal_atoms'][1-c['target_index']][1]==c['protected']
            with np.load(root/'trace.npz') as f:t={k:f[k] for k in f.files}
            assert len(t['applied'])==n+SETTLE_STEPS
            assert np.array_equal(t['phase'],[0]*n+[1]*SETTLE_STEPS)
            qrows=[json.loads(x) for x in (root/'policy_queries.jsonl').read_text().splitlines()]
            rng=np.random.default_rng(r['noise_seed']);actions=[]
            for q in qrows:
                with np.load(root/f'policy_inputs/step{q["step"]:04d}.npz') as f:inputs={k:f[k] for k in f.files}
                assert inputs['image'].shape==inputs['wrist'].shape==(224,224,3)
                assert inputs['state'].shape==(8,)
                noise=rng.standard_normal((10,32)).astype(np.float32)
                assert np.array_equal(inputs['noise'],noise)
                assert hashlib.sha256(noise.tobytes()).hexdigest()==q['noise_sha256']
                for name,key in [('observation/image','image'),('observation/wrist_image','wrist'),('observation/state','state')]:
                    assert hashlib.sha256(np.ascontiguousarray(inputs[key]).tobytes()).hexdigest()==q['hashes'][name]
                assert str(inputs['instruction'])==r['policy_instruction']==q['instruction']
                if q['step']:
                    np.testing.assert_array_equal(inputs['state'][:3],t['eef'][q['step']-1])
                    np.testing.assert_array_equal(inputs['state'][-2:],t['gripper'][q['step']-1])
                actions.extend(inputs['actions'][:5]);counters['policy_queries']+=1
            np.testing.assert_array_equal(actions,t['applied'][:n])
            hold=np.zeros(7);hold[-1]=t['applied'][n-1,-1]
            np.testing.assert_array_equal(np.tile(hold,(SETTLE_STEPS,1)),t['applied'][n:])
            rule=StopRule(r['mode'],r['timer_threshold'])
            events={e['step']:e for e in r['events'] if e['kind']=='check'}
            for step in range(0,n+1,5):
                if step==MAX_STEPS:break
                if not rule.eligible(step,t['gripper'][step-1] if step else [0,0],t['applied'][step-1] if step else [0,0,0,0,0,0,-1]):
                    continue
                assert step in events
                e=events[step];decision=rule.decide(step,e['reply']);assert decision==e['decision']
                if decision['stop']:assert step==n
                if r['mode'] in ['visual','frozen']:
                    assert e['sent_images']==[e['images'][0],e['images'][0] if r['mode']=='frozen' else e['images'][1]]
                    info=e['reply']['inference'];assert info['revision']=='ebb281ec70b05090aa6165b016eac8ec08e71b17'
                    assert hashlib.sha256(info['prompt'].encode()).hexdigest()==info['prompt_sha256']
                    for name,description in zip(e['sent_images'],info['images'],strict=True):
                        with Image.open(name) as im:assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==description['pixel_sha256']
                else:assert e['reply'] is None and e['sent_images']==[]
                counters['decisions']+=1
            assert r['original_done_used_for_stopping'] is False and r['control_truth_inputs'] is False
            assert (r['stop_reason']=='budget')==(n==MAX_STEPS)
            stable=bool(t['target_in'][-10:].all())
            free=bool(t['target_free'][-10:].all())
            protected=bool(np.linalg.norm(t['object_positions'][:,1-c['target_index']]-np.asarray(r['initial_object_positions'])[1-c['target_index']],axis=1).max()<=DISPLACEMENT_LIMIT)
            assert r['selective_success']==(stable and free and protected)
            assert protected==r['protected_untouched']
            reader=imageio_ffmpeg.read_frames(str(root/'rollout.mp4'));next(reader);frames=sum(1 for _ in reader)
            assert frames==len(t['applied'])+1;counters['video_frames']+=frames
            counters['original_success_continued']+=r['policy_steps_after_original_success']>0
            if a.physics:
                from libero.libero import benchmark,get_libero_path
                from libero.libero.envs import OffScreenRenderEnv
                if env_task!=c['task']:
                    if env is not None:env.close()
                    task=benchmark.get_benchmark_dict()['libero_10']().get_task(c['task'])
                    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
                    env.seed(7);env.reset();env_task=c['task']
                ids=[env.env.obj_body_id[o[0]] for o in c['objects']]
                restored_relations=[];restored_contacts=[];restored_displacements=[]
                for k,state in enumerate(t['sim_state']):
                    env.sim.set_state_from_flattened(state);env.sim.forward()
                    positions=env.sim.data.body_xpos[ids]
                    max_position_error=max(max_position_error,float(np.max(np.abs(positions-t['object_positions'][k]))))
                    relation=bool(env.env._eval_predicate(r['goal_atoms'][c['target_index']]))
                    contact=env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(c['target']).contact_geoms)
                    predicate_disagreements+=relation!=bool(t['target_in'][k])
                    contact_disagreements+=(not contact)!=bool(t['target_free'][k])
                    restored_relations.append(relation);restored_contacts.append(contact)
                    restored_displacements.append(float(np.linalg.norm(positions[1-c['target_index']]-np.asarray(r['initial_object_positions'])[1-c['target_index']])))
                    counters['physics_states']+=1
                restored_success=all(restored_relations[-10:]) and not any(restored_contacts[-10:]) and max(restored_displacements)<=DISPLACEMENT_LIMIT
                if bool(restored_success)!=r['selective_success']:
                    final_score_disagreements.append(str(p.relative_to(S)))
            counters['episodes']+=1
    finally:
        if env is not None:env.close()
    result=dict(**counters,phase=a.phase,expected={'skills':4,'pilot':28,'eval':140}[a.phase],
        max_restored_object_position_error_m=max_position_error,
        restored_predicate_disagreements=int(predicate_disagreements),restored_contact_disagreements=int(contact_disagreements),
        restored_final_score_disagreements=final_score_disagreements,
        vlm_requests_audited_all_phases=request_count,
        physics_checked=a.physics,execution_sources_unchanged=True)
    atomic_json(S/f'VERIFICATION_{a.phase}.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
