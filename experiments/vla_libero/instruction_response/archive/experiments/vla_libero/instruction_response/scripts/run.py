"""Bounded target-swap diagnostic using the existing official policy interface."""
import copy,hashlib,json,sys,time,traceback,shutil
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
sys.path[:0]=[str(V/'scripts'),str(V/'selective_execution/scripts')]
import evaluate as ev
from episode_journal import atomic_json,append_jsonl
from protocol import OBJECTS,skill,score
from run_hierarchical import upright

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arrhash=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def freeze():
    sources=[Path(__file__),V/'scripts/evaluate.py',V/'scripts/episode_journal.py',V/'selective_execution/scripts/protocol.py',V/'scripts/run_hierarchical.py',V/'third_party/openpi/src/openpi/policies/policy.py',V/'third_party/openpi/src/openpi/policies/policy_config.py',V/'third_party/openpi/src/openpi/training/config.py']
    value=dict(purpose='Same observation/noise target-swap response; then matched closed-loop rollout. No new model or training.',tasks=[0,1],initials=[0,11],modes=['target0','target1','original'],episodes=12,policy_steps=300,hold_steps=25,execute_first=5,horizon=10,denoising_steps=10,probe_steps=[0,50,100,150,200,250],probe_anchor='target0 rollout only, all four requests use identical observations and explicit noise',probe_order=['target0','target1','original','repeat_target0'],noise_seed='930000+task*100+init; no target/method term',first_move='first object with center displacement >30mm for five consecutive simulated steps; proxy, not a grasp label',first_lift='center rises >20mm, positive gripper command AND gripper contact for three consecutive simulated steps; proxy',expansion='No extra episode batch in this diagnostic. Decide interface repair versus skill adaptation from saved inputs and behavior.',sources={str(p.relative_to(ROOT)):sha(p) for p in sources},resource_manifest_sha256=sha(V/'checkpoints/pi05_libero/download_verified.json'),official_sources=json.loads((V/'third_party/sources.json').read_text()),no_hardware=True,no_training=True)
    p=S/'PROTOCOL.json'
    if p.exists():assert json.loads(p.read_text())==value,'Frozen protocol changed'
    else:
        atomic_json(p,value)
        for name in value['sources']:
            dest=S/'archive'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    return sha(p)

def tokenize(policy,element):
    x=policy._input_transform(copy.deepcopy(element))
    tokens=np.asarray(x['tokenized_prompt']);mask=np.asarray(x['tokenized_prompt_mask'])
    return dict(tokens=tokens.tolist(),mask=mask.tolist(),active_tokens=tokens[mask.astype(bool)].tolist(),nonlanguage_hashes={k:arrhash(v) for k,v in x.items() if k not in ['tokenized_prompt','tokenized_prompt_mask'] and isinstance(v,np.ndarray)})

def probes(policy,obs,prompts,noise,task,init,step):
    folder=S/f'probes/task{task:02d}_init{init:02d}/step{step:04d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise RuntimeError('Refuse probe overwrite')
    items=[];outputs={}
    for mode in ['target0','target1','original','repeat_target0']:
        text=prompts['target0' if mode=='repeat_target0' else mode];element=ev.make_input(obs,text)
        transformed=tokenize(policy,element)
        t=time.monotonic();actions=np.asarray(policy.infer(element,noise=noise)['actions'],np.float64)
        assert actions.shape==(10,7) and np.isfinite(actions).all()
        outputs[mode]=actions
        row=dict(mode=mode,prompt=text,noise_sha256=arrhash(noise),raw_hashes={k:arrhash(v) for k,v in element.items() if k!='prompt'},transformed=transformed,seconds=time.monotonic()-t)
        items.append(row)
    np.savez_compressed(folder/'arrays.npz',noise=noise,image=element['observation/image'],wrist=element['observation/wrist_image'],state=element['observation/state'],**outputs)
    atomic_json(folder/'result.json',dict(task=task,initial=init,step=step,anchor='target0',queries=items))

def rollout(policy,task_id,init,mode,protocol_sha):
    import imageio.v2 as imageio
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(task_id)
    prompts={f'target{i}':skill(o[1]) for i,o in enumerate(OBJECTS[task_id])};prompts['original']=task.language
    folder=S/f'episodes/task{task_id:02d}_init{init:02d}/{mode}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise RuntimeError('Refuse episode overwrite: '+str(folder))
    (folder/'inputs').mkdir()
    seed=930000+task_id*100+init;rng=np.random.default_rng(seed)
    record=dict(status='running',task=task_id,initial=init,mode=mode,prompt=prompts[mode],all_prompts=prompts,protocol_sha256=protocol_sha,noise_seed=seed,original_done_used_for_stopping=False,steps=0)
    atomic_json(folder/'result.json',record);env=writer=None;rows=[];start=time.monotonic()
    try:
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
        initial=np.asarray(ev.initial_states(suite,task_id)[init]);env.seed(7);env.reset();obs=env.set_init_state(initial)
        for _ in range(10):obs,_,_,_=env.step(ev.DUMMY.tolist())
        record['initial_sha256']=arrhash(initial);record['settled_sha256']=arrhash(env.sim.get_state().flatten())
        ids=[env.env.obj_body_id[o[0]] for o in OBJECTS[task_id]];origins=env.sim.data.body_xpos[ids].copy()
        record['origins']=origins.tolist();record['goal_atoms']=env.env.parsed_problem['goal_state']
        atomic_json(folder/'result.json',record)
        writer=imageio.get_writer(folder/'rollout.mp4',fps=20,codec='libx264',quality=7,macro_block_size=16);writer.append_data(upright(obs))
        def apply(action,phase):
            nonlocal obs
            obs,_,done,_=env.step(action.tolist());positions=env.sim.data.body_xpos[ids].copy()
            contact=np.array([env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(o[0]).contact_geoms) for o in OBJECTS[task_id]],bool)
            atoms=np.array([bool(env.env._eval_predicate(a)) for a in record['goal_atoms']])
            r=dict(applied=action.copy(),phase=phase,object_positions=positions,contact=contact,atoms=atoms,gripper=obs['robot0_gripper_qpos'].copy(),eef=obs['robot0_eef_pos'].copy(),original_done=bool(done),sim_state=env.sim.get_state().flatten().copy())
            rows.append(r);writer.append_data(upright(obs));append_jsonl(folder/'trace.jsonl',dict(step=len(rows),**r))
        for step in range(0,300,5):
            noise=rng.standard_normal((10,32)).astype(np.float32)
            if mode=='target0' and step in [0,50,100,150,200,250]:probes(policy,obs,prompts,noise,task_id,init,step)
            element=ev.make_input(obs,prompts[mode]);actions=np.asarray(policy.infer(element,noise=noise)['actions'],np.float64)
            assert actions.shape==(10,7) and np.isfinite(actions).all()
            np.savez_compressed(folder/f'inputs/step{step:04d}.npz',image=element['observation/image'],wrist=element['observation/wrist_image'],state=element['observation/state'],noise=noise,actions=actions,prompt=prompts[mode])
            append_jsonl(folder/'queries.jsonl',dict(step=step,prompt=prompts[mode],hashes={k:arrhash(v) for k,v in element.items() if k!='prompt'},noise_sha256=arrhash(noise)))
            for action in actions[:5]:apply(action,0)
            record['steps']=len(rows);atomic_json(folder/'result.json',record)
        hold=np.zeros(7);hold[-1]=rows[-1]['applied'][-1]
        for _ in range(25):apply(hold,1)
        arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(folder/'trace.npz',**arrays)
        scores={}
        for i in [0,1]:
            a=dict(target_in=arrays['atoms'][:,i],target_free=~arrays['contact'][:,i],protected_displacement=np.linalg.norm(arrays['object_positions'][:,1-i]-origins[1-i],axis=1))
            scores[f'target{i}']=score(a,300,'budget')
        record.update(status='complete',steps=len(rows),scores=scores,original_success_observed=bool(arrays['original_done'].any()),wall_seconds=time.monotonic()-start)
        atomic_json(folder/'result.json',record);return record
    except BaseException as exc:
        if rows:np.savez_compressed(folder/'partial_trace.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        record.update(status='error',error=repr(exc),traceback=traceback.format_exc());atomic_json(folder/'result.json',record);raise
    finally:
        if writer is not None:writer.close()
        if env is not None:env.close()

def main():
    version=freeze()
    from openpi.training import config
    from openpi.policies import policy_config
    policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),V/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
    rows=[]
    for task in [0,1]:
        for init in [0,11]:
            for mode in ['target0','target1','original']:
                r=rollout(policy,task,init,mode,version);rows.append(r)
                atomic_json(S/'STATUS.json',dict(state='running',episodes=len(rows),expected=12,latest=[task,init,mode]));print(json.dumps(dict(task=task,init=init,mode=mode,scores=r['scores'])),flush=True)
    atomic_json(S/'STATUS.json',dict(state='complete',episodes=len(rows),probes=24));print('COMPLETE',flush=True)

if __name__=='__main__':main()
