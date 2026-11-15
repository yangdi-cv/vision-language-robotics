"""Selective execution in original LIBERO scenes, with no oracle success stop."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

S = Path(__file__).resolve().parents[1]
V = S.parent
ROOT = V.parents[1]
sys.path[:0] = [str(V/'scripts'), str(ROOT/'perception/qwen_scene')]
from episode_journal import EpisodeJournal, append_jsonl, atomic_json
from worker_client import read_json_line, close_json_worker
import evaluate as ev
from run_hierarchical import upright, monitor_view, save_png
from protocol import MODES, OBJECTS, MAX_STEPS, SETTLE_STEPS, case, skill, calibrate, score, released
from control import StopRule


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Worker:
    def __init__(self):
        self.log = open(S/f'logs/worker_{time.time_ns()}.log','w')
        self.proc = subprocess.Popen([str(ROOT/'.venvs/qwen-scene/bin/python'),str(S/'scripts/worker.py')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,start_new_session=True)
        self.ready = read_json_line(self.proc,timeout=240)
        self.requests = S/f'logs/requests_{time.time_ns()}.jsonl'
    def request(self, payload):
        append_jsonl(self.requests,dict(kind='request',payload=payload))
        self.proc.stdin.write(json.dumps(payload,ensure_ascii=False)+'\n'); self.proc.stdin.flush()
        reply = read_json_line(self.proc,timeout=120)
        append_jsonl(self.requests,dict(kind='reply',reply=reply))
        return reply
    def close(self):
        close_json_worker(self.proc);self.log.close()


def freeze():
    sources = [*sorted((S/'scripts').glob('*.py')),
        V/'scripts/evaluate.py',V/'scripts/run_hierarchical.py',V/'scripts/qwen_planner.py',
        V/'scripts/episode_journal.py',ROOT/'perception/qwen_scene/engine.py',ROOT/'perception/qwen_scene/worker_client.py']
    # Execution sources only: reporting and analysis are versioned separately at packaging.
    sources = [p for p in sources if p.name in ['run.py','worker.py','control.py','protocol.py','evaluate.py','run_hierarchical.py','qwen_planner.py','episode_journal.py','engine.py','worker_client.py']]
    value = dict(
        modes=list(MODES), tasks=[0,1], target_indices=[0,1],
        development_initials=[0], evaluation_initials=[3,4,5,6,7],
        max_policy_steps=MAX_STEPS, settle_steps=SETTLE_STEPS,
        protected_position_limit_m=.03, final_target_stability_steps=10,
        score='target In basket AND no gripper-target contact throughout final10 settling frames AND protected center never moves more than 30mm; empty closed gripper is not penalized',
        scoring_scope='Custom selective tasks; not original LIBERO-Long success rate; ignores original done for ALL controller stopping',
        control='10-action chunks, execute first5; 10-step checks, released gripper gate, two consecutive eligible yes votes; reset on closed gripper or >20-step vote gap',
        calibration='target-specific ceil(first end of10 consecutive target-free/placed/protected-safe steps /10)*10 from initial0 reference skill; 150 fallback when unavailable; fixed before evaluation',
        model='Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17; no adapter',
        policy='official pi05_libero; 10 denoising steps; explicit paired noise',
        selector='Qwen selects one target and protected ID from a supplied two-object vocabulary; deterministic English single-skill template',
        monitor='Existing base before/after Qwen check; frozen substitutes reference for current; no simulator states sent',
        environment_seed=7, noise_seed='920000+task*100+initial; same across methods AND target directions',
        no_hardware=True,no_training=True,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources})
    path=S/'PROTOCOL.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Frozen protocol/source changed; use a new directory'
    else:
        atomic_json(path,value)
        for name in value['sources']:
            out=S/'archive'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((ROOT/name).read_bytes())
    return sha(path)


def run_episode(c, mode, phase, policy, worker, calibration, protocol_sha):
    import imageio.v2 as imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    out=S/f'episodes/{phase}/{c["key"]}/{mode}'
    if (out/'result.json').exists():
        old=json.loads((out/'result.json').read_text())
        assert old.get('status') == 'complete' and old.get('protocol_sha256') == protocol_sha, f'Incomplete/different {out}'
        return old
    out.mkdir(parents=True,exist_ok=True)
    (out/'images').mkdir();(out/'policy_inputs').mkdir()
    seed=920000+c['task']*100+c['initial'];rng=np.random.default_rng(seed)
    record=dict(case=c,mode=mode,phase=phase,protocol_sha256=protocol_sha,
                noise_seed=seed,status='running',steps=0,events=[],policy_queries=0,model_checks=0,
                control_truth_inputs=False,original_done_used_for_stopping=False)
    journal=EpisodeJournal(out,record)
    env=writer=None;trace=[];stop_reason='budget';decision_step=0
    def event(kind,step,**kw):
        row=dict(kind=kind,step=step,**kw);record['events'].append(row);journal.event(row)
    started=time.monotonic()
    try:
        suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(c['task'])
        initial=np.asarray(ev.initial_states(suite,c['task'])[c['initial']])
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
        env.seed(7);env.reset();obs=env.set_init_state(initial)
        for _ in range(10):obs,_,_,_=env.step(ev.DUMMY.tolist())
        record['initial_state_sha256']=hashlib.sha256(initial.tobytes()).hexdigest()
        record['settled_state_sha256']=hashlib.sha256(env.sim.get_state().flatten().tobytes()).hexdigest()
        record['goal_atoms']=env.env.parsed_problem['goal_state']
        body_ids=[env.env.obj_body_id[o[0]] for o in c['objects']]
        origins=env.sim.data.body_xpos[body_ids].copy()
        record['initial_object_positions']=origins.tolist()
        reference=save_png(monitor_view(env,512),out/'images/reference.png')
        selection=None
        if phase=='skills':
            instruction=c['reference_skill'];selected_target=c['target']
        else:
            plan_path=S/f'plans/{phase}/{c["key"]}.json';plan_path.parent.mkdir(parents=True,exist_ok=True)
            if plan_path.exists():
                plan=json.loads(plan_path.read_text())
                assert plan['reference_pixel_sha256']==sha(reference)
            else:
                selection=worker.request(dict(op='select',image=reference,instruction=c['instruction'],objects=c['objects']))
                translated=worker.request(dict(op='translate',image=reference,instruction=c['instruction']))
                plan=dict(case=c,reference_pixel_sha256=sha(reference),selection=selection,translation=translated)
                atomic_json(plan_path,plan)
            selection=plan['selection'];translated=plan['translation']
            if not selection.get('ok') or not translated.get('ok'):
                raise RuntimeError('Planning/translation invalid: '+str(plan))
            selected_target=selection['target']
            names={o[0]:o[1] for o in c['objects']}
            selected_skill=skill(names[selected_target])
            instruction=c['instruction'] if mode=='flat_zh' else translated['subgoals'][0]['instruction'] if mode=='translated_en' else selected_skill
            record.update(plan=str(plan_path.relative_to(S)),plan_sha256=sha(plan_path),selector_correct=selected_target==c['target'],selection=selection,translation=translated)
        completion_skill=skill({o[0]:o[1] for o in c['objects']}[selected_target])
        threshold=calibration.get(selected_target,150) if mode=='timer_calibrated' else 150
        rule=StopRule(mode,threshold)
        record.update(policy_instruction=instruction,selected_target=selected_target,timer_threshold=threshold)
        journal.save()
        writer=imageio.get_writer(out/'rollout.mp4',fps=20,codec='libx264',quality=7,macro_block_size=16)
        last=ev.DUMMY.copy()
        def apply(action,phase_id):
            nonlocal obs,last
            obs,_,original_done,_=env.step(action.tolist());last=np.asarray(action)
            # Below is a write-only scoring path. Its return values never enter StopRule or Qwen.
            atoms=np.array([bool(env.env._eval_predicate(a)) for a in record['goal_atoms']])
            positions=env.sim.data.body_xpos[body_ids].copy()
            sample=dict(applied=last.copy(),phase=np.int8(phase_id),eef=obs['robot0_eef_pos'].copy(),
                gripper=obs['robot0_gripper_qpos'].copy(),released=released(obs['robot0_gripper_qpos'],last),
                target_free=not env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(c['target']).contact_geoms),
                target_in=bool(atoms[c['target_index']]),atoms=atoms,object_positions=positions,
                protected_displacement=float(np.linalg.norm(positions[1-c['target_index']]-origins[1-c['target_index']])),
                original_done=bool(original_done),sim_state=env.sim.get_state().flatten().copy())
            trace.append(sample);writer.append_data(upright(obs));journal.step(len(trace),sample)
        writer.append_data(upright(obs))
        for step in range(0,MAX_STEPS,5):
            eligible=rule.eligible(step,obs['robot0_gripper_qpos'],last)
            if eligible:
                current=save_png(monitor_view(env,512),out/f'images/current_{step:04d}.png')
                reply=None;t0=time.monotonic()
                sent=[]
                if mode in ['visual','frozen']:
                    sent=[reference,reference if mode=='frozen' else current]
                    reply=worker.request(dict(op='check',image=sent,subgoal=completion_skill))
                    record['model_checks']+=1
                decision=rule.decide(step,reply)
                event('check',step,decision=decision,images=[reference,current],sent_images=sent,
                      reply=reply,seconds=time.monotonic()-t0)
                if decision['stop']:
                    stop_reason='visual' if mode in ['visual','frozen'] else mode
                    decision_step=step;break
            element=ev.make_input(obs,instruction);noise=rng.standard_normal((10,32)).astype(np.float32)
            actions=np.asarray(policy.infer(element,noise=noise)['actions'],np.float64)
            assert actions.shape==(10,7) and np.isfinite(actions).all()
            np.savez_compressed(out/f'policy_inputs/step{step:04d}.npz',image=element['observation/image'],wrist=element['observation/wrist_image'],state=element['observation/state'],noise=noise,actions=actions,instruction=instruction)
            append_jsonl(out/'policy_queries.jsonl',dict(step=step,instruction=instruction,hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in element.items() if k!='prompt'},noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest()))
            record['policy_queries']+=1
            for action in actions[:5]:apply(action,0)
        if not decision_step:decision_step=MAX_STEPS
        event('stop',decision_step,reason=stop_reason)
        # Stationary OSC command; preserve the last commanded gripper sign, never force a release.
        hold=np.zeros(7);hold[6]=last[6]
        for _ in range(SETTLE_STEPS):apply(hold,1)
        arrays={key:np.array([row[key] for row in trace]) for key in trace[0]}
        record.update(score(arrays,decision_step,stop_reason),status='complete',decision_step=decision_step,
            stop_reason=stop_reason,wall_seconds=time.monotonic()-started,
            original_success_observed=bool(arrays['original_done'].any()),
            policy_steps_after_original_success=int(sum(arrays['phase'][np.flatnonzero(arrays['original_done'])[0]+1:]==0)) if arrays['original_done'].any() else 0)
        record['success']=record['selective_success']
        np.savez_compressed(out/'trace.npz',**arrays);journal.save()
        return record
    except BaseException as exc:
        record.update(status='error',error=repr(exc),traceback=traceback.format_exc())
        if trace:np.savez_compressed(out/'trace.npz',**{k:np.array([row[k] for row in trace]) for k in trace[0]})
        journal.save();raise
    finally:
        if writer is not None:writer.close()
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','eval'],default='pilot');args=parser.parse_args()
    protocol_sha=freeze();worker=Worker()
    try:
        from openpi.training import config
        from openpi.policies import policy_config
        atomic_json(S/'STATUS.json',dict(state='loading_policy',stage=args.stage))
        policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),V/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
        policy.infer({'observation/image':np.zeros((224,224,3),np.uint8),'observation/wrist_image':np.zeros((224,224,3),np.uint8),'observation/state':np.zeros(8,np.float32),'prompt':'put the object in the basket'},noise=np.zeros((10,32),np.float32))
        if args.stage=='pilot':
            skills=[]
            for task in [0,1]:
                for target in [0,1]:
                    c=case(task,target,0);r=run_episode(c,'reference_skill','skills',policy,worker,{},protocol_sha);skills.append(r)
                    atomic_json(S/'STATUS.json',dict(state='running',stage='skills',completed=len(skills),expected=4,latest=dict(case=c['key'],first_valid_step=r['first_valid_step'],target_ever_reached=r['target_ever_reached'])))
                    print(json.dumps(dict(stage='skills',case=c['key'],first_valid_step=r['first_valid_step'],target_ever_reached=r['target_ever_reached'])),flush=True)
            calibration=calibrate(skills);atomic_json(S/'CALIBRATION.json',dict(thresholds=calibration,source_results=[str((S/f'episodes/skills/{r["case"]["key"]}/reference_skill/result.json').relative_to(S)) for r in skills]))
            if sum(r['first_valid_step'] is not None for r in skills)<2:
                raise RuntimeError('Fewer than two selective skills are feasible before moving the protected object; do not expand')
            initials=[0]
        else:
            calibration=json.loads((S/'CALIBRATION.json').read_text())['thresholds'];initials=[3,4,5,6,7]
            lock=dict(protocol_sha256=protocol_sha,calibration_sha256=sha(S/'CALIBRATION.json'),initials=initials)
            path=S/'EVALUATION_LOCK.json'
            if path.exists():assert json.loads(path.read_text())==lock
            else:atomic_json(path,lock)
        completed=[]
        for task in [0,1]:
            for initial in initials:
                for target in [0,1]:
                    c=case(task,target,initial);offset=(task*10+initial*2+target)%len(MODES)
                    for mode in MODES[offset:]+MODES[:offset]:
                        r=run_episode(c,mode,args.stage,policy,worker,calibration,protocol_sha)
                        item=dict(case=c['key'],mode=mode,success=r['selective_success'],protected=r['protected_untouched'],target=r['target_final_stable'],decision_step=r['decision_step'])
                        completed.append(item)
                        atomic_json(S/'STATUS.json',dict(state='running',stage=args.stage,completed=len(completed),expected=2*len(initials)*2*len(MODES),latest=item))
                        print(json.dumps(item),flush=True)
        atomic_json(S/'STATUS.json',dict(state='complete',stage=args.stage,completed=len(completed),rows=completed))
    finally:worker.close()


if __name__=='__main__':main()
