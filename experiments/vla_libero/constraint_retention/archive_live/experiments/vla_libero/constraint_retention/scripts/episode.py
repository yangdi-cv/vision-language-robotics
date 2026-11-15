"""Active configurable episode engine for live camera ablations. The earlier execution.py is a frozen prompt-study artifact. Shared scoring/controller remain unchanged."""
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
from design import MODES, SPECS, OBJECTS, MAX_STEPS, SETTLE_STEPS, case, skill, score, released, StopRule, make_prompt
from view_checker import input_images


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_episode(c, mode, phase, policy, worker, calibration, protocol_sha, monitor_scheme=None):
    import imageio.v2 as imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    spec=SPECS[mode]
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
                control_truth_inputs=False,original_done_used_for_stopping=False,monitor_scheme=monitor_scheme)
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
            plan_path=S/f'plans/live/{c["key"]}.json';plan_path.parent.mkdir(parents=True,exist_ok=True)
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
            instruction=make_prompt(c['objects'],selection,spec['prompt'])
            record.update(plan=str(plan_path.relative_to(S)),plan_sha256=sha(plan_path),selector_correct=selected_target==c['target'],selection=selection,translation=translated)
        completion_skill=skill({o[0]:o[1] for o in c['objects']}[selected_target])
        threshold=150
        rule=StopRule(spec['stop'],threshold)
        record.update(policy_instruction=instruction,selected_target=selected_target,timer_threshold=threshold,prompt_family=spec['prompt'],stop_family=spec['stop'])
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
                wrist=None
                if monitor_scheme is not None:
                    wrist=save_png(env.sim.render(camera_name='robot0_eye_in_hand',height=512,width=512)[::-1,::-1],out/f'images/wrist_{step:04d}.png')
                reply=None;t0=time.monotonic()
                sent=[]
                if spec['stop']=='visual':
                    sent=[reference,current] if monitor_scheme is None else input_images({'images':{'reference':{'path':reference},'agent':{'path':current},'wrist':{'path':wrist}}},monitor_scheme)
                    reply=worker.request(dict(op='check',image=sent,subgoal=completion_skill))
                    record['model_checks']+=1
                decision=rule.decide(step,reply)
                event('check',step,decision=decision,images=[reference,current],sent_images=sent,
                      reply=reply,wrist_image=wrist,seconds=time.monotonic()-t0)
                if decision['stop']:
                    stop_reason='visual' if spec['stop']=='visual' else 'timer150'
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

