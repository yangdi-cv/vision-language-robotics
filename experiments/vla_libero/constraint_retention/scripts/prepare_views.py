"""Re-render aligned agent/wrist views from saved states; no labels enter the VLM."""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from design import S,V,PREVIOUS,case,skill

ROOT=V.parents[1];sys.path.insert(0,str(V/'scripts'))
from episode_journal import atomic_json
import evaluate as ev
from run_hierarchical import monitor_view


def save(array,path):
    path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.ascontiguousarray(array)).save(path)
    return dict(path=str(path.relative_to(ROOT)),pixel_sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['development','test'],required=True);a=ap.parse_args()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    if a.split=='development':
        paths=[PREVIOUS/f'episodes/pilot/{case(t,o,0)["key"]}/skill_continue/result.json' for t in [0,1] for o in [0,1]]
    else:
        gate=json.loads((S/'PILOT_GATE.json').read_text())['expand']
        if gate:
            paths=[S/f'episodes/eval/{case(t,o,i)["key"]}/positive_continue/result.json' for t in [0,1] for i in [8,9,10] for o in [0,1]]
        else:paths=[ROOT/name for name in json.loads((S/'CAMERA_INDEX.json').read_text())]
    rows=[]
    for source in paths:
        r=json.loads(source.read_text());assert r['status']=='complete';c=r['case']
        suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(c['task'])
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
        try:
            env.seed(7);env.reset();env.set_init_state(np.asarray(ev.initial_states(suite,c['task'])[c['initial']]))
            for _ in range(10):env.step(ev.DUMMY.tolist())
            folder=S/f'view_audit/images/{a.split}/{c["key"]}'
            ref=save(monitor_view(env,512),folder/'reference.png')
            selected=r['selected_target'];names={o[0]:o[1] for o in c['objects']}
            atom=next(x for x in r['goal_atoms'] if x[1]==selected)
            with np.load(source.parent/'trace.npz') as f:states=f['sim_state']
            for e in r['events']:
                if e['kind']!='check':continue
                step=e['step'];env.sim.set_state_from_flattened(states[step-1]);env.sim.forward()
                agent=save(monitor_view(env,512),folder/f'agent_{step:04d}.png')
                wrist=save(env.sim.render(camera_name='robot0_eye_in_hand',height=512,width=512)[::-1,::-1],folder/f'wrist_{step:04d}.png')
                inside=bool(env.env._eval_predicate(atom));contact=env.env.check_contact(env.env.robots[0].gripper.contact_geoms,env.env.get_object(selected).contact_geoms)
                rows.append(dict(id=f'{a.split}_{c["key"]}_{step:04d}',split=a.split,case=c['key'],task=c['task'],target_index=c['target_index'],step=step,
                    subgoal=skill(names[selected]),images=dict(reference=ref,agent=agent,wrist=wrist),
                    goal_done=inside and not contact,inside=inside,gripper_contact=bool(contact),
                    selected_target=selected,requested_target=c['target'],observability='not_reviewed',
                    source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    label_source='In + no gripper contact after restoring and forwarding the SAME state used to render BOTH candidate views'))
        finally:env.close()
    dest=S/'view_audit';dest.mkdir(exist_ok=True)
    atomic_json(dest/f'{a.split}.json',rows)
    print(json.dumps(dict(split=a.split,queries=len(rows),positive=sum(r['goal_done'] for r in rows),negative=sum(not r['goal_done'] for r in rows))))


if __name__=='__main__':main()
