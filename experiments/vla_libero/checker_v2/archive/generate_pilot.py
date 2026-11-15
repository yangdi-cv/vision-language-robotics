# ruff: noqa: E402
"""Matched static counterfactuals plus genuine grasp/release frames. No policy inference."""
from pathlib import Path
import json,sys,hashlib,math
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];V=R.parent
sys.path.insert(0,str(V/'scripts'))
import evaluate as ev
from run_hierarchical import monitor_view
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
TASKS=json.loads((V/'configs/libero_long_tasks.json').read_text())
NAMES={'alphabet_soup':'alphabet soup can','tomato_sauce':'tomato sauce can','cream_cheese':'cream cheese box','butter':'butter','porcelain_mug':'white mug','white_yellow_mug':'yellow and white mug','moka_pot':'moka pot','black_book':'black book','chocolate_pudding':'chocolate pudding'}
TARGETS={'basket_1_contain_region':'in the basket','plate_1':'on the plate','plate_2':'on the plate','flat_stove_1_cook_region':'on the stove','desk_caddy_1_back_contain_region':'in the back compartment of the caddy','microwave_1_heating_region':'in the microwave','living_room_table_plate_right_region':'to the right of the plate'}

def env_for(suite,tid):
 t=suite.get_task(tid)
 e=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/t.problem_folder/t.bddl_file),camera_heights=256,camera_widths=256)
 e.seed(7);return e

def reset(e,suite,tid,init):
 e.reset();o=e.set_init_state(np.asarray(ev.initial_states(suite,tid)[init]))
 for _ in range(10):o,_,_,_=e.step(ev.DUMMY.tolist())
 return o

def save(im,name):
 p=R/'data/images'/name;p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(im).save(p)
 return str(p.relative_to(R/'data'))

def regen(e):return e.regenerate_obs_from_state(e.sim.get_state().flatten())
def grasp(e,name):return bool(e.env._check_grasp(e.env.robots[0].gripper,e.env.get_object(name).contact_geoms))
def target_pos(e,target):return np.array(e.env.object_states_dict[target].get_geom_state()['pos']).copy()

def main():
 out=R/'data/cases.json'
 if out.exists():raise FileExistsError(out)
 rows=[];rejections=[];suite=benchmark.get_benchmark_dict()['libero_10']()
 initials={'train':[3,4,5,6],'validation':[7,8],'test':[9,10]}
 for tid in [0,1,4,5,6,7,8,9]:
  e=env_for(suite,tid)
  try:
   reset(e,suite,tid,1);base=V/f'results/baseline/libero_10/task{tid:02d}_init01'
   trace=np.load(base/'trace.npz')
   for a in trace['applied']:e.step(a.tolist())
   templates=[]
   for atom in TASKS[tid]['goal_predicates']:
    if atom[0] not in ['on','in'] or atom[1] not in e.env.objects_dict:continue
    name,target=atom[1:];joint=e.env.get_object(name).joints[0];q=np.array(e.sim.data.get_joint_qpos(joint)).copy()
    templates.append((atom,q,q[:3]-target_pos(e,target)))
   for split,ids in initials.items():
    for init in ids:
     for goal,(atom,q,offset) in enumerate(templates):
      name,target=atom[1:];rng=np.random.default_rng(420000+tid*1000+init*10+goal)
      reset(e,suite,tid,init)
      cam=e.sim.model.camera_name2id('agentview');e.sim.model.cam_pos[cam]+=rng.uniform(-.025,.025,3);e.sim.model.cam_fovy[cam]+=float(rng.uniform(-2,2))
      # Both robot-pose options occur for every label in each matched scene group.
      if rng.random()<.5:e.sim.data.qpos[:9]=trace['qpos'][-1,:9]
      e.sim.forward();regen(e)
      group=f'synth_t{tid:02d}_i{init:02d}_g{goal}'
      ref=save(monitor_view(e,512),group+'_ref.png')
      joint=e.env.get_object(name).joints[0];q=q.copy();q[:3]=target_pos(e,target)+offset
      # Prefer the center for direct plate targets; keep validated offsets for narrow cavities.
      if target.startswith('plate_'):q[:2]=target_pos(e,target)[:2]
      e.sim.data.set_joint_qpos(joint,q);e.sim.data.set_joint_qvel(joint,np.zeros(6));e.sim.forward();regen(e)
      for _ in range(20):e.step(ev.DUMMY.tolist())
      if not e.env._eval_predicate(atom) or grasp(e,name):
       rejections.append({'group':group,'reason':'no supported released positive anchor'});continue
      anchor=np.asarray(e.sim.get_state().flatten()).copy();center=np.array(e.sim.data.get_joint_qpos(joint)).copy()
      angle=rng.uniform(0,2*math.pi);direction=np.array([math.cos(angle),math.sin(angle),0.])
      def place(delta):
       e.set_init_state(anchor);qq=center.copy();qq[:3]+=delta;e.sim.data.set_joint_qpos(joint,qq);e.sim.data.set_joint_qvel(joint,np.zeros(6));e.sim.forward();regen(e)
       return bool(e.env._eval_predicate(atom))
      hi=.02
      while hi<.65 and place(direction*hi):hi*=1.6
      if hi>=.65:
       rejections.append({'group':group,'reason':'no local predicate boundary; unsupported region'});continue
      lo=0.
      for _ in range(12):
       m=(lo+hi)/2
       if place(direction*m):lo=m
       else:hi=m
      variants=[('center','yes',np.zeros(3)),('center_jitter','yes',direction*min(.004,lo/4)),('edge_inside','unknown',direction*max(0,lo-.001)),('edge_outside','unknown',direction*(hi+.002)),('offset','no',direction*(hi+.04)),('far','no',direction*(hi+.09)),('floating','no',np.array([0,0,.14])),('near_contact','unknown',np.array([0,0,.004]))]
      instruction='put the '+NAMES.get(name.rsplit('_',1)[0],name.rsplit('_',1)[0].replace('_',' '))+' '+TARGETS.get(target,target.replace('_',' '))
      # Name two identical pots by their reference-frame side, rather than hidden object IDs.
      if tid==8:instruction='put the '+('right' if name=='moka_pot_1' else 'left')+' moka pot in the starting image on the stove'
      if tid==4:instruction='put the '+NAMES[name.rsplit('_',1)[0]]+' on the '+('left' if target=='plate_1' else 'right')+' plate'
      for kind,status,delta in variants:
       relation=place(delta);held=grasp(e,name);done=relation and not held
       if status=='yes' and not done or status=='no' and done:
        rejections.append({'group':group,'kind':kind,'reason':'desired label not established'});continue
       cur=save(monitor_view(e,512),group+'_'+kind+'.png')
       rows.append({'id':group+'_'+kind,'group':group,'split':split,'source':'synthetic','task':tid,'init':init,'goal':goal,'instruction':instruction,'images':[ref,cur],'status':status,'done':done,'relation_predicate':relation,'grasp':held,'kind':kind,'atom':atom,'offset_m':delta.tolist(),'boundary_radius_m':hi})
   print(json.dumps({'task':tid,'rows':len(rows),'rejections':len(rejections)}),flush=True)
  finally:e.close()
 # Add natural physical grasps from different suites; never fit Long0-2 deployment frames.
 reserved={json.loads(p.read_text())['initial_state_sha256'] for p in (V/'results/baseline/libero_10').glob('*/result.json')}
 for p in sorted((V/'results/baseline').glob('*/*/result.json')):
  r=json.loads(p.read_text())
  if r['suite']=='libero_10' or r['initial_state_sha256'] in reserved:continue
  suite2=benchmark.get_benchmark_dict()[r['suite']]();e=env_for(suite2,r['task_id'])
  try:
   reset(e,suite2,r['task_id'],r['initial_state_id']);atoms=e.env.parsed_problem['goal_state']
   if len(atoms)!=1 or atoms[0][0] not in ['in','on'] or atoms[0][1] not in e.env.objects_dict:continue
   name=atoms[0][1];group=f'natural_{r["suite"]}_t{r["task_id"]:02d}_i{r["initial_state_id"]:02d}';ref=save(monitor_view(e,512),group+'_ref.png')
   tr=np.load(p.parent/'trace.npz');candidates=[]
   for step,a in enumerate(tr['applied']):
    e.step(a.tolist())
    if step%5==0 and grasp(e,name):candidates.append((step,monitor_view(e,512),bool(e.check_success())))
   selected=[candidates[0],candidates[-1]] if len(candidates)>1 else candidates
   split={0:'train',1:'validation',2:'test'}[r['initial_state_id']]
   for j,(step,im,rel) in enumerate(selected):
    cur=save(im,group+f'_held{j}.png');rows.append({'id':group+f'_held{j}','group':group,'split':split,'source':'natural','instruction':r['original_prompt'],'images':[ref,cur],'status':'no','done':False,'relation_predicate':rel,'grasp':True,'kind':'held_unreleased','step':step})
   for _ in range(20):e.step(ev.DUMMY.tolist())
   if e.check_success() and not grasp(e,name):
    cur=save(monitor_view(e,512),group+'_released.png');rows.append({'id':group+'_released','group':group,'split':split,'source':'natural','instruction':r['original_prompt'],'images':[ref,cur],'status':'yes','done':True,'grasp':False,'kind':'natural_released'})
  finally:e.close()
 splits={s:{x['group'] for x in rows if x['split']==s} for s in initials}
 for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not splits[a]&splits[b]
 out.write_text(json.dumps(rows,indent=2));(R/'data/rejections.json').write_text(json.dumps(rejections,indent=2))
 audit={'n':len(rows),'splits':{s:{'n':sum(x['split']==s for x in rows),'groups':len(splits[s]),'classes':{c:sum(x['split']==s and x['status']==c for x in rows) for c in ['yes','no','unknown']}} for s in initials},'source_counts':{s:sum(x['source']==s for x in rows) for s in ['synthetic','natural']},'rejected':len(rejections),'protocol_sha256':hashlib.sha256((R/'PROTOCOL.json').read_bytes()).hexdigest()}
 (R/'data/audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
