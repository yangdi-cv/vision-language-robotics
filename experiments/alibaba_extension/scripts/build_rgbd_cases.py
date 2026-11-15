"""Independent rendered RGB-D scenes; labels derived from paired material renders, not VLM."""
from pathlib import Path
import json,numpy as np,gymnasium as gym,gym_aloha
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];out=ROOT/'experiments/alibaba_extension/data/rgbd_cases';out.mkdir(parents=True,exist_ok=False)
env=gym.make('gym_aloha/AlohaTransferCube-v0',obs_type='pixels_agent_pos');cases=[];rng=np.random.default_rng(20261014)
colors={'red':([1,0,0,1],'红色'),'blue':([0,0,.9,1],'蓝色'),'green':([0,.8,0,1],'绿色')}
try:
 for index in range(30):
  color=list(colors)[index%3];rgba,chinese=colors[color];seed=4000+index;env.reset(seed=seed);physics=env.unwrapped._env.physics;geom=physics.model.name2id('red_box','geom');camera=physics.model.name2id('top','camera')
  physics.model.geom_size[geom]=np.array([.02,.025,.03][(index//3)%3]);physics.model.cam_pos[camera,:2]+=rng.uniform(-.015,.015,2)
  def render(rgba):physics.model.geom_rgba[geom]=rgba;return physics.render(height=480,width=640,camera_id='top').copy()
  white=render([1,1,1,1]);black=render([0,0,0,1]);mask=np.max(abs(white.astype(int)-black.astype(int)),axis=2)>10
  yy,xx=np.where(mask);assert len(xx)>30
  bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
  image=render(rgba);depth=physics.render(height=480,width=640,camera_id='top',depth=True).copy()
  Image.fromarray(image).save(out/f'scene_{index:03d}.png');np.save(out/f'depth_{index:03d}.npy',depth);np.save(out/f'mask_{index:03d}.npy',mask)
  metadata={'scene':index,'seed':seed,'color':color,'cube_half_size_m':float(physics.model.geom_size[geom,0]),'bbox_xyxy_pixels':bbox,'visible_pixels':int(mask.sum()),'target_depth_m':float(np.median(depth[mask])),'camera_pos':physics.model.cam_pos[camera].tolist()}
  (out/f'scene_{index:03d}.json').write_text(json.dumps(metadata,indent=2))
  for query,(_,name) in colors.items():
   visible=query==color;cases.append({'id':f'{index:03d}_{query}','scene':index,'image':f'scene_{index:03d}.png','instruction':f'抓起{name}方块。','target_visible':visible,'bbox_xyxy_pixels':bbox if visible else None,'expected_skill':'transfer_red_cube' if visible and query=='red' else 'observe','query_color':query,'negated':False,'depth_m':metadata['target_depth_m'] if visible else None})
  cases.append({'id':f'{index:03d}_negation','scene':index,'image':f'scene_{index:03d}.png','instruction':f'不要抓起{chinese}方块，只观察。','target_visible':True,'bbox_xyxy_pixels':bbox,'expected_skill':'observe','query_color':color,'negated':True,'depth_m':metadata['target_depth_m']})
finally:env.close()
(out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2));(out/'PROVENANCE.json').write_text(json.dumps({'source':'Actual MuJoCo renders, independent reset seeds4000-4029; color/size/camera perturbation for perception only','labels':'Matched material-difference mask between white and black target renders; RGB-D optical depth via dm_control; no evaluated-model labels','scene_count':30,'cases':120,'limitations':'One simulator and single visible cube per scene; not real multiobject or sensor-depth evaluation','modifications':'Perception-only geom size/color and camera changes; no ACT success evaluation on these modified scenes'},indent=2));print('Built',len(cases),'cases',flush=True)
