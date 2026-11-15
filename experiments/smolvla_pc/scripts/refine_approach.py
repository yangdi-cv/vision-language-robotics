"""Fine-tune the vendor SmolVLA on complete local mousepad demonstrations."""
import argparse,json,time,sys,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'SO101/code/lerobot/src'))
import numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,load_smolvla
TASK='Move the gripper toward the blue earbud case without grasping it.'
HORIZON=10
class Demonstrations(Dataset):
 def __init__(self,episodes):
  self.episodes=[];self.index=[]
  for ep in episodes:
   p=ROOT/'datasets/earbud_mousepad/prepared_v1'/f'episode_{ep:02d}.json';d=json.loads(p.read_text());rows=d['rows'];g=np.array([r['state'][5] for r in rows]);closing=np.flatnonzero((np.maximum.accumulate(g)>20)&(g<np.maximum.accumulate(g)-8))
   if not len(closing):raise RuntimeError('Cannot identify pre-close boundary')
   rows=rows[:max(1,int(closing[0])-16)];frames=[]
   for r in rows:
    im=Image.open(ROOT/d['source']/r['image']).convert('RGB');im.thumbnail((256,256));canvas=Image.new('RGB',(256,256));canvas.paste(im,((256-im.width)//2,(256-im.height)//2));frames.append(torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1))
   q=torch.tensor([r['state'] for r in rows],dtype=torch.float32);a=torch.tensor([r['action'] for r in rows],dtype=torch.float32)
   idx=len(self.episodes);self.episodes.append((torch.stack(frames),q,a))
   self.index.extend((idx,i) for i in range(len(rows)))
 def __len__(self):return len(self.index)
 def __getitem__(self,index):
  ep,i=self.index[index];images,q,a=self.episodes[ep];indices=torch.arange(i,i+HORIZON);pad=indices>=len(a);indices=indices.clamp(max=len(a)-1)
  return {'observation.images.camera1':images[i].float()/255,'observation.state':q[i],'action':a[indices],'actions_id_pad':pad,'task':TASK}
 def stats(self):
  return {key:{'mean':x.mean(0),'std':x.std(0,unbiased=False).clamp(min=.01)} for key,j in [('observation.state',1),('action',2)] for x in [torch.cat([e[j] for e in self.episodes])]}
def move(batch):return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
def evaluate(model,data,indices):
 model.eval();errors=[];hold=[];per_frame=[];horizons={n:[] for n in (1,5,10)};hold_horizons={n:[] for n in (1,5,10)}
 for j,idx in enumerate(indices):
  sample=data[idx];batch=move({k:v[None] if isinstance(v,torch.Tensor) else [v] for k,v in sample.items()})
  model.reset();torch.manual_seed(500+j)
  with torch.inference_mode():pred=model.predict_action_chunk(batch)[0].float().cpu()
  valid=~sample['actions_id_pad'];target=sample['action'][valid];pr=pred[valid]
  errors.append(abs(pr-target).mean(0).numpy());hold.append(abs(sample['observation.state'][None]-target).mean(0).numpy())
  for n in horizons:
   horizons[n].append(abs(pr[:n]-target[:n]).mean(0).numpy());hold_horizons[n].append(abs(sample['observation.state'][None]-target[:n]).mean(0).numpy())
  per_frame.append({'dataset_index':idx,'first_prediction':pred[0].tolist(),'first_target':sample['action'][0].tolist()})
 return {'horizon_joint_mae':{str(n):np.mean(v,axis=0).tolist() for n,v in horizons.items()},'hold_horizon_joint_mae':{str(n):np.mean(v,axis=0).tolist() for n,v in hold_horizons.items()},'joint_mae':np.mean(errors,axis=0).tolist(),'hold_current_joint_mae':np.mean(hold,axis=0).tolist(),'evaluated_contexts':len(indices),'per_frame':per_frame,'units':['deg']*5+['gripper_percent'],'task_success_measured':False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-episodes',default='1,2,3');ap.add_argument('--eval-episode',type=int,default=3);ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--name',default='approach_refined_v1');args=ap.parse_args()
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(6)
 out=ROOT/'experiments/smolvla_pc/runs'/args.name;out.mkdir(exist_ok=False)
 train=Demonstrations([int(x) for x in args.train_episodes.split(',')]);val=Demonstrations([args.eval_episode]);stats=train.stats()
 cfg=PreTrainedConfig.from_pretrained(str(ROOT/'models/smolvla_base'));cfg.load_vlm_weights=False
 cfg.resize_imgs_with_padding=(256,256)
 cfg.chunk_size=HORIZON;cfg.n_action_steps=HORIZON
 model=SmolVLAPolicy(cfg,dataset_stats=stats);load_smolvla(model,ROOT/'experiments/smolvla_pc/runs/approach_all3_v1/best_checkpoint/model.safetensors');model.cuda()
 parameters=[p for p in model.parameters() if p.requires_grad]
 config={'train_episodes':args.train_episodes,'evaluation_episode':args.eval_episode,'evaluation_is_training_overlap':str(args.eval_episode) in args.train_episodes.split(','),'phase':'upright_to_pregrasp_only','horizon':HORIZON,'cutoff':'16 frames before first measured gripper closing; heuristic boundary','steps':args.steps,'batch_size':2,'learning_rate':2e-5,'resume_weights':'approach_all3_v1/best_checkpoint','image_size':256,'resize_note':'256x256 aspect-preserving black padding; same preprocessing required in inference','train_frames':len(train),'eval_frames':len(val),'normalization':'training episodes only, per joint mean/std','trainable_parameters':sum(p.numel() for p in parameters),'task':TASK,'base_revision':'5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6','robot_motion_enabled':False}
 (out/'config.json').write_text(json.dumps(config,indent=2));(out/'normalization.json').write_text(json.dumps({k:{s:v.tolist() for s,v in d.items()} for k,d in stats.items()},indent=2))
 print(json.dumps(config),flush=True)
 indices=np.linspace(0,len(val)-1,12).astype(int).tolist()
 before=evaluate(model,val,indices);(out/'before.json').write_text(json.dumps(before,indent=2));print('Before MAE',before['joint_mae'],flush=True)
 optimizer=torch.optim.AdamW(parameters,lr=2e-5,weight_decay=1e-10)
 loader=DataLoader(train,batch_size=2,shuffle=True,num_workers=0);iterator=iter(loader);started=time.monotonic();model.train();best=float('inf');best_step=None
 for step in range(1,args.steps+1):
  try:batch=next(iterator)
  except StopIteration:iterator=iter(loader);batch=next(iterator)
  optimizer.zero_grad(set_to_none=True)
  loss,_=model(move(batch))
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,10);optimizer.step()
  if step==1 or step%50==0:
   row={'step':step,'loss':loss.item(),'grad_norm':float(norm),'elapsed_s':time.monotonic()-started}
   with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
  if step%1000==0:
   check=evaluate(model,val,indices);score=float(np.mean(np.array(check['horizon_joint_mae']['1'])/stats['action']['std'].numpy()))
   print('Validation',step,score,check['horizon_joint_mae']['1'],flush=True)
   (out/f'validation_{step}.json').write_text(json.dumps(check,indent=2))
   if score<best:
    best=score;best_step=step;model.save_pretrained(out/'best_checkpoint')
   model.train()
 model.save_pretrained(out/'checkpoint')
 after=evaluate(model,val,indices);(out/'after.json').write_text(json.dumps(after,indent=2));print('After MAE',after['joint_mae'],flush=True)
 summary={'status':'completed','config':config,'before':before,'after':after,'best_first_step_normalized_mae':best,'best_step':best_step,'training_seconds':time.monotonic()-started,'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print('Finished',out,flush=True)
if __name__=='__main__':main()
