"""Evaluate the selected approach checkpoint and issue a conservative trial gate."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_approach import ROOT,Demonstrations,evaluate,SmolVLAPolicy,PreTrainedConfig,load_smolvla,torch,np
name=sys.argv[1] if len(sys.argv)>1 else 'approach_all3_v1';run=ROOT/'experiments/smolvla_pc/runs'/name
config=json.loads((run/'config.json').read_text());d=json.loads((run/'normalization.json').read_text());stats={k:{n:torch.tensor(v,dtype=torch.float32) for n,v in r.items()} for k,r in d.items()}
cfg=PreTrainedConfig.from_pretrained(str(run/'best_checkpoint'));cfg.load_vlm_weights=False
model=SmolVLAPolicy(cfg,dataset_stats=stats);load_smolvla(model,run/'best_checkpoint/model.safetensors');model.cuda().eval()
data=Demonstrations([1,2,3]);indices=np.linspace(0,len(data)-1,36).astype(int).tolist();r=evaluate(model,data,indices)
errors=np.array([np.array(x['first_prediction'][:5])-np.array(x['first_target'][:5]) for x in r['per_frame']]);mae=float(abs(errors).mean());p95=float(np.percentile(abs(errors),95))
first_deltas=[];starts=[]
for images,q,a in data.episodes:starts.append(q[0,:5].tolist())
for item in r['per_frame']:
 first_deltas.append(abs(np.array(item['first_prediction'][:5])-data[item['dataset_index']]['observation.state'][:5].numpy()).tolist())
delta_p95=float(np.percentile(first_deltas,95))
gate={'checkpoint':name,'weights_sha256':hashlib.sha256((run/'best_checkpoint/model.safetensors').read_bytes()).hexdigest(),'evaluation_training_overlap':True,'contexts':len(indices),'first_five_joint_mae_deg':mae,'first_five_joint_abs_error_p95_deg':p95,'proposed_first_step_delta_p95_deg':delta_p95,'bounded_trial_allowed':mae<3 and p95<8 and delta_p95<10,'max_trial_seconds':8,'max_start_relative_joint_degrees':8,'recorded_start_states':starts,'reason':'Only bounded in-distribution upright-start approach; not validated target tracking or grasping','motion_started':False}
gate['source_has_independent_validation_pass']=False
(run/'offline_gate.json').write_text(json.dumps(gate,indent=2));(run/'selected_evaluation.json').write_text(json.dumps(r,indent=2));print(json.dumps(gate,indent=2))
