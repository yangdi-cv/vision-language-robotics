"""Reload a fine-tuned checkpoint and reproduce a saved offline prediction."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_mousepad import ROOT,Demonstrations,evaluate,SmolVLAPolicy,PreTrainedConfig,load_smolvla,torch,np
name=sys.argv[1] if len(sys.argv)>1 else 'mousepad_all3_v1'
out=ROOT/'experiments/smolvla_pc/runs'/name
statistics=json.loads((out/'normalization.json').read_text());stats={k:{n:torch.tensor(v,dtype=torch.float32) for n,v in d.items()} for k,d in statistics.items()}
cfg=PreTrainedConfig.from_pretrained(str(out/'checkpoint'));cfg.load_vlm_weights=False
model=SmolVLAPolicy(cfg,dataset_stats=stats);load_smolvla(model,out/'checkpoint/model.safetensors');model.cuda().eval()
data=Demonstrations([json.loads((out/'config.json').read_text())['evaluation_episode']])
result=evaluate(model,data,[0]);expected=json.loads((out/'after.json').read_text())['per_frame'][0]['first_prediction'];actual=result['per_frame'][0]['first_prediction'];error=float(abs(np.array(actual)-expected).max())
r={'checkpoint':name,'reload_prediction_max_abs_difference':error,'passed':error<1e-3,'motion_enabled':False}
(out/'reload_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));assert r['passed']
