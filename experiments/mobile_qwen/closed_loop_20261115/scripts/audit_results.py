"""Check completed trial artifacts independently of the batch driver."""
from pathlib import Path
import argparse,json
import numpy as np
D=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--prefix',default='paired_v1');a=p.parse_args()
rows=json.loads((D/'results'/f'{a.prefix}_summary.json').read_text());assert len(rows)==12
assert {(r['initial'],r['variant'],r['mode']) for r in rows}=={(i,v,m) for i in [2,3] for v in ['camera_right_3cm','camera_up_3cm'] for m in ['oracle','base','lora']}
report=[]
for r in rows:
 folder=D/'results'/r['tag'];assert (folder/'rollout.mp4').stat().st_size>1000
 assert json.loads((folder/'case_result.json').read_text())==r
 controls=list(folder.glob('controller_*.json'));samples=0;peak_error=0
 for f in controls:
  x=json.loads(f.read_text());tr=x['trajectory'];samples+=len(tr)
  if not tr:continue
  assert all(v['physics_id_unchanged'] for v in tr)
  assert all(np.isfinite(v['joint_positions']).all() and np.isfinite(v['actual_generalized_actuation']).all() for v in tr)
  times=np.array([v['time'] for v in tr]);assert np.all(np.diff(times)>0)
  if x['error_code']==0:
   err=float(np.max(np.abs(np.array(tr[-1]['joint_positions'])-np.array(tr[-1]['command']))));assert err<=.015
   peak_error=max(peak_error,err)
 if r['success']:
  assert samples>0 and r['final_audit']['physics_id_unchanged']
  assert not r['final_audit']['any_arm_environment_contact'] and not r['final_audit']['any_self_contact']
  assert r['final_audit']['shadow_true_precontact_error_m']<=.03
 for f in folder.glob('prediction_*.json'):
  x=json.loads(f.read_text())
  if r['mode']!='oracle':assert x['inference']['adapter_enabled']==(r['mode']=='lora')
 report.append({'tag':r['tag'],'control_samples':samples,'successful_controller_final_max_joint_error':peak_error,'artifact_check_passed':True})
(D/'ARTIFACT_AUDIT.json').write_text(json.dumps(report,indent=2));print(f'{len(report)} trial artifact checks passed')
