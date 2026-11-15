"""Paired fresh-layout evaluation; no teacher positions or oracle stopping."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from runtime import S,V,ROOT,load_policy
sys.path[:0]=[str(V/'scripts')]
from episode_journal import atomic_json
import importlib.util
spec=importlib.util.spec_from_file_location('diagnostic_engine',V/'instruction_response/scripts/run.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['base','lora','zero','native_reference'],required=True);a=ap.parse_args()
    if a.variant in ['zero','native_reference']:
        print('Loading ONE model: '+a.variant,flush=True)
        p=load_policy(None if a.variant=='native_reference' else S/'lora_pilot/adapter_zero.npz')
        import runtime
        fingerprints=runtime.LAST_LOAD_INFO['chunk_sha256']
        folder=V/'instruction_response/probes/task00_init00/step0000'
        row=json.loads((folder/'result.json').read_text());prompt=row['queries'][0]['prompt']
        with np.load(folder/'arrays.npz') as f:
            element={'observation/image':f['image'],'observation/wrist_image':f['wrist'],'observation/state':f['state'],'prompt':prompt}
            actual=np.asarray(p.infer(element,noise=f['noise'])['actions']);historical=np.abs(actual-f['target0'])
        record=dict(variant=a.variant,actions=actual.tolist(),parameter_sha256=fingerprints,native_vs_historical_max_difference=float(historical.max()),runtime_sha256=hashlib.sha256((S/'scripts/runtime.py').read_bytes()).hexdigest())
        if a.variant=='native_reference':
            atomic_json(S/'lora_pilot/NATIVE_REFERENCE.json',record);print('Native reference saved',flush=True);return
        ref=json.loads((S/'lora_pilot/NATIVE_REFERENCE.json').read_text());assert ref['runtime_sha256']==record['runtime_sha256']
        identical=ref['parameter_sha256']==fingerprints;d=np.abs(actual-np.asarray(ref['actions']))
        record.update(max_difference=float(d.max()),rms=float(np.sqrt(np.mean(d*d))),pass_tolerance=.001,passed=bool(d.max()<.001 and identical),all_base_parameter_arrays_equal=identical,comparison='separate single-model processes; exact parameter hash comparison',different_parameter_paths=[k for k in fingerprints if fingerprints[k]!=ref['parameter_sha256'].get(k)])
        atomic_json(S/'lora_pilot/ZERO_ADAPTER_CHECK.json',record);print(json.dumps({k:v for k,v in record.items() if k not in ['actions','parameter_sha256']}),flush=True);assert record['passed'];return
    if a.variant=='lora':assert json.loads((S/'lora_pilot/ZERO_ADAPTER_CHECK.json').read_text())['passed']
    folder=S/'policy_eval'/a.variant;folder.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),S/'scripts/runtime.py',S/'scripts/bounded_restore.py',S/'scripts/chunk_math.py',V/'instruction_response/scripts/run.py',S/'DATA_PROTOCOL.json']
    protocol=dict(variant=a.variant,initials=[28,29],tasks=[0,1],modes=['target0','target1'],episodes=8,policy_steps=300,hold_steps=25,no_oracle_control=True,no_teacher_pose=True,engine='same immutable diagnostic rollout; only its output root is redirected',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    adapter=S/'lora_pilot/adapter_best.npz' if a.variant=='lora' else None
    if adapter:protocol['adapter_sha256']=hashlib.sha256(adapter.read_bytes()).hexdigest();protocol['selected']=json.loads((S/'lora_pilot/BEST.json').read_text())
    assert not (folder/'PROTOCOL.json').exists(),'Refuse evaluation overwrite';atomic_json(folder/'PROTOCOL.json',protocol)
    version=hashlib.sha256((folder/'PROTOCOL.json').read_bytes()).hexdigest();engine.S=folder
    policy=load_policy(adapter);n=0
    for task in [0,1]:
        for initial in [28,29]:
            for mode in ['target0','target1']:
                r=engine.rollout(policy,task,initial,mode,version);n+=1
                atomic_json(folder/'STATUS.json',dict(state='running',complete=n,expected=8));print(json.dumps(dict(task=task,initial=initial,mode=mode,scores=r['scores'])),flush=True)
    atomic_json(folder/'STATUS.json',dict(state='complete',complete=n))

if __name__=='__main__':main()
