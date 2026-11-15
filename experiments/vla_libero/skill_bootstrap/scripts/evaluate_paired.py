"""Fresh paired test for branch-focused adaptation; v1 test layouts stay separate."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
from runtime import S,V,ROOT,load_policy
sys.path.insert(0,str(V/'scripts'))
from episode_journal import atomic_json
spec=importlib.util.spec_from_file_location('diagnostic_engine',V/'instruction_response/scripts/run.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['base','paired'],required=True);a=ap.parse_args()
    assert json.loads((S/'lora_pilot/ZERO_ADAPTER_CHECK.json').read_text())['passed']
    folder=S/'policy_eval_paired'/a.variant;folder.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),S/'scripts/runtime.py',S/'scripts/bounded_restore.py',S/'scripts/chunk_math.py',V/'instruction_response/scripts/run.py',S/'DATA_PROTOCOL.json',S/'lora_paired/PROTOCOL.json']
    protocol=dict(variant=a.variant,initials=[30,31],tasks=[0,1],modes=['target0','target1'],episodes=8,policy_steps=300,hold_steps=25,no_oracle_control=True,no_teacher_pose=True,engine='immutable diagnostic rollout; output root redirected',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    adapter=S/'lora_paired/adapter_best.npz' if a.variant=='paired' else None
    if adapter:protocol.update(adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),selected=json.loads((S/'lora_paired/BEST.json').read_text()))
    assert not (folder/'PROTOCOL.json').exists(),'Refuse overwrite';atomic_json(folder/'PROTOCOL.json',protocol)
    engine.S=folder;version=hashlib.sha256((folder/'PROTOCOL.json').read_bytes()).hexdigest();policy=load_policy(adapter);n=0
    for task in [0,1]:
        for initial in [30,31]:
            for mode in ['target0','target1']:
                r=engine.rollout(policy,task,initial,mode,version);n+=1;atomic_json(folder/'STATUS.json',dict(state='running',complete=n,expected=8));print(json.dumps(dict(task=task,initial=initial,mode=mode,scores=r['scores'])),flush=True)
    atomic_json(folder/'STATUS.json',dict(state='complete',complete=8))

if __name__=='__main__':main()
