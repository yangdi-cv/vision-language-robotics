"""Small paired live-loop check; expand on fresh states only after net pilot gain."""
import hashlib,json
from pathlib import Path
import shutil,subprocess,sys,time
import numpy as np
from design import S,V,PREVIOUS,case
from episode import run_episode

ROOT=V.parents[1];sys.path[:0]=[str(V/'scripts'),str(ROOT/'perception/qwen_scene')]
from episode_journal import atomic_json,append_jsonl
from worker_client import read_json_line,close_json_worker
METHODS=('duplicate_control','wrist','timer')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Worker:
    def __init__(self):
        self.log=open(S/f'logs/live_worker_{time.time_ns()}.log','w');self.requests=S/f'logs/live_requests_{time.time_ns()}.jsonl'
        self.proc=subprocess.Popen([str(ROOT/'.venvs/qwen-scene/bin/python'),str(S/'scripts/live_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,start_new_session=True)
        self.ready=read_json_line(self.proc,timeout=240)
    def request(self,payload):
        append_jsonl(self.requests,dict(kind='request',payload=payload));self.proc.stdin.write(json.dumps(payload)+'\n');self.proc.stdin.flush()
        reply=read_json_line(self.proc,timeout=120);append_jsonl(self.requests,dict(kind='reply',reply=reply));return reply
    def close(self):close_json_worker(self.proc);self.log.close()


def freeze():
    files=[S/f'scripts/{n}.py' for n in ['live','live_worker','episode','design','view_checker']]
    files += [PREVIOUS/f'scripts/{n}.py' for n in ['protocol','control','worker']]
    files += [V/f'scripts/{n}.py' for n in ['evaluate','run_hierarchical','qwen_planner','episode_journal']]
    files += [ROOT/f'perception/qwen_scene/{n}.py' for n in ['engine','worker_client']]
    value=dict(methods=list(METHODS),development_initials=[0],test_initials=[11,12,13],
        all_policy_prompts='unchanged positive English skill; suffix branch rejected separately',
        primary='same 3-image prompt, parser, model, gate, votes and actor; only third camera differs',
        timer='150-step threshold with same gate/votes, not a fixed elapsed unconditional stop',
        development_reuse='four prior initial0 timer results; new two-view methods run live',
        expansion_gate='new-state24+12(timer) trials only if wrist has more pilot selective successes than matched duplicate-control; offline metrics never block the8 live pilot episodes',
        old_score='unchanged final10 In/no contact and protected max displacement<=30mm after25 hold steps',
        no_truth_control=True,no_hardware=True,no_training=True,
        view_protocol_sha256=sha(S/'VIEW_PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in files})
    path=S/'LIVE_PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:
        atomic_json(path,value)
        for name in value['sources']:
            p=S/'archive_live'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    return sha(path)


def main():
    version=freeze();worker=Worker()
    try:
        from openpi.training import config
        from openpi.policies import policy_config
        policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),V/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
        policy.infer({'observation/image':np.zeros((224,224,3),np.uint8),'observation/wrist_image':np.zeros((224,224,3),np.uint8),'observation/state':np.zeros(8,np.float32),'prompt':'put the object in the basket'},noise=np.zeros((10,32),np.float32))
        for stage,initials in [('pilot',[0]),('test',[11,12,13])]:
            if stage=='test' and not json.loads((S/'LIVE_GATE.json').read_text())['expand']:break
            if stage=='test':atomic_json(S/'LIVE_TEST_LOCK.json',dict(protocol_sha256=version,gate_sha256=sha(S/'LIVE_GATE.json'),initials=initials))
            rows=[]
            for task in [0,1]:
                for initial in initials:
                    for target in [0,1]:
                        c=case(task,target,initial)
                        if stage=='pilot':
                            plan=S/f'plans/live/{c["key"]}.json';plan.parent.mkdir(parents=True,exist_ok=True)
                            if not plan.exists():shutil.copyfile(PREVIOUS/f'plans/pilot/{c["key"]}.json',plan)
                        shift=(task*6+initial*2+target)%3
                        for method in METHODS[shift:]+METHODS[:shift]:
                            if stage=='pilot' and method=='timer':
                                p=PREVIOUS/f'episodes/pilot/{c["key"]}/timer150/result.json';r=json.loads(p.read_text());reused=True
                            else:
                                mode='positive_timer' if method=='timer' else 'positive_visual'
                                phase=f'live_{stage}_{method}'
                                r=run_episode(c,mode,phase,policy,worker,{},version,monitor_scheme='duplicate_control' if method=='timer' else method)
                                p=S/f'episodes/{phase}/{c["key"]}/{mode}/result.json';reused=False
                            rows.append(dict(case=c['key'],method=method,source=str(p.relative_to(ROOT)),source_sha256=sha(p),reused=reused,success=r['selective_success'],target_stable=r['target_final_stable'],protected_ok=r['protected_untouched'],premature=r['premature_stop'],step=r['decision_step']))
                            atomic_json(S/f'INDEX_live_{stage}.json',rows)
                            atomic_json(S/'LIVE_STATUS.json',dict(state='running',stage=stage,completed=len(rows),expected=12 if stage=='pilot' else 36,latest=rows[-1]))
                            print(json.dumps(rows[-1]),flush=True)
            if stage=='pilot':
                bycase={r['case']:{q['method']:q for q in rows if q['case']==r['case']} for r in rows}
                gains=sum(g['wrist']['success'] and not g['duplicate_control']['success'] for g in bycase.values())
                losses=sum(not g['wrist']['success'] and g['duplicate_control']['success'] for g in bycase.values())
                atomic_json(S/'LIVE_GATE.json',dict(expand=gains>losses,gains=gains,losses=losses,offline_metrics_not_used_as_gate=True))
        atomic_json(S/'LIVE_STATUS.json',dict(state='complete',expanded=json.loads((S/'LIVE_GATE.json').read_text())['expand']))
    finally:worker.close()


if __name__=='__main__':main()
