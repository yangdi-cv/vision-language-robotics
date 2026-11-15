"""Audit frozen code, replayed policy inputs, and matched image checks."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from design import S,V
ROOT=V.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    counts=dict(frozen_sources=0,new_episodes=0,policy_queries=0,image_checks=0)
    for name,archive in [('PROTOCOL.json','archive'),('VIEW_PROTOCOL.json','archive_view'),('LIVE_PROTOCOL.json','archive_live')]:
        for path,digest in json.loads((S/name).read_text())['sources'].items():
            assert sha(ROOT/path)==digest,(name,path)
            assert sha(S/archive/path)==digest,(archive,path)
            counts['frozen_sources']+=1
    paths=sorted((S/'episodes').glob('*/*/*/result.json'))
    for p in paths:
        r=json.loads(p.read_text());assert r['status']=='complete',p
        with np.load(p.parent/'trace.npz') as f:applied=f['applied'].copy()
        n=r['decision_step'];assert len(applied)==n+25
        actions=[]
        for line in (p.parent/'policy_queries.jsonl').read_text().splitlines():
            q=json.loads(line)
            with np.load(p.parent/f'policy_inputs/step{q["step"]:04d}.npz') as f:
                assert str(f['instruction'])==r['policy_instruction']==q['instruction']
                assert f['image'].shape==f['wrist'].shape==(224,224,3)
                for key,label in [('image','observation/image'),('wrist','observation/wrist_image'),('state','observation/state')]:
                    assert hashlib.sha256(np.ascontiguousarray(f[key]).tobytes()).hexdigest()==q['hashes'][label]
                actions.extend(f['actions'][:5])
            counts['policy_queries']+=1
        np.testing.assert_array_equal(actions,applied[:n]);counts['new_episodes']+=1
    for line in (S/'view_audit/predictions.jsonl').read_text().splitlines():
        r=json.loads(line);info=r['reply']['inference']
        assert info['revision']=='ebb281ec70b05090aa6165b016eac8ec08e71b17'
        assert hashlib.sha256(info['prompt'].encode()).hexdigest()==info['prompt_sha256']
        for path,meta in zip(r['images'],info['images'],strict=True):
            with Image.open(ROOT/path) as im:assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==meta['pixel_sha256']
        counts['image_checks']+=1
    live=json.loads((S/'LIVE_SUMMARY.json').read_text())
    assert all(p['inputs_equal'] and p['max_action_difference']==0 for p in live['pilot']['prefixes'])
    counts['matched_live_prefixes']=len(live['pilot']['prefixes'])
    (S/'AUDIT.json').write_text(json.dumps(dict(passed=True,counts=counts,physics_rescore=False),indent=2))
    print(json.dumps(counts))

if __name__=='__main__':main()
