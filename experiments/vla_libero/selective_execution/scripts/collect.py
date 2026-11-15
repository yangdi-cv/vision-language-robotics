"""Archive real visual decisions; evaluation images remain evaluation-only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

S=Path(__file__).resolve().parents[1]


def main():
    samples=[]
    for phase in ['pilot','eval']:
        for p in sorted((S/f'episodes/{phase}').glob('*/visual/result.json')):
            r=json.loads(p.read_text())
            if r['status']!='complete':continue
            with np.load(p.parent/'trace.npz') as f:t={key:f[key] for key in ['target_in','target_free','protected_displacement']}
            for e in r['events']:
                if e['kind']!='check':continue
                step=e['step'];correct=r['selected_target']==r['case']['target']
                hashes=[]
                for name in e['sent_images']:
                    with Image.open(name) as im:hashes.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
                samples.append(dict(case=r['case']['key'],phase=phase,step=step,
                    images=[str(Path(n).relative_to(S)) for n in e['sent_images']],pixel_sha256=hashes,
                    prompt=e['reply']['inference']['prompt'],prompt_sha256=e['reply']['inference']['prompt_sha256'],
                    goal_done_label=bool(t['target_in'][step-1] and t['target_free'][step-1]) if correct else None,
                    protected_constraint_broken=bool(t['protected_displacement'][:step].max()>.03),
                    prediction=e['reply'].get('done'),confirmed_stop=e['decision']['stop'],
                    label_source='recorded In predicate and gripper-contact label; per-frame cached state, not human observability',
                    observability='not_reviewed',
                    split='development' if phase=='pilot' else 'held_out_evaluation_do_not_train'))
    (S/'DECISION_FRAMES.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2))
    (S/'DECISION_SUMMARY.json').write_text(json.dumps({phase:dict(n=sum(r['phase']==phase for r in samples),positive=sum(r['phase']==phase and r['goal_done_label'] is True for r in samples),negative=sum(r['phase']==phase and r['goal_done_label'] is False for r in samples)) for phase in ['pilot','eval']},indent=2))


if __name__=='__main__':main()
