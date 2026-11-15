"""CPU protocol validation, no weights or training/test outputs generated."""
import json,random,sys
from pathlib import Path
Q=Path(__file__).resolve().parents[3];S=Path(__file__).resolve().parents[1]
# Same applicability rules as existing evaluator, without GPU imports.
rows=[json.loads(x) for x in (Q/'affordance/data_train_v1/labels.jsonl').read_text().splitlines()]
train=[r for r in rows if not 13<=r['initial']<=16]
val=[r for r in rows if 13<=r['initial']<=16 and r['handle_mask']['pixels']>=50]
val=random.Random(20261114).sample(val,min(48,len(val)))
old=json.loads((Q/'affordance/lora_v1/protocol.json').read_text())
assert [r['id'] for r in val]==old['val_ids']
assert len(train)==old['train_frames']==780
assert {r['initial'] for r in train}==set(range(17,50))
test=[json.loads(x) for x in (S/'test350/labels.jsonl').read_text().splitlines()]
assert len(test)==350 and {r['initial'] for r in test}==set(range(13))
assert not {r['id'] for r in test}&{r['id'] for r in train+val}
print('Same training/validation split and validation IDs; exact test350; no initial overlap: PASS')
