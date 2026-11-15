"""Freeze/copy the exact historical 350-frame subset; no model inference."""
from pathlib import Path
import hashlib,json,shutil
Q=Path(__file__).resolve().parents[3];S=Path(__file__).resolve().parents[1]
old=json.loads((Q/'affordance/eval_lora_v1_test.json').read_text())
ids={x['id'] for x in old};assert len(ids)==350 and len(old)==1014
rows=[json.loads(x) for x in (Q/'affordance/data_v1/labels.jsonl').read_text().splitlines()]
rows=[x for x in rows if x['id'] in ids]
assert len(rows)==350 and {x['initial'] for x in rows}==set(range(13))
out=S/'test350';out.mkdir(exist_ok=False);(out/'rgb').mkdir();(out/'aux').mkdir()
(out/'labels.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
for x in rows:
    for folder,ext in [('rgb','.png'),('aux','.npz')]:shutil.copyfile(Q/'affordance/data_v1'/folder/(x['id']+ext),out/folder/(x['id']+ext))
(out/'MANIFEST.json').write_text(json.dumps({'frames':350,'queries':1014,'initials':list(range(13)),'sha256':{str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}},indent=2))
print(out)
