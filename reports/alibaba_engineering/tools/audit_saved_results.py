"""Read saved results only; never launch an experiment or modify old outputs."""
from pathlib import Path
from collections import Counter
import json,hashlib
D=Path(__file__).resolve().parents[1];R=D.parents[1]
rows=[]
for name,rel,n,expected in [('official_baseline','results/baseline',120,117),('chinese_flat','hierarchical/zh_eval_flat',30,22),('chinese_hierarchy_v3','hierarchical/zh_eval_hier_v3',30,28),('translation','hierarchical/zh_eval_translate',30,29)]:
 files=sorted((R/'experiments/vla_libero'/rel).rglob('result.json'))
 values=[json.loads(p.read_text()) for p in files]
 successes=sum(x['success'] is True for x in values)
 assert (len(files),successes)==(n,expected),(name,len(files),successes)
 rows.append(dict(name=name,episodes=len(files),successes=successes,files=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
import numpy as np
probe=R/'experiments/vla_libero/target_skill/deployment_probe'
diffs=[]
for i in range(6):
 u=np.load(probe/'unmerged'/f'case{i}.npz');f=np.load(probe/'fixed'/f'case{i}.npz')
 diffs.append(dict(case=i,old_max_abs=float(np.max(np.abs(u['actions']-u['stored_old']))),fixed_max_abs=float(np.max(np.abs(u['actions']-f['actions'])))))
rows.append(dict(name='merge_numeric_probe',cases=diffs,closed_loop_retested=False))
(D/'RESULT_AUDIT.json').write_text(json.dumps(dict(scope='Saved official baseline and Chinese hierarchy/translation results; not an audit of every historical metric.',checks=rows),ensure_ascii=False,indent=2))
print([(x['name'],x.get('successes'),x.get('episodes')) for x in rows])
