"""Secondary CPU diagnostic only: extract complete coordinate arrays from saved
truncated responses. Not a production parser or replacement for formal scores.
"""
from pathlib import Path
import json,re,sys
import numpy as np
Q=Path(__file__).resolve().parents[3];S=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(Q/'affordance'),str(Q/'scripts')]
from eval_qwen import score
rows=json.loads((S/'evaluations/pc_2b_base_bf16_512.json').read_text())
labels={x['id']:x for x in [json.loads(s) for s in (S/'test350/labels.jsonl').read_text().splitlines()]}
results=[]
for row in rows:
    x=dict(row)
    if 'error' in row:
        name='bbox_2d' if row['query']=='cabinet' else 'point_2d'
        match=re.search(r'"'+name+r'"\s*:\s*(\[[^\]]+\])',row['raw'])
        if match:
            try:
                v=json.loads(match[1]);assert len(v)==(4 if name=='bbox_2d' else 2)
                assert all(type(n) in (int,float) and np.isfinite(n) for n in v)
                with np.load(S/'test350/aux'/f"{row['id']}.npz") as aux:
                    x['score']=score(row['query'],labels[row['id']],v,aux)
                x['parsed']=v;x['diagnostic_recovered']=True
            except (ValueError,AssertionError,TypeError):pass
    results.append(x)
out=S/'diagnostic_base_2b_partial_coordinates.json'
out.write_text(json.dumps(results))
print('Diagnostic recovered',sum(x.get('diagnostic_recovered',False) for x in results),'responses; formal files unchanged')
