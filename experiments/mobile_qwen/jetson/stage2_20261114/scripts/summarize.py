"""Read-only paired scoring, including strict masks and output agreement."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

S=Path(__file__).resolve().parents[1];Q=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
expected=json.loads((Q/'affordance/eval_lora_v1_test.json').read_text());keys={(x['id'],x['query']) for x in expected}
def key(x):return x['id'],x['query']
def metrics(rows):
    assert len(rows)==1014 and {key(x) for x in rows}==keys
    result={}
    for query in ['handle','bowl','place','cabinet']:
        sub=[x for x in rows if x['query']==query]
        if query=='cabinet':
            result[query]={'n':len(sub),'mean_iou':float(np.mean([x.get('score',{}).get('rel1000',{}).get('iou',0) for x in sub]))};continue
        strict=0
        for x in sub:
            if 'parsed' not in x:continue
            uv=np.array(x['parsed'],float)/1000*512
            if uv.shape!=(2,) or not np.isfinite(uv).all() or not ((uv>=0)&(uv<512)).all():continue
            u,v=map(int,uv)
            label={'handle':1,'bowl':3,'place':2}[query]
            with np.load(S/'test350/aux'/f"{x['id']}.npz") as aux:
                strict+=bool(aux['cls'][v,u]==label)
        scores=[x.get('score',{}).get('rel1000',{}) for x in sub]
        valid3d=[x['err3d_cm'] for x in scores if 'err3d_cm' in x]
        result[query]={'n':len(sub),'strict_hits':strict,'strict_rate':strict/len(sub),'near_hits':sum(bool(x.get('hit')) for x in scores),'near_rate':sum(bool(x.get('hit')) for x in scores)/len(sub),'valid_3d':len(valid3d),'median_3d_cm':float(np.median(valid3d)) if valid3d else None,'parse_fail':sum('parsed' not in x for x in sub)}
    return result

def agreement(pc,j):
    assert {key(x) for x in pc}=={key(x) for x in j}
    refs={key(x):x for x in pc};raw=parsed=0
    for x in j:
        r=refs[key(x)];raw+=x['raw']==r['raw'];parsed+=x.get('parsed') is not None and x.get('parsed')==r.get('parsed')
    return {'n':len(j),'raw_equal':raw,'raw_rate':raw/len(j),'parsed_equal':parsed,'parsed_rate':parsed/len(j)}
results={}
old=json.loads((Q/'affordance/eval_base_v2.json').read_text())
for name,rows in [('pc_4b_base_bf16_512',[x for x in old if key(x) in keys]),('pc_4b_lora_bf16_512',expected)]:
    results[name]={'historical':True,'metrics':metrics(rows)}
for f in (S/'evaluations').glob('*.json'):
    if any(f.name.endswith(suffix) for suffix in ['.resources.json','.deferred.json','.load.json']):continue
    rows=json.loads(f.read_text())
    if len(rows)!=1014:continue
    r=f.with_suffix('.resources.json')
    results[f.stem]={'metrics':metrics(rows),'resources':json.loads(r.read_text()) if r.exists() else None}
diag=S/'diagnostic_base_2b_partial_coordinates.json'
if diag.exists():
    results['diagnostic_partial_coordinates_base_2b']={'formal':False,'note':'CPU extraction from truncated saved responses; not valid production JSON','metrics':metrics(json.loads(diag.read_text()))}
for name,r in results.items():
    if name.startswith('jetson_'):
        pcname='pc_'+name[len('jetson_'):].removesuffix('_noswap')
        if pcname not in results:continue
        pcfile=S/'evaluations'/f'{pcname}.json';jfile=S/'evaluations'/f'{name}.json'
        r['pc_agreement']=agreement(json.loads(pcfile.read_text()),json.loads(jfile.read_text()))
        pm=[json.loads(x) for x in pcfile.with_suffix('.metadata.jsonl').read_text().splitlines()]
        jm=[json.loads(x) for x in jfile.with_suffix('.metadata.jsonl').read_text().splitlines()]
        pk={(x['id'],x['prompt_sha256']):x for x in pm}
        counts={field:sum(x['metadata'][field]==pk[(x['id'],x['prompt_sha256'])]['metadata'][field] for x in jm) for field in ['input_token_sha256','pixel_values_sha256','image_grid_thw','output_token_ids']}
        r['pc_agreement']['input_and_token_equal_counts']=counts
        r['pc_agreement']['inputs_all_identical']=all(counts[k]==len(jm) for k in ['input_token_sha256','pixel_values_sha256','image_grid_thw'])
a.out.write_text(json.dumps(results,indent=2))
print(a.out)
