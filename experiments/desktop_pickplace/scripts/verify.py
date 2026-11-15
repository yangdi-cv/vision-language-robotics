"""Check dataset boundaries, pose labels, model interfaces and exported media."""
from pathlib import Path
import ast,json,re,hashlib
import numpy as np,av
ROOT=Path(__file__).resolve().parents[1];counts={'scripts':0,'videos':0,'demonstrations':0,'links':0}
for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text());counts['scripts']+=1
manifest=json.loads((ROOT/'data/demonstrations/manifest.json').read_text());seeds={s:set() for s in ('train','validation','test')};total=0
for item in manifest['items']:
 with np.load(ROOT/'data/demonstrations'/item['file']) as d:
  n=item['frames'];assert d['images'].shape==(n,240,320,3);assert d['states'].shape==(n,12);assert d['actions'].shape==(n,8);assert np.isfinite(d['states']).all() and np.isfinite(d['actions']).all();assert np.max(abs(np.linalg.norm(d['actions'][:,3:7],axis=1)-1))<1e-5;assert np.all(np.diff(d['sim_time_s'])>0)
 seeds[item['split']].add(item['seed']);counts['demonstrations']+=1;total+=n
assert not seeds['train']&seeds['validation'] and not seeds['train']&seeds['test'] and not seeds['test']&seeds['validation'];assert total==manifest['frames']==26466
for name,expected in [('policy_test60_rgb',20),('policy_test60_rgbd',60)]:
 audit=json.loads((ROOT/'runs'/name/'release_audit.json').read_text());assert audit['cases']==60 and audit['strict_successes']==expected and audit['max_position_replay_error_m']==0
assert json.loads((ROOT/'runs/live_demo_blue/summary.json').read_text())['success']
assert not json.loads((ROOT/'runs/live_demo_ambiguous/summary.json').read_text())['executed']
assert json.loads((ROOT/'runs/batch002/scoring_audit.json').read_text())['passed']==96
for p in (ROOT/'runs').rglob('*.mp4'):
 with av.open(str(p)) as c:frames=sum(1 for _ in c.decode(video=0))
 assert frames>100,p;counts['videos']+=1
for p in ROOT.glob('*.md'):
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' not in link:assert (p.parent/link).exists(),link;counts['links']+=1
(ROOT/'VERIFICATION.json').write_text(json.dumps({'checks':counts,'frames':total,'seed_sets':{k:sorted(v) for k,v in seeds.items()},'action_dimensions':8,'state_dimensions':[12,15],'passed':True},indent=2));print(counts)
