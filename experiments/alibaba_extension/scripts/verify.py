"""Validate result consistency, media, saved quantization, dataset and doc links."""
from pathlib import Path
import json,ast,re,sqlite3,hashlib
import numpy as np,av
ROOT=Path(__file__).resolve().parents[1];runs=ROOT/'runs';report={'scripts':0,'videos':0,'relative_links':0,'dataset_images':30,'errors':[]}
for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text());report['scripts']+=1
for name,successes in [('act_improved_test30',16),('act_original_test30',8),('act_original_best_test30',8),('act_public_test30',25)]:
 p=runs/name;m=json.loads((p/'metrics.json').read_text());traces=[json.loads(x.read_text()) for x in p.glob('rollout_*.json')];assert len(traces)==30 and sum(x['summary']['success'] for x in traces)==successes==m['successes'];assert {x['summary']['seed'] for x in traces}==set(range(3000,3030))
for p in runs.rglob('*.mp4'):
 with av.open(str(p)) as c:n=sum(1 for _ in c.decode(video=0))
 assert n>=50,p;report['videos']+=1
cases=json.loads((ROOT/'data/rgbd_cases/cases.json').read_text());assert len(cases)==120
hashes=set()
for index in range(30):
 data=ROOT/'data/rgbd_cases';hashes.add(hashlib.sha256((data/f'scene_{index:03d}.png').read_bytes()).hexdigest());depth=np.load(data/f'depth_{index:03d}.npy');mask=np.load(data/f'mask_{index:03d}.npy');metadata=json.loads((data/f'scene_{index:03d}.json').read_text());assert depth.shape==mask.shape==(480,640) and mask.sum()==metadata['visible_pixels'];assert np.isfinite(depth).all() and abs(float(np.median(depth[mask]))-metadata['target_depth_m'])<1e-7
assert len(hashes)==30
assert json.loads((runs/'profile_int8_cache/reload_check.json').read_text())['fresh_process_reloaded']
assert json.loads((runs/'qwen_nav_act_001/summary.json').read_text())['workflow_success']
assert json.loads((runs/'qwen_nav_act_001/workflow/rosbag_inventory.json').read_text())['sqlite_integrity']=='ok'
for p in ROOT.glob('*.md'):
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in link:continue
  assert (p.parent/link).exists(),f'{p}: missing {link}';report['relative_links']+=1
(ROOT/'VERIFICATION.json').write_text(json.dumps(report,indent=2));print(report)
