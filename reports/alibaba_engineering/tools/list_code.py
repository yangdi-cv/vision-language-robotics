"""Inventory referenced own/mixed source trees without copying implementations."""
from pathlib import Path
import csv,json,hashlib
D=Path(__file__).resolve().parents[1];R=D.parents[1]
stages=json.loads((D/'outline.json').read_text());found={}
skip={'third_party','runs','results','data','datasets','models','weights','checkpoints','archive','archive_live','cache','__pycache__','figures','logs','.git','.venv','site-packages','records','lora_v1','lora_2b','nf4_2b','nf4_4b'}
for s in stages:
 for name in s['code']:
  p=R/name
  if not p.exists():raise FileNotFoundError(p)
  candidates=[p] if p.is_file() else p.rglob('*')
  for f in candidates:
   if not f.is_file() or f.suffix not in {'.py','.sh','.yaml','.yml','.xml','.urdf','.srdf'}:continue
   rel=f.relative_to(R)
   if any(part in skip or part.startswith('archive') for part in rel.parts):continue
   key=str(rel);found.setdefault(key,{'stages':set(),'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})['stages'].add(s['id'])
with (D/'code_inventory.csv').open('w',newline='') as stream:
 writer=csv.writer(stream);writer.writerow(['source_path','stage_ids','source_category','bytes','sha256'])
 for path,v in sorted(found.items()):writer.writerow([path,','.join(sorted(v['stages'])),'vendor_or_mixed' if path.startswith('SO101/') else 'project_source_to_review',v['size'],v['sha256']])
print('Indexed',len(found),'unique source/config files; implementations not moved.')
