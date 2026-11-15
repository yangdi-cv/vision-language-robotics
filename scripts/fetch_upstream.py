"""Restore the recorded upstream revisions; no weights or datasets are downloaded."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,shutil
R=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git',*map(str,args)],text=True,env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'}).strip()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',choices=['openpi','libero','lerobot','all'],default='all');args=ap.parse_args()
 base=R/'experiments/vla_libero/third_party';sources=json.loads((base/'sources.json').read_text());partial=json.loads((base/'partial_sources.json').read_text())
 for name,rec in sources.items():
  if args.project not in ['all',name]:continue
  dst=base/name
  if (dst/'.git').exists():
   if git('-C',dst,'rev-parse','HEAD')!=rec['revision']:raise RuntimeError(f'{dst}: existing checkout differs from recorded revision')
   print(name,'already at recorded revision');continue
  if dst.exists():
   expected=partial.get(name,{})
   actual={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*') if p.is_file()}
   if actual!=expected:raise RuntimeError(f'{dst}: unrecognized or edited files; refusing to replace')
   shutil.rmtree(dst)
  git('init',dst);git('-C',dst,'remote','add','origin','https://github.com/'+rec['repo']+'.git');git('-C',dst,'fetch','--depth','1','origin',rec['revision']);git('-C',dst,'checkout','--detach','FETCH_HEAD')
  assert git('-C',dst,'rev-parse','HEAD')==rec['revision'];print(name,rec['revision'])
if __name__=='__main__':main()
