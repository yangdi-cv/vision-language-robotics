"""Read-only file checks before model evaluation."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('path',type=Path);p.add_argument('--export',action='store_true');a=p.parse_args()
x=json.loads((a.path/('EXPORT.json' if a.export else 'MANIFEST.json')).read_text())
files=x['files'] if a.export else x['sha256']
for name,expected in files.items():
    h=hashlib.sha256()
    with (a.path/name).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==expected,name
print('PASS',len(files),'file hashes')
