"""Check report structure, source references and copied assets."""
from pathlib import Path
import json,hashlib,re
D=Path(__file__).resolve().parents[1];R=D.parents[1]
stages=json.loads((D/'outline.json').read_text());assets=json.loads((D/'asset_manifest.json').read_text())
assert len(stages)==24 and len({s['id'] for s in stages})==24
for ch in range(1,7):assert sum(s['chapter']==ch for s in stages)==4
sections=sum(p.read_text().count('\\section{') for p in (D/'chapters').glob('0[2-7]_*.tex'));assert sections==24,sections
for s in stages:
 assert (D/'code_index'/f"ch{s['report_chapter']:02d}"/(s['id']+'.md')).exists()
 for p in s['code']+s['sources']:assert (R/p).exists(),p
for a in assets:
 p=D/a['local'];assert p.exists(),p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'],p
 assert a['local'].isascii(),a['local']
for tex in D.rglob('*.tex'):
 for p in re.findall(r'\\(?:stagefigure|stagevideo)\{([^}]+)\}',tex.read_text()):assert (D/p).exists(),(tex,p)
assert (D/'build/main_zh.pdf').stat().st_size>10000
print('24 stages, source links, copied media hashes and PDF: OK')
print('Figures:',sum(a['kind']=='figure' for a in assets),'Videos:',sum(a['kind']=='video' for a in assets))

assert json.loads((D/'missing_assets.json').read_text()) == []
chapters='\n'.join(p.read_text() for p in (D/'chapters').glob('*.tex'))
assert len(re.findall(r'\\stagefigure\{',chapters))==60
assert chapters.count('\\begin{longtable}')>=99
for marker in ['本稿先给出','本稿的审阅重点','报告后续完善顺序']:
 assert marker not in chapters,marker
print('Expanded chapters: 60 figure citations, 99+ detailed tables, no draft-only placeholders: OK')
