"""Check report coverage and source/media alignment; no model or GPU required."""
from pathlib import Path
import re,json,hashlib,collections,subprocess
E=Path(__file__).resolve().parent;D=E.parent
sources=sorted((D/'chapters').glob('*.tex'))+sorted((D/'appendices').glob('*.tex'))+[D/'references.tex']
protected=re.compile(r'\\(?:label|cite|ref|url|hyperlink)\{[^}]*\}|\\path\|[^|]*\|')
figs=re.compile(r'\\(?:stagefigure|motionfigure)\{([^}]+)\}')
video=re.compile(r'\\stagevideo\{([^}]+)\}')
summary={'edition':'Complete English v0.2','translated_files':len(sources),'files':[]}
all_zh=[];all_en=[]
for s in sources:
 rel=s.relative_to(D);p=E/rel;assert p.is_file(),p;a=s.read_text();b=p.read_text();assert not re.search(r'[\u3400-\u9fff]',b.replace(r'deliverables/20261113\_阿里\_代码修订/阿里\_修订源码.zip','')),p
 assert collections.Counter(protected.findall(a))==collections.Counter(protected.findall(b)),('Protected references',p)
 assert collections.Counter(figs.findall(a))==collections.Counter(figs.findall(b)),('Figures',p)
 assert collections.Counter(video.findall(a))==collections.Counter(video.findall(b)),('Videos',p)
 for cmd in ['chapter','section','subsection','subsubsection','paragraph']:
  pattern=r'\\'+cmd+r'\*?\{';assert len(re.findall(pattern,a))==len(re.findall(pattern,b)),(p,cmd)
 assert a.count('\\begin{longtable}')==b.count('\\begin{longtable}'),p
 assert a.count('&')==b.count('&'),('Table cells',p)
 summary['files'].append({'source':str(rel),'source_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'english_sha256':hashlib.sha256(p.read_bytes()).hexdigest()});all_zh.append(a);all_en.append(b)
a='\n'.join(all_zh);b='\n'.join(all_en)
media=json.loads((D/'asset_manifest.json').read_text())
for p in set(figs.findall(b)+video.findall(b)):assert (D/p).is_file(),p
info=subprocess.check_output(['pdfinfo',str(E/'build/main_en.pdf')],text=True);pages=int(re.search(r'Pages:\s*(\d+)',info)[1])
summary.update(chapters=8,appendices=4,stages=24,longtables=b.count('\\begin{longtable}'),unique_figure_assets=len(set(figs.findall(b))),video_references=len(set(video.findall(b))),english_pdf_pages=pages,english_pdf_sha256=hashlib.sha256((E/'build/main_en.pdf').read_bytes()).hexdigest(),chinese_pdf_sha256=hashlib.sha256((D/'build/main_zh.pdf').read_bytes()).hexdigest(),original_embedded_screenshot_text_retained=True)
diagnostics=E/'BUILD_DIAGNOSTICS.json'
if (E/'build/main_en.log').exists():
 log=(E/'build/main_en.log').read_text(errors='replace')
 assert 'Missing character:' not in log,'Missing font glyphs'
 assert 'Undefined control sequence' not in log,'Undefined LaTeX'
 assert 'There were undefined references' not in log,'Unresolved references'
else:
 d=json.loads(diagnostics.read_text())
 assert d['pdf_sha256']==summary['english_pdf_sha256'],'Build diagnostics belong to a different PDF'
 assert d['missing_font_glyphs']==d['undefined_control_sequences']==d['undefined_references']==0
summary['checks']='All source files, section counts, table cells, figure/video paths, citations, labels and font glyphs passed.'
(E/'VALIDATION.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2))
