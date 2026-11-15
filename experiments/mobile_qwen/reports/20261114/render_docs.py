"""Render the two new Markdown documents using local Chrome and CJK fonts.
Usage: PYTHONPATH=/tmp/mobile-qwen-doclibs python3 render_docs.py
"""
from pathlib import Path
import subprocess
import markdown

HERE = Path(__file__).resolve().parent
STYLE = '''@page{size:A4;margin:18mm 17mm}body{font-family:"Noto Sans CJK SC",sans-serif;color:#243448;font-size:9.7pt;line-height:1.65}h1{font-size:23pt;line-height:1.4;margin-bottom:10mm}h2{font-size:15pt;margin-top:8mm;break-after:avoid}h3{font-size:12pt;break-after:avoid}table{border-collapse:collapse;width:100%;font-size:8.5pt;break-inside:avoid;margin:5mm 0}th,td{border:1px solid #c7d0db;padding:5px;text-align:left}th{background:#e9eff4}pre{font-size:8pt;white-space:pre-wrap;background:#f2f5f7;padding:10px}code{overflow-wrap:anywhere;font-family:"Noto Sans Mono CJK SC",monospace;font-size:8pt}p{orphans:3;widows:3}a{color:#365e85;overflow-wrap:anywhere}img{max-width:100%;max-height:75mm}'''
for stem in ['STAGE_REPORT','PAPER_v0.2']:
    source=HERE/(stem+'.md')
    body=markdown.markdown(source.read_text(),extensions=['tables','fenced_code','sane_lists'])
    html=HERE/(stem+'.html')
    html.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+stem+'</title><style>'+STYLE+'</style><body>'+body+'</body></html>')
    subprocess.run(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--no-pdf-header-footer','--print-to-pdf='+str(HERE/(stem+'.pdf')),html.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    print(HERE/(stem+'.pdf'))
