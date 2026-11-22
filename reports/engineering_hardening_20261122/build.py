"""Build the plain Chinese stage report using Pandoc and Tectonic."""
from pathlib import Path
import argparse
import re
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pandoc',default='pandoc')
    parser.add_argument('--tectonic',default='tectonic')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    subprocess.run([args.pandoc,'REPORT.md','--syntax-highlighting=none','-o','body.tex'],cwd=root,check=True)
    header=r'''\documentclass[UTF8,a4paper,11pt,fontset=fandol]{ctexart}
\usepackage[margin=22mm]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setmonofont{DejaVu Sans Mono}[Scale=0.8]
\usepackage{graphicx,longtable,booktabs,array,calc,fvextra,xurl,hyperref}
\hypersetup{hidelinks}
\setcounter{secnumdepth}{0}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\DefineVerbatimEnvironment{verbatim}{Verbatim}{breaklines=true,fontsize=\small}
\emergencystretch=4em
\sloppy
\begin{document}
'''
    body=(root/'body.tex').read_text().replace('\\LTcaptype{none}','\\LTcaptype{table}')
    body=re.sub(r'\\texttt\{([^{}]*)\}',lambda match:'\\path|'+match[1].replace('\\_','_')+'|',body)
    (root/'main.tex').write_text(header+body+'\n\\end{document}\n')
    result=subprocess.run([args.tectonic,'--keep-logs','main.tex'],cwd=root,capture_output=True,text=True)
    (root/'build.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError((result.stdout+result.stderr)[-3000:])
    (root/'REPORT.pdf').write_bytes((root/'main.pdf').read_bytes())
    print(root/'REPORT.pdf')


if __name__=='__main__':
    main()
