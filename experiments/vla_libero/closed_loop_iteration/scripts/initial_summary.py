from pathlib import Path
import shutil
import subprocess
from markdown_it import MarkdownIt
from pypdf import PdfReader, PdfWriter


def main():
    R = Path(__file__).resolve().parents[1]
    V = R.parent
    OUT = V.parents[1] / "deliverables/20261112_阿里_闭环恢复与检查器"
    css = '@page{size:A4;margin:15mm}body{font-family:"Noto Sans CJK SC";font-size:10pt;line-height:1.55;color:#243448}h1{font-size:21pt}table{width:100%;border-collapse:collapse}td,th{border:1px solid #cbd5df;padding:5px}th{background:#edf2f7}li{margin:3mm 0}code{font-size:8pt}'
    h = (
        MarkdownIt("commonmark")
        .enable("table")
        .render((R / "SUMMARY_INITIAL.md").read_text())
    )
    p = R / "summary_initial.html"
    p.write_text('<meta charset="utf-8"><style>' + css + "</style>" + h)
    subprocess.run(
        [
            "/usr/bin/google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--print-to-pdf=" + str(R / "SUMMARY_INITIAL.pdf"),
            p.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert len(PdfReader(R / "SUMMARY_INITIAL.pdf").pages) == 1
    w = PdfWriter()
    w.append(R / "SUMMARY_INITIAL.pdf")
    w.append(V / "recovery_study/REPORT.pdf")
    w.write(OUT / "阶段摘要与诊断附录.pdf")
    shutil.copy2(R / "SUMMARY_INITIAL.pdf", OUT / "一页阶段摘要.pdf")
    shutil.copy2(
        V / "hierarchical/recovery_contract_smoke/libero_10/task00_init00/rollout.mp4",
        OUT / "01_分层闭环_双物体入篮成功.mp4",
    )
    print("One-page summary and success-first demonstration ready")


if __name__ == "__main__":
    main()
