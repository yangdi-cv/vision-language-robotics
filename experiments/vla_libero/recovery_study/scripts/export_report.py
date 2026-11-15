"""Render the saved Chinese report; package source files without weights or trajectories."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile
from markdown_it import MarkdownIt
from pypdf import PdfReader


def main():
    R = Path(__file__).resolve().parents[1]
    V = R.parent
    ROOT = V.parents[1]
    OUT = ROOT / "deliverables/20261112_阿里_分层执行与恢复分析"
    OUT.mkdir(parents=True, exist_ok=True)
    css = """@page{size:A4;margin:16mm 15mm 18mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:10pt;line-height:1.65;color:#243448}h1{font-size:23pt;line-height:1.4}h2{font-size:15pt;margin-top:7mm;break-after:avoid}h3{font-size:11.5pt;break-after:avoid}p{orphans:3;widows:3}table{border-collapse:collapse;width:100%;font-size:9pt;margin:4mm 0}th,td{border:1px solid #cad4df;padding:5px 7px}th{background:#eaf0f5}thead{display:table-header-group}tr{break-inside:avoid}img{max-width:100%;max-height:115mm;object-fit:contain}p:has(img){text-align:center;break-inside:avoid}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:10px;font-size:8pt;line-height:1.45}code{font-family:"Noto Sans Mono CJK SC",monospace;font-size:8pt;overflow-wrap:anywhere}a{color:#245b8e;text-decoration:none}li{margin:1mm 0}"""
    html = (
        MarkdownIt("commonmark", {"html": True})
        .enable("table")
        .render((R / "REPORT.md").read_text())
    )
    (R / "REPORT.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Qwen分层执行与失败恢复</title><style>'
        + css
        + "</style>"
        + html
    )
    pdf = R / "REPORT.pdf"
    subprocess.run(
        [
            "/usr/bin/google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--print-to-pdf=" + str(pdf),
            (R / "REPORT.html").as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    reader = PdfReader(pdf)
    text = "".join(p.extract_text() for p in reader.pages)
    for phrase in ["38.5", "10432", "155", "26"]:
        assert phrase in text, phrase
    name = "Qwen分层执行与失败恢复_进度报告.pdf"
    shutil.copy2(pdf, OUT / name)
    shutil.copy2(R / "REPORT.md", OUT / "Qwen分层执行与失败恢复_进度报告.md")
    source = []
    for n in [
        "subgoal_state.py",
        "run_hierarchical_recovery.py",
        "qwen_planner_recovery.py",
        "run_hierarchical.py",
        "qwen_planner.py",
        "evaluate.py",
        "python_shared",
    ]:
        source.append(V / "scripts" / n)
    source += list((R / "scripts").glob("*.py")) + list((R / "archive").glob("*.py"))
    source += [
        R / n
        for n in [
            "README.md",
            "REPORT.md",
            "protocol.json",
            "contract_protocol.json",
            "guided_protocol.json",
            "typed_gate_protocol.json",
            "verification.json",
            "audit.json",
            "summary.json",
            "episode_index.csv",
            "monitor_diagnostic.json",
            "plate_contact_diagnostic.json",
            "historical_prefix_comparison.json",
            "source_hashes.json",
        ]
    ]
    source += list((R / "figures").glob("*.png"))
    source += [
        V / "configs/libero_long_zh.json",
        V / "configs/libero_long_tasks.json",
        V / "requirements.lock.txt",
        V / "third_party/sources.json",
        ROOT / "perception/qwen_scene/engine.py",
        ROOT / "perception/qwen_scene/worker_client.py",
    ]
    source = sorted(set(source))
    for p in source:
        assert p.is_file(), p
    sourcezip = OUT / "Qwen分层执行与失败恢复_源码.zip"
    with zipfile.ZipFile(sourcezip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in source:
            z.write(p, str(p.relative_to(ROOT)))
        z.writestr(
            "README.md",
            "本包为现有VLA工程的恢复分析与改进源码，含报告、协议、汇总表和配图。逐回合数组、视频、环境和模型权重未打包。运行前需按原工程目录放置OpenPI/LIBERO固定版本、π0.5检查点与Qwen4B缓存；版本见third_party/sources.json和requirements.lock.txt。具体目录、命令和限制见recovery_study/REPORT.md。所有新增动作均为仿真。\n",
        )
    with zipfile.ZipFile(sourcezip) as z:
        assert z.testzip() is None
        assert not any(
            x.endswith((".npz", ".mp4", ".safetensors", ".whl")) for x in z.namelist()
        )
    # Include three representative videos as separate files, avoiding dependency on external links.
    videos = [
        (
            R / "results/task04_init02/full_seed31001/rollout.mp4",
            "白杯偏心_整句重试.mp4",
        ),
        (
            V
            / "hierarchical/recovery_contract_smoke/libero_10/task00_init00/rollout.mp4",
            "分层新入口_双物体入篮成功.mp4",
        ),
        (
            V
            / "hierarchical/recovery_contract_smoke/libero_10/task09_init00/rollout.mp4",
            "微波炉_视觉误确认失败.mp4",
        ),
    ]
    for p, n in videos:
        shutil.copy2(p, OUT / n)
    (OUT / "说明.txt").write_text(
        "PDF汇总了已有结果、本轮改动、配对恢复实验和失败原因。源码ZIP不含模型、环境和原始轨迹。三个视频分别展示偏心状态恢复失败、新入口入篮成功及微波炉误确认失败；不应把成功示例当作新版整体通过。\n"
    )
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.iterdir()
        if p.is_file() and p.name != "SHA256.json"
    }
    (OUT / "SHA256.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (R / "export_status.json").write_text(
        json.dumps(
            {
                "pages": len(reader.pages),
                "pdf_bytes": pdf.stat().st_size,
                "zip_files": len(source) + 1,
                "output": str(OUT),
                "pdf_text_checked": True,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "pages": len(reader.pages),
                "zip_files": len(source) + 1,
                "output": str(OUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
