"""Build the one-page summary, detailed appendix, source and standalone adapter packages."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile
from markdown_it import MarkdownIt
from pypdf import PdfReader, PdfWriter

R = Path(__file__).resolve().parents[1]
V = R.parent
ROOT = R.parents[2]
OUT = ROOT / "deliverables/20261112_阿里_闭环恢复与检查器"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    css = """@page{size:A4;margin:15mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:10pt;line-height:1.55;color:#243448}h1{font-size:21pt;line-height:1.35}h2{font-size:14pt;margin-top:6mm;break-after:avoid}h3{font-size:11pt;break-after:avoid}table{width:100%;border-collapse:collapse;font-size:9pt;margin:3mm 0}td,th{border:1px solid #cbd5df;padding:4px 6px}th{background:#edf2f7}tr{break-inside:avoid}p{orphans:3;widows:3}img{max-width:100%;max-height:95mm}p:has(img){break-inside:avoid;text-align:center}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:8px;font-size:8pt;line-height:1.4}code{font-family:"Noto Sans Mono CJK SC",monospace;font-size:8pt;overflow-wrap:anywhere}li{margin:1.5mm 0}a{color:#275d8d;text-decoration:none}"""
    for stem in ["SUMMARY", "REPORT"]:
        h = (
            MarkdownIt("commonmark", {"html": True})
            .enable("table")
            .render((R / (stem + ".md")).read_text())
        )
        page = R / (stem + ".html")
        page.write_text(
            '<meta charset="utf-8"><title>Qwen分层闭环与完成检查器</title><style>'
            + css
            + "</style>"
            + h
        )
        subprocess.run(
            [
                "/usr/bin/google-chrome",
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--print-to-pdf=" + str(R / (stem + ".pdf")),
                page.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
    assert len(PdfReader(R / "SUMMARY.pdf").pages) == 1, "Summary must fit one page"
    w = PdfWriter()
    w.append(R / "SUMMARY.pdf", outline_item="一页阶段摘要")
    w.append(R / "REPORT.pdf", outline_item="本轮实现与独立测试")
    w.append(V / "recovery_study/REPORT.pdf", outline_item="上一轮失败诊断（历史版本）")
    report = OUT / "Qwen分层闭环与完成检查器_汇总报告.pdf"
    w.write(report)
    shutil.copy2(R / "SUMMARY.pdf", OUT / "一页阶段摘要.pdf")
    shutil.copy2(R / "SUMMARY.md", OUT / "一页阶段摘要.md")
    shutil.copy2(R / "REPORT.md", OUT / "本轮实现与测试.md")
    shutil.copy2(
        V / "hierarchical/recovery_contract_smoke/libero_10/task00_init00/rollout.mp4",
        OUT / "01_分层闭环_双物体入篮成功.mp4",
    )
    shutil.copy2(
        R / "results/injection/task00_init02/full/rollout.mp4",
        OUT / "02_受控扰动_整句重试成功.mp4",
    )
    sources = [
        V / "scripts" / n
        for n in [
            "run_hierarchical.py",
            "run_hierarchical_recovery.py",
            "qwen_planner.py",
            "qwen_planner_recovery.py",
            "subgoal_state.py",
            "evaluate.py",
            "python_shared",
        ]
    ]
    sources += [
        p
        for p in (R / "scripts").glob("*.py")
        if p.name
        not in ["initial_summary.py", "train_when_free.py", "calibrate_when_free.py"]
    ]
    sources += list((R / "archive").glob("*.py")) + list((R / "figures").glob("*.png"))
    sources += [
        R / n
        for n in [
            "SUMMARY.md",
            "REPORT.md",
            "README.md",
            "MODEL_CARD.md",
            "environment.json",
            "verification.json",
            "timeboxes.json",
            "t9_protocol.json",
            "t9_results.json",
            "t9_conclusion.json",
            "injection_protocol.json",
            "injection_results.json",
            "injection_summary.json",
            "calibration_protocol.json",
            "data/cases.json",
            "data/audit.json",
        ]
    ]
    sources += [
        R / "results/checker_lora" / n
        for n in [
            "protocol.json",
            "history.json",
            "summary.json",
            "calibrated_summary.json",
            "base_test.json",
            "adapter_test.json",
            "base_scores.json",
            "adapter_scores.json",
            "reload_check.json",
            "cli_reload.json",
        ]
    ]
    sources += [
        ROOT / "perception/qwen_scene" / n for n in ["engine.py", "worker_client.py"]
    ] + [V / "third_party/sources.json", V / "requirements.lock.txt"]
    sources = sorted(set(sources))
    sourcezip = OUT / "闭环恢复与检查器_源码.zip"
    with zipfile.ZipFile(sourcezip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            assert p.is_file(), p
            z.write(p, str(p.relative_to(ROOT)))
        z.writestr(
            "README.md",
            "本包为VLA工程增量源码、协议、索引及评测结果，不含基座模型、环境、720组图像或原始仿真资产。环境版本见environment.json；运行命令见closed_loop_iteration/REPORT.md。LoRA权重单独提供。训练数据可由已生成的baseline视频和trace按prepare_checker_data.py重新提取。统一入口run_hierarchical.py，原recovery入口仅兼容。\n",
        )
    with zipfile.ZipFile(sourcezip) as z:
        assert z.testzip() is None
    weights = OUT / "完成检查器_LoRA权重.zip"
    with zipfile.ZipFile(weights, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((R / "results/checker_lora/best_adapter").iterdir()):
            if p.is_file():
                z.write(p, "completion_checker/best_adapter/" + p.name)
        for n in ["summary.json", "calibrated_summary.json", "protocol.json"]:
            z.write(R / "results/checker_lora" / n, "completion_checker/" + n)
        z.write(R / "MODEL_CARD.md", "completion_checker/MODEL_CARD.md")
    with zipfile.ZipFile(weights) as z:
        assert z.testzip() is None
    (OUT / "说明.txt").write_text(
        "先读一页摘要。汇总报告按摘要、本轮实现与测试、上一轮历史诊断排序。01是分层正常任务成功；02是π0.5整句重试恢复成功，不是Qwen重规划成功。源码与LoRA权重分开打包，基座模型另备。该检查器边界误报仍高，未接入动作切换。\n"
    )
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.iterdir()
        if p.is_file() and p.name != "SHA256.json"
    }
    (OUT / "SHA256.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    pages = len(PdfReader(report).pages)
    (R / "export_status.json").write_text(
        json.dumps(
            {
                "report_pages": pages,
                "summary_pages": 1,
                "source_entries": len(sources) + 1,
                "output": str(OUT),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "report_pages": pages,
                "source_entries": len(sources) + 1,
                "output": str(OUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
