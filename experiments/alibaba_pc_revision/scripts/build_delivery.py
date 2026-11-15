from pathlib import Path
import sys, re, subprocess, zipfile, json, hashlib

sys.path.insert(0, "/tmp/task2-tools")
import markdown

R = Path(__file__).resolve().parents[3]
A = R / "experiments/alibaba_pc_revision"
B = R / "experiments/bytedance_week3"
O = R / "deliverables/20261020_PC_阿里与字节"
O.mkdir(exist_ok=True)
style = """@page{size:A4;margin:18mm 16mm 19mm;@bottom-right{content:counter(page);font-size:9pt;color:#6a7b8f}}body{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;font-size:10pt;line-height:1.7;color:#233247}h1{font-size:23pt;line-height:1.4}h2{font-size:15pt;margin-top:7mm;break-after:avoid}table{border-collapse:collapse;width:100%;font-size:8.8pt;break-inside:avoid;margin:4mm 0}th,td{border:1px solid #cad5df;padding:5px 7px;text-align:left;overflow-wrap:anywhere}th{background:#eaf0f5}img{max-width:100%;max-height:108mm;object-fit:contain}p:has(img){text-align:center;break-inside:avoid}pre{background:#f2f5f7;padding:10px;white-space:pre-wrap;font-size:8pt;line-height:1.5}code{font-family:"Noto Sans Mono CJK SC",monospace;font-size:8.6pt;overflow-wrap:anywhere}a{color:#2a6089}.refs{font-size:9pt;color:#466173;border-left:3px solid #d1e2ec;padding-left:9px}.toc{break-before:page}p{orphans:3;widows:3}"""
for task, name in [(A, "阿里_PC反馈改进.pdf"), (B, "字节_Week3柔顺控制.pdf")]:
    md = (task / "REPORT.md").read_text()
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    for image in images:
        assert (task / image).exists(), image
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    body = body.replace("<h2>目录</h2>", '<h2 class="toc">目录</h2>')
    body = re.sub(
        r"<p>(相关代码在.*?)</p>", r'<p class="refs">\1</p>', body, flags=re.S
    )
    page = task / "REPORT.html"
    page.write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>'
        + name[:-4]
        + "</title><style>"
        + style
        + "</style><body>"
        + body
        + "</body></html>"
    )
    subprocess.run(
        [
            "/usr/bin/google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--print-to-pdf=" + str(task / "REPORT.pdf"),
            page.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    (O / name).write_bytes((task / "REPORT.pdf").read_bytes())


def files(folder, allowed=None):
    return [
        p
        for p in folder.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and (allowed is None or allowed(p))
    ]


ali = []
for folder in [
    A / "scripts",
    A / "config",
    R / "perception/qwen_scene",
    R / "perception/web_workbench",
    R / "sensors/imu",
    R / "SO101/code/scripts",
    R / "SO101/code/lerobot/src",
]:
    ali += files(
        folder,
        lambda p: p.suffix in [".py", ".js", ".html", ".css", ".txt", ".json"]
        or p.suffix == ""
        or p.name == "start",
    )
# Only the runtime settings are relevant; exclude hardware-specific saved references/calibration.
ali = [
    p
    for p in ali
    if p.name
    not in [
        "reference_pose.json",
        "build_delivery.py",
        "write_reports.py",
        "run_pc_gpu.py",
    ]
]
for n in ["LICENSE", "pyproject.toml"]:
    ali.append(R / "SO101/code/lerobot" / n)
for folder in [
    R / "experiments/desktop_pickplace/scripts",
    R / "experiments/desktop_pickplace/assets",
]:
    ali += files(folder)
for folder in [
    R / "experiments/alibaba_baseline/scripts",
    R / "experiments/alibaba_extension/scripts",
]:
    ali += files(folder)
ali += [
    A / "README.md",
    R / "perception/web_workbench/README.md",
    R / "perception/qwen_scene/requirements-local.lock.txt",
]
# Bundle the IMU parser dependency at a relative path for the copied web source.
imu_vendor = R / "sensors/imu/vendor/YbImuSerialLib.py"
if imu_vendor.exists():
    ali.append(imu_vendor)
byte = (
    files(B / "scripts")
    + files(B / "assets")
    + files(B / "config")
    + [B / "README.md", B / "requirements.txt"]
)
manifest = []
for name, src in [("阿里_PC源码.zip", ali), ("字节_Week3源码.zip", byte)]:
    path = O / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(set(src)):
            if name == "阿里_PC源码.zip" and p == R / "sensors/imu/server.py":
                text = p.read_text().replace(
                    "Path('/home/yd/Documents/IMU/demo')", "HERE / 'vendor'"
                )
                z.writestr(str(p.relative_to(R)), text)
            else:
                z.write(p, p.relative_to(R))
        if name == "阿里_PC源码.zip" and not imu_vendor.exists():
            z.write(
                R / "experiments/imu_week2/vendor/YbImuSerialLib.py",
                "sensors/imu/vendor/YbImuSerialLib.py",
            )
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
    manifest.append(
        dict(
            file=name,
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    )
(A / "DELIVERY.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps(manifest, ensure_ascii=False, indent=2))
