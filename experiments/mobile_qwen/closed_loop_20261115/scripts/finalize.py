"""Wait for the bounded paired batch, then build its stage deliverables."""
from pathlib import Path
import json,time,subprocess,sys
D=Path(__file__).resolve().parents[1];ROOT=D.parents[2]
for _ in range(240):
 p=D/'results/paired_v1_summary.json'
 try:rows=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):rows=[]
 if len(rows)==12:break
 active=subprocess.run(['systemctl','--user','is-active','ali-precontact-paired-20261115'],capture_output=True,text=True).stdout.strip()
 if active!='active':raise RuntimeError(f'Batch stopped incomplete ({len(rows)}/12)')
 time.sleep(15)
else:raise TimeoutError('Paired experiment exceeded one hour')
subprocess.run([sys.executable,str(D/'scripts/audit_results.py')],check=True)
subprocess.run([sys.executable,str(D/'scripts/build_report.py')],check=True)
x=json.loads((D/'SUMMARY.json').read_text());log=ROOT/'PROJECT_LOG.md';marker='### 2026-11-15 导航与可供性预接触配对结果'
if marker not in log.read_text():
 with log.open('a') as f:
  f.write('\n'+marker+'\n\n')
  for mode,m in x['metrics'].items():f.write(f"{mode}: 导航{m['navigation_success']}/{m['n']}，预接触{m['success']}/{m['n']}，到位误差中位{m['median_error_mm']:.1f}mm，清理异常{m['cleanup_failures']}。\n")
  f.write('\n报告与源码见experiments/mobile_qwen/closed_loop_20261115/REPORT.pdf、SOURCE.zip。单厨房、真值深度/碰撞地图、理想导航定位、固定预接触模板；未完成抓取/原LIBERO整任务，不连接机械臂，端侧暂缓。\n')
print('Finalized',D,flush=True)
