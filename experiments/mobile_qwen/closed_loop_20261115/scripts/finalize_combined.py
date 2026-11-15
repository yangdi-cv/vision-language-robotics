"""Finish the finite same-world validation and record actual full-task outcomes."""
from pathlib import Path
import subprocess,json,time,sys
D=Path(__file__).resolve().parents[1];ROOT=D.parents[2]
for _ in range(240):
 p=D/'results/combined_validation1_summary.json'
 try:rows=json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):rows=[]
 if len(rows)==3:break
 state=subprocess.run(['systemctl','--user','is-active','ali-precontact-combined-validation-20261115'],capture_output=True,text=True).stdout.strip()
 if state!='active':raise RuntimeError(f'Validation ended incomplete {len(rows)}/3')
 time.sleep(15)
else:raise TimeoutError('Validation exceeded one hour')
subprocess.run([sys.executable,str(D/'scripts/audit_results.py')],check=True)
subprocess.run([sys.executable,str(D/'scripts/build_report.py')],check=True)
x=json.loads((D/'COMBINED_SUMMARY.json').read_text());audits=[]
for r in x['rows']:
 f=D/'results'/r['tag'];assert (f/'rollout.mp4').stat().st_size>1000
 if r.get('combined_official_success'):
  h=json.loads((f/'policy_handover.json').read_text());assert h['no_state_reset'];assert h['before']['joint_positions']==h['after']['joint_positions']
  final=json.loads((f/'result.json').read_text());assert final['success']
 audits.append({'tag':r['tag'],'complete_artifacts':True,'joint_handover_checked':(f/'policy_handover.json').exists()})
(D/'COMBINED_AUDIT.json').write_text(json.dumps(audits,indent=2))
marker='### 2026-11-15 同场景完整交接小规模验证结束';log=ROOT/'PROJECT_LOG.md'
if marker not in log.read_text():
 with log.open('a') as f:
  f.write('\n'+marker+f"\n\n共{x['n']}回合（含1个开发探针+3个重复性回合），Qwen视觉与官方整任务同时通过{x['official_and_visual_success']}/{x['n']}，误停{x['false_visual_stops']}。不重置关节、同一世界Nav2→LoRA定位→MoveIt预接触→π0.5操作→Qwen停止。\n")
  for r in x['rows']:f.write(f"初态{r['initial']}: 官方成功{r.get('combined_official_success',False)}，Qwen完成{r.get('visual_milestones_complete',False)}，策略步{r.get('policy_steps')}，确认延迟{r.get('visual_delay_after_first_success_steps')}步，错误{r.get('error')}。\n")
  f.write('\n单场景已有初态、无原生π0.5配对效益对照，不能声称提高整任务成功率。PDF和源码已更新；纯预接触旧版保留REPORT_PRECONTACT.pdf，完整统计单列COMBINED_SUMMARY.json。端侧后置，没有连接实体机械臂。\n')
print('Full-chain finalized',x['official_and_visual_success'],x['n'],flush=True)
