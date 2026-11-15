import json,html,hashlib
from pathlib import Path
S=Path(__file__).resolve().parents[1];V=S.parent;ROOT=V.parents[1]
rows=[]
for p in sorted((S/'runs/fresh_original').glob('task*/result.json')):
 r=json.loads(p.read_text());assert r['status']=='success' and r['success'];r['folder']=str(p.parent.relative_to(S));rows.append(r)
assert len(rows)==6
labels={0:'汤罐与番茄酱放入篮子',1:'奶酪与黄油放入篮子',9:'杯子放进微波炉并关门'}
cards=[]
for r in rows:
 cards.append(f'''<article><h2>{labels[r['task_id']]}</h2><p>初态 {r['initial_state_id']} · {r['steps']} 步 · <b>完成</b></p><video controls preload="metadata" src="{r['folder']}/rollout.mp4"></video><p class="prompt">{html.escape(r['input_prompt'])}</p><details><summary>模型与动作记录</summary><p>{r['query_count']}次图像→动作块查询；首次编译后中位查询 {r['query_median_excluding_first_s']:.3f} 秒。</p><a href="{r['folder']}/result.json">回合结果</a> · <a href="{r['folder']}/queries.json">推理记录</a></details></article>''')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>π0.5 自主任务实例</title><style>body{margin:0;background:#edf2f7;color:#1d3349;font:16px/1.65 "Noto Sans CJK SC",sans-serif}main{max-width:1280px;margin:36px auto;padding:0 24px}h1{font-size:30px;margin:0}header{padding:24px 30px;background:#133d57;color:white;border-radius:18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;margin:24px 0}article{background:white;border-radius:15px;padding:20px;box-shadow:0 2px 12px #1737500a}h2{font-size:20px;margin:0}video{width:100%;border-radius:9px;background:#111}b{color:#15764a}.prompt{font-size:13px;color:#53697a}code{background:#dce7ef;padding:3px 7px}a{color:#236b96}footer{font-size:14px;color:#546879}details{font-size:14px}</style><main><header><h1>π0.5 自主任务实例</h1><p>官方公开权重 · LIBERO-Long · 6 / 6 新回合完成</p><div>输入相机图像与原任务指令 → 模型输出动作块 → MuJoCo执行 → 再取新图推理</div></header><p>这里播放的是本次真实仿真回合录像，按仿真20Hz播放，不包含推理等待。没有人类遥操作、脚本搬运或回放示教来生成动作；官方成功信号仅用于判分与结束回合。不是实体机械臂演示，也不表示任意新任务都能完成。</p><div class="grid">'''+''.join(cards)+'''</div><article><h2>重新运行</h2><p>在项目根目录运行：<code>bash experiments/vla_libero/official_instance/start.sh --tag my_demo --tasks 9 --initials 32</code></p><p>使用新tag避免覆盖记录。通过独立内存限额进程加载原模型，不连接任何实体设备。可改为任务0或1；完整六回合由默认参数运行。</p></article><footer>openpi 与 LIBERO 固定版本沿用现有环境。原始输入、预测动作块和实际施加动作均保存在每个回合目录；本页不是新训练模型的成绩。</footer></main></html>'''
(S/'index.html').write_text(page)
(S/'SUMMARY.json').write_text(json.dumps(dict(episodes=6,successes=6,tasks=[0,1,9],initials=[32,33],fresh=True,policy='official pi05_libero, no adapter',oracle_actions=False,hardware=False,rows=rows),ensure_ascii=False,indent=2))
# Independently check persisted chunks vs actual commands and official final label.
import numpy as np
for r in rows:
 folder=S/r['folder']
 with np.load(folder/'trace.npz') as f,np.load(folder/'action_chunks.npz') as c:
  applied=f['applied'];chunks=c['actions'];planned=chunks[:,:5].reshape(-1,7)[:len(applied)];np.testing.assert_array_equal(applied,planned);assert bool(f['predicate'][-1])
(S/'CHECK.json').write_text(json.dumps(dict(episodes=6,actions_match_model=True,final_official_predicates=True),indent=2))
