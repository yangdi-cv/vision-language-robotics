"""Frozen single-world camera-grounding -> MoveIt -> real joint feedback trials."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time,shutil,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import numpy as np
D=Path(__file__).resolve().parents[1];ROOT=D.parents[2];M=ROOT/'experiments/mobile_vla';V=ROOT/'experiments/vla_libero'
sys.path[:0]=[str(M/'scripts'),str(ROOT/'perception/qwen_scene')]
from rpc import call
from worker_client import read_json_line,close_json_worker
p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--modes',default='oracle,lora,base');p.add_argument('--initials',default='0');p.add_argument('--variants',default='nominal');p.add_argument('--navigate',action='store_true');p.add_argument('--max-corrections',type=int,default=1);a=p.parse_args()
worker=None;workerlog=None;summary=[]
def stop(proc):
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGINT)
  try:proc.wait(timeout=8)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
def ground(image,mode):
 global worker,workerlog
 if worker is None:
  workerlog=(D/'logs'/f'{a.prefix}_qwen.log').open('w')
  worker=subprocess.Popen([str(ROOT/'.venvs/qwen-scene/bin/python'),str(D/'scripts/combined_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=workerlog,text=True,bufsize=1,start_new_session=True)
  assert read_json_line(worker,120).get('ready')
 worker.stdin.write(json.dumps({'image':image,'mode':mode})+'\n');worker.stdin.flush();response=read_json_line(worker,90)
 if not response.get('ok'):raise RuntimeError(f'Grounding failed: {response}')
 return response
try:
 for initial in map(int,a.initials.split(',')):
  for variant in a.variants.split(','):
   for mode in a.modes.split(','):
    tag=f'{a.prefix}_{variant}_i{initial:02d}_{mode}';out=D/'results'/tag;logs=D/'logs'/tag;sock=Path('/tmp/'+tag+'.sock')
    assert not out.exists() and not sock.exists();logs.mkdir()
    env=os.environ.copy();env['MOBILE_VLA_DOMAIN']='119';env['MOBILE_TF_OVERLAY']=''
    processes=[];streams=[];row={'tag':tag,'mode':mode,'initial':initial,'variant':variant,'navigate':a.navigate,'success':False,'precontact_success':False,'success_criterion':'precontact passes, Qwen completes both milestones, official full-task shadow check succeeds','attempts':[]}
    def launch(cmd,name,overlay=False):
     f=(logs/(name+'.log')).open('w');streams.append(f);e=env.copy()
     if overlay:e['MOBILE_TF_OVERLAY']=str(M/'runtime/geometry2_0_25_24/root')
     z=subprocess.Popen(cmd,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);processes.append(z);return z
    try:
     cmd=[str(V/'scripts/python_shared'),str(D/'scripts/sim_combined.py'),'--socket',str(sock),'--out',str(out),'--initial',str(initial),'--variant',variant]
     if not a.navigate:cmd.append('--start-docked')
     sim=launch(cmd,'sim');end=time.monotonic()+90
     while not sock.exists():
      if sim.poll() is not None:raise RuntimeError('Simulator exited')
      if time.monotonic()>end:raise TimeoutError('Simulator startup')
      time.sleep(.2)
     if a.navigate:
      nav=launch([str(M/'scripts/ros-python'),str(D/'scripts/nav_only.py'),'--socket',str(sock),'--out',str(out)],'nav',overlay=True);nav.wait(timeout=350)
      n=json.loads((out/'navigation.json').read_text());row['navigation']=n
      if not n.get('navigation_success'):raise RuntimeError('Navigation did not complete')
     bridge=launch([str(M/'scripts/ros-python'),str(D/'scripts/physics_bridge.py'),'--socket',str(sock),'--out',str(out)],'bridge')
     moveit=launch([str(M/'scripts/ros-python'),'/opt/ros/humble/bin/ros2','launch',str(D/'scripts/moveit.launch.py')],'moveit')
     # The same model is queried again after execution; official goal geometry never decides a correction.
     for attempt in range(a.max_corrections+1):
      capture=call(sock,'capture_grounding');prediction=ground(capture['image'],mode) if mode!='oracle' else {'ok':True,'mode':'oracle'}
      (out/f'prediction_{attempt:02d}.json').write_text(json.dumps(prediction,indent=2))
      g=call(sock,'ground_target',mode=mode,**({'point_2d':prediction['point_2d']} if mode!='oracle' else {}))
      planout=out/'plans'/f'attempt{attempt:02d}';planout.mkdir(parents=True);shutil.copyfile(out/'goal.json',planout/'goal.json')
      client=launch([str(M/'scripts/ros-python'),str(D/'scripts/plan_execute.py'),'--socket',str(sock),'--out',str(planout)],f'plan_{attempt:02d}');client.wait(timeout=220)
      plan=json.loads((planout/'moveit_result.json').read_text());audit=call(sock,'evaluate_precontact')
      event={'planning_success':plan.get('success'),'audit':audit};row['attempts'].append(event)
      if not plan.get('success'):raise RuntimeError(plan.get('error','MoveIt failed'))
      if mode=='oracle':row['visual_confirmed']=None;break
      after=call(sock,'capture_grounding');check=ground(after['image'],mode);(out/f'visual_check_{attempt:02d}.json').write_text(json.dumps(check,indent=2))
      checkgoal=call(sock,'ground_target',mode=mode,point_2d=check['point_2d'])
      actual=np.array(after['state']['tcp_p']);desired=np.array(checkgoal['base_state']['base_p'])+np.array(checkgoal['base_state']['base_R'])@np.array(checkgoal['target_position'])
      residual=float(np.linalg.norm(actual-desired));event['visual_residual_m']=residual
      row['visual_confirmed']=residual<=.025
      if row['visual_confirmed']:break
     audit=call(sock,'evaluate_precontact');row['final_audit']=audit
     row['libero_original_task_success']=call(sock,'state')['success']
     row['success']=bool(row.get('visual_confirmed',True) is not False and audit['shadow_true_precontact_error_m']<=.03 and not audit['any_arm_environment_contact'] and not audit['any_self_contact'] and audit['physics_id_unchanged'])
     row['precontact_success']=row['success']
     if not row['precontact_success']:raise RuntimeError('Precontact failed; policy handover skipped')
     # Stop MoveIt and its bridge before handing sole actuator ownership back to OSC.
     for proc in [moveit,bridge]:stop(proc)
     resumed=call(sock,'resume_policy',timeout=240)
     if not resumed.get('ok',True):raise RuntimeError(resumed.get('error'))
     sys.path.insert(0,str(D.parent/'scripts'))
     from milestones import MilestoneMonitor
     from skills import SKILLS
     monitor=MilestoneMonitor(SKILLS['bowl_to_bottom_drawer_and_close']);checks=[]
     for step in range(1,521):
      state=call(sock,'policy_tick',timeout=120)
      if not state.get('ok',True):raise RuntimeError(state.get('error'))
      if step%10==0:
       cap=call(sock,'capture_grounding');payload=monitor.request(cap['image'])
       worker.stdin.write(json.dumps(payload)+'\n');worker.stdin.flush();response=read_json_line(worker,90)
       event=monitor.update(step,response);checks.append(event)
       (out/'milestone_checks.json').write_text(json.dumps(checks,indent=2))
       if monitor.done:break
     final=call(sock,'state');row['combined_official_success']=bool(final['success']);row['visual_milestones_complete']=monitor.done;row['policy_steps']=step
     row['success']=bool(row['combined_official_success'] and monitor.done)
     row['combined_stop_reason']='qwen_milestones' if monitor.done else 'step_limit'

    except Exception as e:row['error']=repr(e)
    finally:
     if sock.exists():
      try:call(sock,'close_episode')
      except Exception as exc:row['close_error']=repr(exc)
     for z in reversed(processes):stop(z)
     row['process_exit_codes']=[z.returncode for z in processes]
     row['cleanup_ok']=not any('Segmentation fault' in f.read_text(errors='replace') for f in logs.glob('*.log'))
     for f in streams:f.close()
     sock.unlink(missing_ok=True)
     if out.exists():(out/'case_result.json').write_text(json.dumps(row,indent=2))
     summary.append(row);(D/'results'/f'{a.prefix}_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(row),flush=True)
finally:
 if worker:close_json_worker(worker)
 if workerlog:workerlog.close()
