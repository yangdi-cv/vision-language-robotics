"""Run packaged interface checks in isolated CPU-only processes."""
from pathlib import Path
import argparse,json,os,subprocess,sys
R=Path(__file__).resolve().parents[1]
GROUPS=[
('libero_interfaces','experiments/vla_libero/scripts','tests.py'),
('recovery','experiments/vla_libero/recovery_study/scripts','test_*.py'),
('iteration_inputs','experiments/vla_libero/closed_loop_iteration/scripts','test_*.py'),
('checker','experiments/vla_libero/checker_v2/scripts','test_*.py'),
('delivery','experiments/vla_libero/tests','test_*.py'),
('runtime_logging','experiments/mobile_qwen/tests','test_*.py'),
('selective_execution','experiments/vla_libero/selective_execution/tests','test_*.py'),
('visual_interventions','experiments/vla_libero/monitor_causality/tests','test_*.py'),
('constraint_interfaces','experiments/vla_libero/constraint_retention/tests','test_*.py'),
('action_guidance','experiments/vla_libero/cag_reproduction/tests','test_*.py'),
('chunk_math','experiments/vla_libero/skill_bootstrap/tests','test_*.py'),
('visual_target','experiments/vla_libero/point_target/tests','test_*.py')]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=R/'repository/offline_checks.json');args=ap.parse_args()
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONPATH':str(R/'experiments/vla_libero/scripts')+os.pathsep+str(R/'experiments/vla_libero/third_party/openpi/packages/openpi-client/src')}
 out=[]
 for name,folder,pattern in GROUPS:
  p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(R/folder),'-p',pattern],cwd=R,env=env,capture_output=True,text=True,timeout=180)
  out.append(dict(group=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));print(name,'PASS' if p.returncode==0 else 'FAIL',flush=True)
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2))
 return int(any(x['returncode'] for x in out))
if __name__=='__main__':raise SystemExit(main())
