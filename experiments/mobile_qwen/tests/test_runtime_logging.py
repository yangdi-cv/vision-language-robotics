"""CPU-only failure-path checks; no model, ROS or simulator initialization."""
import ast
import io
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

S = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(S))
from durable_log import Journal, atomic_json
from sim_server_qwen import QwenWorker

class RuntimeLoggingTest(unittest.TestCase):
    def test_journal_survives_abrupt_process_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = "from durable_log import Journal; import os; Journal(%r).append('qwen_request', payload={'prompt':'测试'}); os._exit(9)" % tmp
            proc = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'PYTHONPATH': str(S)})
            self.assertEqual(proc.returncode, 9)
            self.assertEqual(json.loads((Path(tmp)/'events.jsonl').read_text())['kind'], 'qwen_request')

    def test_atomic_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'progress.json'
            atomic_json(p, {'steps':1}); atomic_json(p, {'steps':2})
            self.assertEqual(json.loads(p.read_text()), {'steps':2})
            self.assertFalse(p.with_name(p.name+'.tmp').exists())

    def test_worker_records_full_metadata_and_pending_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = QwenWorker.__new__(QwenWorker)
            w.journal = Journal(tmp); w.sequence = 0
            reply = {'ok': True, 'inference': {'prompt':'full prompt', 'revision':'frozen', 'images':[{'pixel_sha256':'abc'}]}}
            w.proc = types.SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(json.dumps(reply)+'\n'))
            w.request({'op':'milestone','image':'x.png'})
            with self.assertRaises(RuntimeError): w.request({'op':'check'})
            records = [json.loads(x) for x in w.journal.path.read_text().splitlines()]
            self.assertEqual([x['kind'] for x in records], ['qwen_request','qwen_response','qwen_request'])
            self.assertEqual(records[1]['payload']['inference'],reply['inference'])

    def test_launch_propagates_failure_without_systemd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'scripts').mkdir(); (root/'bin').mkdir()
            p=root/'scripts/launch.sh';p.write_text((S/'launch.sh').read_text()); p.chmod(0o755)
            fake=root/'bin/systemd-run';fake.write_text('#!/bin/sh\nexit 37\n');fake.chmod(0o755)
            result=subprocess.run([str(p),'test','0','1','request'], env={**os.environ,'PATH':str(root/'bin')+':'+os.environ['PATH']},capture_output=True)
            self.assertEqual(result.returncode,37)

    def test_orchestrator_reports_failure(self):
        import run_case_qwen as runner
        proc=types.SimpleNamespace(poll=lambda: 1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,'Q',Path(tmp)), patch.object(runner.subprocess,'Popen',return_value=proc), patch.object(runner,'stop'), patch.object(sys,'argv',['test','--initial','0','--tag','failuretest','--domain','1','--request','test']):
            self.assertEqual(runner.main(),1)

    def test_control_step_snapshot_without_simulator(self):
        import collections
        import numpy as np
        tree=ast.parse((S/'sim_server_qwen.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='manipulation_step')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'evaluate':types.ModuleType('evaluate')}):
            j=Journal(tmp); scope={'np':np,'journal':j,'atomic_json':atomic_json}
            exec(compile(ast.Module(body=[method],type_ignores=[]),'<step-test>','exec'),scope)
            scene=types.SimpleNamespace(obs={'robot0_joint_pos':np.zeros(7)}, step=lambda action:None,e=types.SimpleNamespace(check_success=lambda:False))
            engine=types.SimpleNamespace(s=scene,queue=collections.deque([np.zeros(7)]),policy_steps=0,applied=[],first_official_success_step=None,a=types.SimpleNamespace(min_steps=100,max_steps=520,monitor='milestone'),stop_reason=None,phase='manipulation',out=Path(tmp),events=[])
            scope['manipulation_step'](engine)
            self.assertEqual(json.loads((Path(tmp)/'progress.json').read_text())['policy_steps'],1)
            self.assertEqual(json.loads(j.path.read_text())['kind'],'control_step')

if __name__ == '__main__': unittest.main()
