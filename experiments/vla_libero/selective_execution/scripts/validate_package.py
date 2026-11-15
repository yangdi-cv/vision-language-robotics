"""Validate final source ZIP in an isolated extraction and CPU interpreter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

S=Path(__file__).resolve().parents[1];ROOT=S.parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--python',default=sys.executable);ap.add_argument('--record',action='store_true');a=ap.parse_args()
    archive=ROOT/'deliverables/20261113_阿里_选择性搬运与视觉停止/选择性搬运_源码.zip'
    with tempfile.TemporaryDirectory(prefix='selective_delivery_') as folder:
        root=Path(folder)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
            launcher=z.getinfo('experiments/vla_libero/scripts/python_shared')
            assert (launcher.external_attr>>16)&0o111
            z.extractall(root)
        manifest=json.loads((root/'PACKAGE_SHA256.json').read_text())
        for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        study=root/'experiments/vla_libero/selective_execution'
        protocol=json.loads((study/'PROTOCOL.json').read_text())
        for name,digest in protocol['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1'
        tests=subprocess.run([str(Path(a.python).absolute()),'-m','unittest','discover','-s','experiments/vla_libero/selective_execution/tests','-p','test_*.py'],cwd=root,env=env,capture_output=True,text=True)
        assert tests.returncode==0,tests.stdout+tests.stderr
        syntax=subprocess.run(['bash','-n',str(root/'experiments/vla_libero/scripts/python_shared')],capture_output=True,text=True)
        assert syntax.returncode==0,syntax.stderr
        result=dict(clean_extracted_cpu_tests=True,manifest_files=len(manifest),frozen_sources=len(protocol['sources']),
            cpu_python=str(Path(a.python).absolute()),test_output=tests.stdout+tests.stderr,
            launcher_permissions_and_bash_syntax=True,scope='CPU self-contained checks, not full GPU assets or model execution')
    if a.record:(S/'CLEAN_ACCEPTANCE.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
