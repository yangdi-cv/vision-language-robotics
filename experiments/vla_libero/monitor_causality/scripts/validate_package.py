"""Unpack delivery, verify every archived byte, then run its self-contained CPU tests."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

S = Path(__file__).resolve().parents[1]
ROOT = S.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "deliverables/20261113_阿里_视觉检查闭环作用"
    archive = destination / "视觉检查干预_源码.zip"
    with tempfile.TemporaryDirectory(prefix="monitor_delivery_") as folder:
        extracted = Path(folder)
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            for name in bundle.namelist():
                path = Path(name)
                assert not path.is_absolute() and ".." not in path.parts, name
            bundle.extractall(extracted)
        hashes = json.loads((extracted / "PACKAGE_SHA256.json").read_text())
        for name, expected in hashes.items():
            assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == expected, name
        study = extracted / "experiments/vla_libero/monitor_causality"
        protocol = json.loads((study / "PROTOCOL.json").read_text())
        for name, expected in protocol["sources"].items():
            assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == expected, name
        cases = json.loads((study / "decision_data/cases.json").read_text())
        assert isinstance(cases, list)
        tests = subprocess.run(
            [str(Path(args.python).absolute()), "-m", "unittest", "discover", "-s",
             "experiments/vla_libero/monitor_causality/tests", "-p", "test_*.py"],
            cwd=extracted, capture_output=True, text=True,
        )
        if tests.returncode:
            raise RuntimeError(tests.stdout + tests.stderr)
        result = dict(
            clean_extracted_tests=True,
            manifest_files_verified=len(hashes),
            protocol_sources_verified=len(protocol["sources"]),
            decision_index_present=True,
            cpu_python=str(Path(args.python).absolute()),
            test_output=tests.stdout + tests.stderr,
            scope="Source ZIP and its self-contained CPU tests; full GPU simulation requires separately documented assets and weights",
        )
    if args.record:
        (S / "CLEAN_ACCEPTANCE.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
