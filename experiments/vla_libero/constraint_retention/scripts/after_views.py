"""Serialize GPU stages; the live pilot is not gated by offline classification scores."""
import argparse,json,os,subprocess,time
from pathlib import Path

S=Path(__file__).resolve().parents[1];V=S.parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);a=ap.parse_args()
    while True:
        try:os.kill(a.pid,0)
        except ProcessLookupError:break
        time.sleep(5)
    status=json.loads((S/'view_audit/STATUS.json').read_text());assert status['state']=='complete'
    subprocess.run(['bash',str(V/'scripts/python_shared'),str(S/'scripts/live.py')],check=True)


if __name__=='__main__':main()
