"""Small out-of-editor observer for the isolated evaluation service."""
import json,subprocess,time,sys
from pathlib import Path
S=Path(__file__).resolve().parents[1];unit=sys.argv[1] if len(sys.argv)>1 else 'ali-vla-eval-safe.service';out=S/'resource_checks'/('memory_'+unit+'.jsonl')
for _ in range(1800):
    r=subprocess.run(['systemctl','--user','show',unit,'-p','MemoryCurrent','-p','MemoryPeak','-p','MainPID','-p','ActiveState','-p','Result'],text=True,capture_output=True)
    row=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line);row['epoch']=time.time();row['unit']=unit
    pid=row.get('MainPID','0')
    try:
        fields=Path('/proc/'+pid+'/status').read_text().splitlines();row['rss_kib']=int(next(x for x in fields if x.startswith('VmRSS:')).split()[1])
    except (OSError,StopIteration):pass
    with out.open('a') as f:f.write(json.dumps(row)+'\n')
    if row.get('ActiveState') not in ['active','activating']:break
    time.sleep(2)
