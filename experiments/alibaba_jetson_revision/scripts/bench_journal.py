"""Measure serialization cost separately from GPU inference."""
from pathlib import Path
import argparse,json,time,statistics
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=json.loads(a.input.read_text());result=[]
for n in [100,1000,2000,5000]:
 history=[rows[i%len(rows)] for i in range(n)]
 for mode in ['full_history','append_one_record']:
  times=[]
  for _ in range(10):
   t=time.perf_counter();text=json.dumps(history,indent=2) if mode=='full_history' else json.dumps(history[-1])+'\n';times.append((time.perf_counter()-t)*1000)
  result.append({'history_length':n,'method':mode,'median_serialization_ms':statistics.median(times),'serialized_bytes':len(text.encode()),'repeats':len(times),'scope':'JSON encoding CPU only; excludes file I/O and GPU'})
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
