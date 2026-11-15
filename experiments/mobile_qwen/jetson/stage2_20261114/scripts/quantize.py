from pathlib import Path
import argparse
import hashlib
import json
from runtime import Runtime
import faulthandler
import os
if os.environ.get("ALI_QUANT_DIAGNOSTIC"):
    faulthandler.dump_traceback_later(3,repeat=True)
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--revision');p.add_argument('--out',type=Path,required=True)
a=p.parse_args();assert not a.out.exists()
r=Runtime(a.model,a.revision,precision='nf4')
faulthandler.cancel_dump_traceback_later()
a.out.mkdir(parents=True)
r.model.save_pretrained(a.out,max_shard_size='1GB');r.processor.save_pretrained(a.out)
(a.out/'EXPORT.json').write_text(json.dumps({'model':a.model,'revision':a.revision,'precision':'NF4 double quant; visual and tied output head kept BF16','resources':r.report(),'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.iterdir() if f.is_file()}},indent=2))
