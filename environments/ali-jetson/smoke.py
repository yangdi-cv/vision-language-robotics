"""Check isolated Python, image I/O, numerical dependencies and data output."""
import sys,platform,json,io
from pathlib import Path
import numpy as np
import scipy,cv2,yaml,requests,psutil,serial
from PIL import Image
assert Path(sys.prefix).name=='ali-jetson',sys.prefix
assert platform.machine()=='aarch64',platform.machine()
a=np.zeros((48,64,3),np.uint8);a[:,:,1]=127
ok,encoded=cv2.imencode('.png',a);assert ok
recovered=cv2.imdecode(encoded,cv2.IMREAD_COLOR);assert np.array_equal(a,recovered)
b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');assert Image.open(io.BytesIO(b.getvalue())).size==(64,48)
result=dict(python=sys.version,executable=sys.executable,prefix=sys.prefix,architecture=platform.machine(),numpy=np.__version__,opencv=cv2.__version__,scipy=scipy.__version__,serial=serial.__version__,available_memory_bytes=psutil.virtual_memory().available,image_roundtrip=True,gpu_stack_installed=False)
print(json.dumps(result,indent=2));Path('SMOKE_RESULT.json').write_text(json.dumps(result,indent=2))
