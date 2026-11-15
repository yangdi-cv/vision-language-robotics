"""Two fixed development mistakes, rendered from actual model inputs and rollout frames."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

S=Path(__file__).resolve().parents[1]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def main():
    cases=[('task01_target0_init00',160,False,True,'漏确认：奶酪已入篮，模型仍回答未完成'),
           ('task00_target1_init00',180,True,False,'误确认：要求番茄酱，模型已确认但目标未入篮')]
    canvas=Image.new('RGB',(1440,1200),'#eef3f7');draw=ImageDraw.Draw(canvas)
    fonts={n:ImageFont.truetype(FONT,n) for n in [22,26]}
    for i,(key,step,prediction,truth,title) in enumerate(cases):
        root=S/f'episodes/pilot/{key}/visual';r=json.loads((root/'result.json').read_text())
        e=next(e for e in r['events'] if e['kind']=='check' and e['step']==step)
        with np.load(root/'trace.npz') as f:
            label=bool(f['target_in'][step-1] and f['target_free'][step-1])
        assert e['reply']['done']==prediction and label==truth
        reader=imageio_ffmpeg.read_frames(str(root/'rollout.mp4'));meta=next(reader)
        last=None
        for frame in reader:last=np.frombuffer(frame,np.uint8).reshape(meta['size'][1],meta['size'][0],3).copy()
        y=i*600
        draw.text((15,y+10),title,font=fonts[26],fill='#17354c')
        for col,(name,caption) in enumerate([(root/'images/reference.png','开始帧'),(root/f'images/current_{step:04d}.png',f'第{step}步：实际模型输入'),(None,'停止/预算后实际末帧')]):
            im=Image.fromarray(last) if name is None else Image.open(name).convert('RGB')
            canvas.paste(im.resize((450,450)),(15+480*col,y+80))
            draw.text((15+480*col,y+47),caption,font=fonts[22],fill='#17354c')
        text=f"模型完成票：{'是' if prediction else '否'}；离线目标入篮且不接触夹爪：{'是' if label else '否'}；均来自开发初态0。"
        draw.text((15,y+545),text,font=fonts[22],fill='#17354c')
    canvas.save(S/'figures/development_errors.png')


if __name__=='__main__':main()
