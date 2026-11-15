"""Render the same predicted depth using fixed or original adaptive contrast."""
import cv2
import numpy as np


def render(depth, objects, mode='adaptive'):
    valid = np.isfinite(depth) & (depth > 0)
    if mode == 'adaptive' and valid.any():
        inverse = np.zeros_like(depth)
        inverse[valid] = 1/depth[valid]
        low, high = np.percentile(inverse[valid], [2,98])
        high = max(high, low+1e-6)
        normalized = np.clip((inverse-low)/(high-low),0,1)
        ticks = [float(1/max(high-f*(high-low),1e-6)) for f in (0,.25,.5,.75,1)]
        label = '自适应对比 · 每帧拉伸，颜色不跨帧固定'
    else:
        normalized = 1-np.clip(depth/3,0,1)
        ticks = [0,.75,1.5,2.25,3]
        label = '固定量程 · 0–3 m'
    normalized[~valid]=0
    color=cv2.applyColorMap((normalized*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    color[~valid]=0
    for obj in objects:
        x1,y1,x2,y2=np.rint(obj['xyxy']).astype(int)
        cv2.rectangle(color,(x1,y1),(x2,y2),(255,255,255),2)
        value=obj['box_region_depth_estimate_m']
        text=f'Depth {value:.2f} m' if value is not None else 'Depth unknown'
        y=max(y1-7,20)
        cv2.putText(color,text,(x1,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(color,text,(x1,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    gradient=np.linspace(255,0,320).astype(np.uint8)[None,:]
    legend=cv2.applyColorMap(gradient,cv2.COLORMAP_TURBO)
    return color, {'ticks_m':ticks,'label':label,'mode':mode}, legend
