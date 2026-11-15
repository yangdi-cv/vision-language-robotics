"""A fixed initial reference image with a red target box; augment image and box together."""
import numpy as np
from PIL import Image,ImageDraw
GENERIC='pick up the object marked by the red box and place it in the basket'
def mark(image,box,rng=None):
 image=np.asarray(image,dtype=np.uint8);box=np.asarray(box,int).copy();assert image.shape==(224,224,3)
 im=Image.fromarray(image.copy())
 if rng is not None:
  dx=int(rng.integers(max(-12,-box[0]),min(12,223-box[2])+1));dy=int(rng.integers(max(-12,-box[1]),min(12,223-box[3])+1))
  canvas=Image.new('RGB',(224,224),(96,96,96));canvas.paste(im,(dx,dy));im=canvas;box+=np.array([dx,dy,dx,dy])
 ImageDraw.Draw(im).rectangle(tuple(box.tolist()),outline=(255,0,0),width=3)
 return np.asarray(im).copy()
