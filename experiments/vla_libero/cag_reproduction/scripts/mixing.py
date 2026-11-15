"""Local implementation of CAG's published final-action mixing equation."""
import numpy as np

def combine(conditional,unconditional,scale):
    c=np.asarray(conditional);u=np.asarray(unconditional)
    if c.shape!=u.shape or c.ndim!=2 or c.shape[-1]!=7:raise ValueError('Expected matching (horizon,7) actions')
    if not np.isfinite(scale) or scale<0:raise ValueError('Invalid guidance scale')
    if not (np.isfinite(c).all() and np.isfinite(u).all()):raise ValueError('Nonfinite branch')
    if scale==1:return c.copy()
    if scale==0:return u.copy()
    return c+(scale-1)*(c-u)
