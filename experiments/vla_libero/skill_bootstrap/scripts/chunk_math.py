"""Shape-preserving LoRA merge on a bounded native checkpoint chunk."""
import numpy as np

def merge_chunk(weight, a, b, slices):
    ai=list(slices);bi=list(slices);ai[-1]=slice(None);bi[-2]=slice(None)
    aa=a[tuple(ai)].astype(weight.dtype).astype(np.float32)
    bb=b[tuple(bi)].astype(weight.dtype).astype(np.float32)
    delta=np.matmul(aa,bb)
    assert delta.shape==weight.shape,(delta.shape,weight.shape)
    return (weight.astype(np.float32)+delta).astype(weight.dtype)
