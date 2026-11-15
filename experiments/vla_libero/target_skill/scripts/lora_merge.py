"""Mirror the pinned openpi runtime contraction, including output-head reduction."""
import numpy as np

def merge_chunk(weight,a,b,slices,*,name):
    ai=list(slices);bi=list(slices);ai[-1]=slice(None);bi[-2]=slice(None)
    aa=a[tuple(ai)].astype(weight.dtype).astype(np.float32)
    if 'attn_vec_einsum' in name:
        # BTNH,NHL->BTL followed by BTL,NLD->BTD reduces N twice.
        # Effective per-head delta is A[n] @ sum_m(B[m]), not A[n] @ B[n].
        reduced=b.astype(weight.dtype).astype(np.float32).sum(axis=-3,keepdims=True)
        bi[-3]=slice(None)
        bb=reduced[tuple(bi)]
    else:
        bb=b[tuple(bi)].astype(weight.dtype).astype(np.float32)
    delta=np.matmul(aa,bb)
    assert delta.shape==weight.shape,(name,delta.shape,weight.shape)
    return (weight.astype(np.float32)+delta).astype(weight.dtype)
