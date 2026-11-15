"""Reference real einsum contractions rather than mirroring a presumed merge formula."""
import unittest,sys,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lora_merge import merge_chunk
class Merge(unittest.TestCase):
 def test_output_reduces_all_heads(self):
  r=np.random.default_rng(6);x=r.normal(size=(1,2,3,4)).astype(np.float32);w=r.normal(size=(3,4,5)).astype(np.float32);a=r.normal(size=(3,4,2)).astype(np.float32);b=r.normal(size=(3,2,5)).astype(np.float32)
  actual=np.einsum('btnh,nhd->btd',x,w)+np.einsum('btl,nld->btd',np.einsum('btnh,nhl->btl',x,a),b)
  merged=merge_chunk(w,a,b,(slice(None),)*3,name='attn_vec_einsum/w')
  np.testing.assert_allclose(np.einsum('btnh,nhd->btd',x,merged),actual,rtol=1e-5,atol=2e-5)
  self.assertGreater(float(abs(np.einsum('btnh,nhd->btd',x,w+a@b)-actual).max()),.01)
 def test_chunked_head_and_layer_axes(self):
  r=np.random.default_rng(2);w=r.normal(size=(2,3,4,5)).astype(np.float32);a=r.normal(size=(2,3,4,2)).astype(np.float32);b=r.normal(size=(2,3,2,5)).astype(np.float32);expected=w+np.matmul(a,b.sum(axis=-3,keepdims=True));actual=np.empty_like(w)
  for starts in itertools.product(*[range(0,n,2) for n in w.shape]):
   idx=tuple(slice(i,min(i+2,n)) for i,n in zip(starts,w.shape));actual[idx]=merge_chunk(w[idx],a,b,idx,name='attn_vec_einsum_1/w')
  np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
 def test_query_keeps_head(self):
  r=np.random.default_rng(7);x=r.normal(size=(1,2,4)).astype(np.float32);w=r.normal(size=(3,4,5)).astype(np.float32);a=r.normal(size=(3,4,2)).astype(np.float32);b=r.normal(size=(3,2,5)).astype(np.float32)
  expected=np.einsum('btd,ndh->btnh',x,w)+np.einsum('btnl,nlh->btnh',np.einsum('btd,ndl->btnl',x,a),b)
  merged=merge_chunk(w,a,b,(slice(None),)*3,name='q_einsum/w');np.testing.assert_allclose(np.einsum('btd,ndh->btnh',x,merged),expected,rtol=1e-5,atol=1e-5)
 def test_zero_delta(self):
  w=np.ones((2,3,4),np.float32);a=np.ones((2,3,2),np.float32);b=np.zeros((2,2,4),np.float32);np.testing.assert_array_equal(merge_chunk(w,a,b,(slice(None),)*3,name='attn_vec_einsum/w'),w)
if __name__=='__main__':unittest.main()
