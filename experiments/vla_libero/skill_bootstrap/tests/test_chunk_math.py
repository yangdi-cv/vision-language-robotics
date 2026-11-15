"""Check chunked BF16 LoRA math against full-tensor multiplication."""
import sys,itertools,unittest
from pathlib import Path
import numpy as np
import ml_dtypes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from chunk_math import merge_chunk

class MergeMath(unittest.TestCase):
    def check_shape(self,shape):
        r=np.random.default_rng(3);a=r.normal(size=(*shape[:-1],2)).astype(np.float32);b=r.normal(size=(*shape[:-2],2,shape[-1])).astype(np.float32);w=r.normal(size=shape).astype(ml_dtypes.bfloat16)
        expected=(w.astype(np.float32)+np.matmul(a.astype(ml_dtypes.bfloat16).astype(np.float32),b.astype(ml_dtypes.bfloat16).astype(np.float32))).astype(ml_dtypes.bfloat16)
        actual=np.empty_like(w);coverage=np.zeros(shape,int)
        for start in itertools.product(*[range(0,n,2) for n in shape]):
            idx=tuple(slice(i,min(i+2,n)) for i,n in zip(start,shape));actual[idx]=merge_chunk(w[idx],a,b,idx);coverage[idx]+=1
        np.testing.assert_array_equal(coverage,1);np.testing.assert_array_equal(expected,actual)
    def test_attention(self):self.check_shape((3,2,5,7))
    def test_key_value(self):self.check_shape((2,2,1,5,7))
    def test_feed_forward(self):self.check_shape((3,5,7))
    def test_zero_increment_preserves_bits(self):
        r=np.random.default_rng(19);w=r.normal(size=(2,3,4)).astype(ml_dtypes.bfloat16);a=r.normal(size=(2,3,2)).astype(np.float32);b=np.zeros((2,2,4),np.float32)
        result=merge_chunk(w,a,b,(slice(None),)*3);self.assertEqual(w.tobytes(),result.tobytes())

if __name__=='__main__':unittest.main()
