import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mixing import combine
class Mixing(unittest.TestCase):
 def setUp(self):
  self.c=np.random.default_rng(3).normal(size=(10,7)).astype(np.float32);self.u=-self.c*.3
 def test_scale_one_bit_exact(self):self.assertEqual(combine(self.c,self.u,1).tobytes(),self.c.tobytes())
 def test_scale_zero(self):np.testing.assert_array_equal(combine(self.c,self.u,0),self.u)
 def test_equation(self):np.testing.assert_allclose(combine(self.c,self.u,1.5),self.u+1.5*(self.c-self.u),rtol=1e-6,atol=1e-6)
 def test_not_magnitude_scaling(self):np.testing.assert_array_equal(combine(self.c,self.c,1.5),self.c)
 def test_invalid(self):
  for w in [-1,np.nan,np.inf]:
   with self.assertRaises(ValueError):combine(self.c,self.u,w)
if __name__=='__main__':unittest.main()
