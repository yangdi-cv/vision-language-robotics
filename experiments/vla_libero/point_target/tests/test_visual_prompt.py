import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from visual_prompt import mark
class Prompt(unittest.TestCase):
 def test_no_mutation(self):
  a=np.zeros((224,224,3),np.uint8);b=mark(a,[10,20,30,40]);self.assertFalse(a.any());self.assertEqual(b[20,10].tolist(),[255,0,0]);self.assertEqual(b[25,15].tolist(),[0,0,0])
 def test_target_changes_reference(self):
  a=np.zeros((224,224,3),np.uint8);self.assertFalse(np.array_equal(mark(a,[10,20,30,40]),mark(a,[100,100,120,120])))
 def test_translation_keeps_marker_in_bounds(self):
  for box in [[0,0,20,20],[200,200,223,223],[100,100,120,120]]:
   b=mark(np.zeros((224,224,3),np.uint8),box,np.random.default_rng(1));red=(b[:,:,0]==255)&(b[:,:,1]==0);self.assertGreater(red.sum(),50)
if __name__=='__main__':unittest.main()
