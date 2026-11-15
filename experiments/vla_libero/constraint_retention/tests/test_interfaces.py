from pathlib import Path
import sys,unittest
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S/'scripts'))
from design import OBJECTS,make_prompt,skill
from view_checker import input_images,parse,prompt


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.objects=OBJECTS[0]
        self.selection=dict(target='alphabet_soup_1',protected='tomato_sauce_1')
        self.row={'images':{k:{'path':k+'.png'} for k in ['reference','agent','wrist']},'goal_done':True}
    def test_positive_matches_previous_skill_exactly(self):
        self.assertEqual(make_prompt(self.objects,self.selection,'positive'),skill('alphabet soup can'))
    def test_suffix_preserves_prefix_and_global_constraint(self):
        short=make_prompt(self.objects,self.selection,'positive');long=make_prompt(self.objects,self.selection,'constrained')
        self.assertTrue(long.startswith(short));self.assertIn('Leave the tomato sauce can at its original position',long)
        self.assertIn('Do not move any other object',long)
    def test_wrong_model_selection_is_not_corrected(self):
        wrong=dict(target='tomato_sauce_1',protected='alphabet_soup_1')
        text=make_prompt(self.objects,wrong,'constrained')
        self.assertTrue(text.startswith(skill('tomato sauce can')));self.assertIn('Leave the alphabet soup can',text)
    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):make_prompt(self.objects,self.selection,'other')
    def test_only_third_image_changes_in_primary_pair(self):
        a=input_images(self.row,'duplicate_control');b=input_images(self.row,'wrist')
        self.assertEqual(a[:2],b[:2]);self.assertEqual(a[1],a[2]);self.assertNotEqual(a[2],b[2])
    def test_truth_label_is_not_part_of_model_input(self):
        a=input_images(self.row,'wrist');self.row['goal_done']=False
        self.assertEqual(a,input_images(self.row,'wrist'));self.assertNotIn('goal_done',prompt('test goal'))
    def test_unknown_is_explicit_non_confirmation(self):
        result=parse('```json\n{"status":"unknown","reason":"occluded"}\n```')
        self.assertFalse(result['done']);self.assertEqual(result['status'],'unknown')
    def test_invalid_status_does_not_become_completion(self):
        for raw in ['{"status":true}','{"status":"yes"}','[]']:
            with self.assertRaises(ValueError):parse(raw)


if __name__=='__main__':unittest.main()
