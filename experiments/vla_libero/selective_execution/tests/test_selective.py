from pathlib import Path
import sys
import unittest
import numpy as np

S=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(S/'scripts'))
from protocol import score, case, calibrate
from control import StopRule
from worker import select


class SelectiveTests(unittest.TestCase):
    def make(self, displacement=None, target=None):
        return dict(target_in=np.ones(40,bool) if target is None else np.asarray(target,bool),
                    target_free=np.ones(40,bool), protected_displacement=np.zeros(40) if displacement is None else displacement)
    def test_protected_excursion_cannot_be_undone(self):
        displacement=np.zeros(40);displacement[2]=.04
        r=score(self.make(displacement),15,'visual')
        self.assertFalse(r['selective_success']);self.assertEqual(r['first_violation_step'],3)
    def test_stable_placement_not_one_frame_contact(self):
        target=np.ones(40,bool);target[-3]=False
        self.assertFalse(score(self.make(target=target),15,'visual')['target_final_stable'])
    def test_ten_valid_frames_required_for_skill_calibration(self):
        r=score(self.make(),15,'budget')
        self.assertEqual(r['first_valid_step'],10)
        self.assertTrue(r['selective_success']);self.assertFalse(r['self_stop_success'])
    def test_unreleased_object_does_not_pass(self):
        t=self.make();t['target_free'][-1]=False
        self.assertFalse(score(t,15,'visual')['selective_success'])
    def test_closed_empty_gripper_is_not_a_task_failure(self):
        t=self.make();t['released']=np.zeros(40,bool)
        self.assertTrue(score(t,15,'budget')['selective_success'])
    def test_two_votes_with_common_gate(self):
        r=StopRule('visual');grip=[.04,-.04];action=np.array([0,0,0,0,0,0,-1])
        self.assertTrue(r.eligible(20,grip,action))
        self.assertFalse(r.decide(20,dict(ok=True,done=True))['stop'])
        self.assertTrue(r.decide(30,dict(ok=True,done=True))['stop'])
    def test_closed_gripper_invalidates_old_vote(self):
        r=StopRule('visual');r.decide(20,dict(ok=True,done=True))
        r.eligible(25,[.01,-.01],np.array([0,0,0,0,0,0,1]))
        self.assertFalse(r.decide(30,dict(ok=True,done=True))['stop'])
    def test_vote_expiration(self):
        r=StopRule('visual');r.decide(20,dict(ok=True,done=True))
        self.assertFalse(r.decide(50,dict(ok=True,done=True))['stop'])
    def test_timer_does_not_use_model_or_oracle(self):
        r=StopRule('timer150')
        self.assertFalse(r.decide(140)['done']);self.assertFalse(r.decide(150)['stop']);self.assertTrue(r.decide(160)['stop'])
    def test_invalid_model_response_resets_streak(self):
        r=StopRule('visual');r.decide(20,dict(ok=True,done=True))
        self.assertFalse(r.decide(30,dict(ok=False,done=True))['done'])
        self.assertFalse(r.decide(40,dict(ok=True,done=True))['stop'])
    def test_calibration_and_reverse_instruction(self):
        self.assertEqual(calibrate([dict(case={'target':'a'},first_valid_step=113),dict(case={'target':'b'},first_valid_step=None)]),{'a':120,'b':150})
        a,b=case(0,0,0),case(0,1,0)
        self.assertEqual(a['target'],b['protected']);self.assertNotEqual(a['instruction'],b['instruction'])
    def test_selector_does_not_silently_correct_wrong_target(self):
        class Fake:
            def generate(self,*args,**kwargs):return dict(raw='{"target":"tomato_sauce_1","protected":"alphabet_soup_1"}',ok=False)
        c=case(0,0,0);r=select(Fake(),'unused',c['instruction'],c['objects'])
        self.assertTrue(r['ok']);self.assertNotEqual(r['target'],c['target'])


if __name__=='__main__':unittest.main()
