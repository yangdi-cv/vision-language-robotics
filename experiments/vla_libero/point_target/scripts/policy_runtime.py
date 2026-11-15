"""Merged native pi0.5 with a trained extra visual goal slot; one model in memory."""
import sys
from pathlib import Path
S=Path(__file__).resolve().parents[1];V=S.parent
sys.path[:0]=[str(V/'skill_bootstrap/scripts')]
def load(variant):
 from runtime import load_policy
 from goal_inputs import transforms_for
 from openpi.training import config
 from openpi.policies.policy import Policy
 native=load_policy(None if variant=='base' else S/'training/adapter_best.npz')
 transform,outputs=transforms_for(config.get_config('pi05_libero'))
 return Policy(native._model,transforms=[transform],output_transforms=outputs,sample_kwargs={'num_steps':10})
