"""Finite task authorization independent of VLM proposal; no hardware interface."""
import re,math
FULL_TRANSFER=re.compile(r'(?:请)?抓起红色方块[，,](?:并把它交给|并交给)另一只机械臂[。.!！]?')
def authorize(instruction,proposal,*,age_s=0,cancelled=False):
 if cancelled:return {'execute':False,'reason':'cancelled'}
 if not isinstance(age_s,(int,float)) or not math.isfinite(age_s) or not 0<=age_s<=5:return {'execute':False,'reason':'stale_or_invalid_observation'}
 if not FULL_TRANSFER.fullmatch(instruction.strip()):return {'execute':False,'reason':'unsupported_full_task'}
 if not isinstance(proposal,dict) or proposal.get('target_visible') is not True or proposal.get('selected_skill')!='transfer_red_cube':return {'execute':False,'reason':'invalid_or_nonexecutable_proposal'}
 return {'execute':True,'reason':'supported_explicit_task_and_visible_target','station':'grasp_station','skill':'transfer_red_cube'}
def next_stage(stage,success):
 if not success:return 'stopped'
 return {'authorized':'navigate','navigate':'manipulate','manipulate':'return','return':'complete'}.get(stage,'stopped')
