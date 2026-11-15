"""Failure-injection checks for the supervisor contract; these are software cases, not trials."""
from pathlib import Path
import json
from task_contract import authorize,next_stage
root=Path(__file__).resolve().parents[1];p={'target_visible':True,'selected_skill':'transfer_red_cube'};text='抓起红色方块，并把它交给另一只机械臂。'
cases=[('supported',text,p,{},True),('blue_target','抓起蓝色方块，并把它交给另一只机械臂。',p,{},False),('negated','不要'+text,p,{},False),('extra_task',text+'然后抓蓝色球。',p,{},False),('grasp_only','抓起红色方块。',p,{},False),('not_visible',text,{**p,'target_visible':False},{},False),('malformed',text,None,{},False),('wrong_skill',text,{**p,'selected_skill':'observe'},{},False),('stale',text,p,{'age_s':6},False),('cancelled',text,p,{'cancelled':True},False),('nonfinite_age',text,p,{'age_s':float('nan')},False)]
rows=[]
for name,instruction,proposal,options,expected in cases:
 result=authorize(instruction,proposal,**options);assert result['execute']==expected;rows.append({'case':name,'expected_execute':expected,'actual':result})
for stage in ('authorized','navigate','manipulate','return'):assert next_stage(stage,False)=='stopped'
(root/'CONTRACT_VERIFICATION.json').write_text(json.dumps({'cases':rows,'checks_passed':len(rows)+4,'notes':'Software failure injection, not empirical robot robustness success rate'},ensure_ascii=False,indent=2));print('Contract checks passed',len(rows)+4)
