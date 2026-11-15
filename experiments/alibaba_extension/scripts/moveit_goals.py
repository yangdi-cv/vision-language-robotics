"""Actual MoveGroup planning/execution against mock Panda controllers; not physical grasp."""
import argparse,json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints,JointConstraint
from sensor_msgs.msg import JointState
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();rclpy.init();n=Node('alibaba_moveit_test');client=ActionClient(n,MoveGroup,'/move_action');last={};trace=[]
def state(m):
 last.update(zip(m.name,m.position));trace.append({'host_monotonic_ns':time.monotonic_ns(),'names':list(m.name),'positions':list(m.position)})
sub=n.create_subscription(JointState,'joint_states',state,100)
def wait(future,seconds):
 deadline=time.monotonic()+seconds
 while not future.done():
  if time.monotonic()>deadline:raise TimeoutError('MoveIt result deadline')
  rclpy.spin_once(n,timeout_sec=.1)
 return future.result()
rows=[]
try:
 if not client.wait_for_server(timeout_sec=60):raise TimeoutError('MoveGroup missing')
 time.sleep(3)
 poses=[[.2,-.65,-.15,-2.2,.1,1.6,.8],[-.2,-.8,.15,-2.3,-.1,1.5,.7],[0.,-.785398,0.,-2.356194,0.,1.570796,.785398],[.1,4.,0.,-2.3,0.,1.5,.7]]
 for index,target in enumerate(poses):
  goal=MoveGroup.Goal();goal.request.group_name='panda_arm';goal.request.pipeline_id='ompl';goal.request.planner_id='RRTConnectkConfigDefault';goal.request.num_planning_attempts=1;goal.request.allowed_planning_time=5.;goal.request.max_velocity_scaling_factor=.2;goal.request.max_acceleration_scaling_factor=.2;goal.request.start_state.is_diff=True
  constraints=Constraints()
  for j,value in enumerate(target):
   c=JointConstraint();c.joint_name=f'panda_joint{j+1}';c.position=value;c.tolerance_above=.01;c.tolerance_below=.01;c.weight=1.;constraints.joint_constraints.append(c)
  goal.request.goal_constraints=[constraints];goal.planning_options.plan_only=index>=3;goal.planning_options.planning_scene_diff.is_diff=True
  tick=time.monotonic();handle=wait(client.send_goal_async(goal),10)
  if not handle.accepted:raise RuntimeError('MoveIt rejected action')
  response=wait(handle.get_result_async(),40);result=response.result
  row={'index':index,'target_rad':target,'plan_only':goal.planning_options.plan_only,'action_status':response.status,'moveit_error_code':result.error_code.val,'succeeded':result.error_code.val==1,'planned_joint_names':list(result.planned_trajectory.joint_trajectory.joint_names),'planned_positions':[list(p.positions) for p in result.planned_trajectory.joint_trajectory.points],'planned_points':len(result.planned_trajectory.joint_trajectory.points),'executed_points':len(result.executed_trajectory.joint_trajectory.points),'wall_seconds':time.monotonic()-tick,'latest_joint_feedback':dict(last),'expected_success':index<3};rows.append(row);(a.out/f'case_{index}.json').write_text(json.dumps(row,indent=2));print(row,flush=True)
finally:
 (a.out/'joint_feedback_trace.json').write_text(json.dumps(trace));(a.out/'results.json').write_text(json.dumps(rows,indent=2));n.destroy_node();rclpy.shutdown()
