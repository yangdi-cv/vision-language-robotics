"""Actual Nav2 action goals in Gazebo; fixed station geometry, no robot hardware."""
import argparse,json,time,csv,math
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[3]
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile,DurabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped
from nav_msgs.msg import Odometry
from lifecycle_msgs.srv import GetState
from rclpy.parameter import Parameter
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--act-model',type=Path);a=p.parse_args()
rclpy.init();n=Node('alibaba_navigation_test',parameter_overrides=[Parameter('use_sim_time',value=True)]);action=ActionClient(n,NavigateToPose,'navigate_to_pose');last={};trace=[]
def odom(m):
 last['x']=m.pose.pose.position.x;last['y']=m.pose.pose.position.y;trace.append({'host_monotonic_ns':time.monotonic_ns(),'sim_time_s':m.header.stamp.sec+m.header.stamp.nanosec*1e-9,'odom_x':last['x'],'odom_y':last['y']})
sub=n.create_subscription(Odometry,'odom',odom,100)
qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL);initial=n.create_publisher(PoseWithCovarianceStamped,'initialpose',qos)
def wait(future,timeout):
 deadline=time.monotonic()+timeout
 while not future.done():
  if time.monotonic()>deadline:raise TimeoutError('ROS future deadline')
  rclpy.spin_once(n,timeout_sec=.1)
 return future.result()
def pose(x,y):
 m=PoseStamped();m.header.frame_id='map';m.header.stamp=n.get_clock().now().to_msg();m.pose.position.x=x;m.pose.position.y=y;m.pose.orientation.w=1.;return m
results=[]
try:
 for node_name in ('amcl',):
  client=n.create_client(GetState,node_name+'/get_state');deadline=time.monotonic()+100
  while time.monotonic()<deadline:
   if client.wait_for_service(timeout_sec=1):
    state=wait(client.call_async(GetState.Request()),5)
    if state.current_state.id==3:break
   rclpy.spin_once(n,timeout_sec=.2)
  else:raise TimeoutError(node_name+' lifecycle not active')
 m=PoseWithCovarianceStamped();m.header.frame_id='map';m.pose.pose.position.x=-2.;m.pose.pose.position.y=-.5;m.pose.pose.orientation.w=1.;m.pose.covariance[0]=.05;m.pose.covariance[7]=.05;m.pose.covariance[35]=.05
 for _ in range(20):m.header.stamp=n.get_clock().now().to_msg();initial.publish(m);rclpy.spin_once(n,timeout_sec=.1)
 client=n.create_client(GetState,'bt_navigator/get_state');deadline=time.monotonic()+60
 while time.monotonic()<deadline:
  if client.wait_for_service(timeout_sec=1) and wait(client.call_async(GetState.Request()),5).current_state.id==3:break
  rclpy.spin_once(n,timeout_sec=.2)
 else:raise TimeoutError('bt_navigator not active after initialpose')
 if not action.wait_for_server(timeout_sec=15):raise TimeoutError('Nav2 action server missing after activation')
 for station,x,y in [('grasp_station',-1.2,-.5),('return_station',-2.,-.5)]:
  g=NavigateToPose.Goal();g.pose=pose(x,y);started=time.monotonic();handle=wait(action.send_goal_async(g),15)
  if not handle.accepted:raise RuntimeError('Navigation goal rejected')
  result=wait(handle.get_result_async(),120)
  row={'station':station,'goal_map_xy':[x,y],'action_status':result.status,'succeeded':result.status==4,'wall_seconds':time.monotonic()-started,'final_odom_xy':[last.get('x'),last.get('y')]};results.append(row);print(row,flush=True)
  (a.out/'navigation_results.json').write_text(json.dumps(results,indent=2))
  if not row['succeeded']:break
  if station=='grasp_station' and a.act_model:
   cmd=[str(ROOT/'experiments/alibaba_baseline/scripts/act-python'),str(ROOT/'experiments/alibaba_baseline/scripts/evaluate_act.py'),'--model',str(a.act_model),'--out',str(a.out/'act_stage'),'--episodes','1','--seed-start','1003']
   with (a.out/'act_stage.log').open('w') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=90)
   if proc.returncode:raise RuntimeError('ACT stage process failed')
   act=json.loads((a.out/'act_stage/metrics.json').read_text());(a.out/'handoff.json').write_text(json.dumps({'navigation_reached':True,'act_successes':act['successes'],'act_episode_seed':1003,'act_model':str(a.act_model),'shared_physical_scene':False,'note':'Workflow-level handoff between independent Gazebo base and MuJoCo arms, known-success smoke-test seed'},indent=2))
   if act['successes']!=1:break
finally:
 (a.out/'odom_trace.json').write_text(json.dumps(trace));n.destroy_node();rclpy.shutdown()
