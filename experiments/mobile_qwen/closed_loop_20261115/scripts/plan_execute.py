"""Load exact scene collision geometry, request MoveIt planning and real execution."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import ApplyPlanningScene,GetStateValidity,GetPositionFK
from moveit_msgs.msg import PlanningScene,CollisionObject,Constraints,JointConstraint,PositionConstraint,OrientationConstraint
from shape_msgs.msg import SolidPrimitive,Mesh,MeshTriangle
from geometry_msgs.msg import Pose,Point
D=Path(__file__).resolve().parents[1];ROOT=D.parents[2]
sys.path.insert(0,str(ROOT/'experiments/mobile_vla/scripts'))
from rpc import call
from kinematics import forward

def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--joint-smoke',action='store_true');a=p.parse_args()
    rclpy.init();n=Node('perception_moveit_client');result={}
    def wait(f,timeout=30):
        end=time.monotonic()+timeout
        while not f.done():
            if time.monotonic()>end:raise TimeoutError('ROS request deadline')
            rclpy.spin_once(n,timeout_sec=.05)
        return f.result()
    def service(typ,name):
        c=n.create_client(typ,name)
        if not c.wait_for_service(timeout_sec=60):raise TimeoutError(name)
        return c
    try:
        capture=call(a.socket,'capture_grounding');state=capture['state'];B=np.array(state['base_R']);bp=np.array(state['base_p'])
        scene=PlanningScene();scene.is_diff=True;scene.robot_state.is_diff=True
        for g in capture['collision_geometries']:
            obj=CollisionObject();obj.id=g['id'];obj.header.frame_id='panda_link0';obj.operation=CollisionObject.ADD
            pose=Pose();pose.position.x,pose.position.y,pose.position.z=map(float,B.T@(np.array(g['p'])-bp));quat=Rotation.from_matrix(B.T@np.array(g['R'])).as_quat();pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,quat)
            if g['type']=='mesh':
                mesh=Mesh();mesh.vertices=[Point(x=float(v[0]),y=float(v[1]),z=float(v[2])) for v in g['vertices']];mesh.triangles=[MeshTriangle(vertex_indices=[int(k) for k in f]) for f in g['triangles']];obj.meshes=[mesh];obj.mesh_poses=[pose]
            else:
                primitive=SolidPrimitive();primitive.type={'box':SolidPrimitive.BOX,'sphere':SolidPrimitive.SPHERE,'cylinder':SolidPrimitive.CYLINDER}[g['type']];primitive.dimensions=list(map(float,g['dimensions']));obj.primitives=[primitive];obj.primitive_poses=[pose]
            scene.world.collision_objects.append(obj)
        # Physical fixed mount is intentionally adjacent to Panda link0 only.
        import xml.etree.ElementTree as ET
        from kinematics import robot_xml,semantic_xml
        names=[x.get('name') for x in ET.fromstring(robot_xml()).findall('link')]+['mobile_column','mobile_chassis']
        disabled={frozenset((x.get('link1'),x.get('link2'))) for x in ET.fromstring(semantic_xml()).findall('disable_collisions')}
        disabled.update([frozenset(('panda_link0','mobile_column')),frozenset(('panda_link0','mobile_chassis'))])
        scene.allowed_collision_matrix.entry_names=names
        from moveit_msgs.msg import AllowedCollisionEntry
        scene.allowed_collision_matrix.entry_values=[AllowedCollisionEntry(enabled=[a==b or frozenset((a,b)) in disabled for b in names]) for a in names]
        apply=service(ApplyPlanningScene,'apply_planning_scene');request=ApplyPlanningScene.Request();request.scene=scene
        assert wait(apply.call_async(request)).success
        names_arm=[f'panda_joint{i}' for i in range(1,8)]
        valid=service(GetStateValidity,'check_state_validity');req=GetStateValidity.Request();req.group_name='panda_arm';req.robot_state.joint_state.name=names_arm+['panda_finger_joint1','panda_finger_joint2'];req.robot_state.joint_state.position=state['joint_positions']+state['finger_positions']
        vr=wait(valid.call_async(req));result['start_state_valid']=vr.valid;result['start_contacts']=[{'body1':c.contact_body_1,'body2':c.contact_body_2} for c in vr.contacts]
        if not vr.valid:raise RuntimeError(f'Invalid planning start: {result["start_contacts"]}')
        fk=service(GetPositionFK,'compute_fk');reqfk=GetPositionFK.Request();reqfk.header.frame_id='panda_link0';reqfk.fk_link_names=['panda_tcp'];reqfk.robot_state=req.robot_state
        actualfk=wait(fk.call_async(reqfk));fp=actualfk.pose_stamped[0].pose.position
        physical_local=B.T@(np.array(state['tcp_p'])-bp)
        result['moveit_physics_fk_error_m']=float(np.linalg.norm(np.array([fp.x,fp.y,fp.z])-physical_local))
        if result['moveit_physics_fk_error_m']>1e-5:raise RuntimeError('MoveIt/physics FK mismatch')
        client=ActionClient(n,MoveGroup,'move_action')
        if not client.wait_for_server(timeout_sec=30):raise TimeoutError('MoveGroup')
        goal=MoveGroup.Goal();r=goal.request;r.group_name='panda_arm';r.pipeline_id='ompl';r.planner_id='RRTConnectkConfigDefault';r.allowed_planning_time=8.;r.num_planning_attempts=3;r.max_velocity_scaling_factor=.12;r.max_acceleration_scaling_factor=.12;r.start_state=req.robot_state
        con=Constraints()
        if a.joint_smoke:
            target=np.array(state['joint_positions']);target[0]+=.05
            for j,v in enumerate(target):con.joint_constraints.append(JointConstraint(joint_name=names_arm[j],position=float(v),tolerance_above=.005,tolerance_below=.005,weight=1.))
        else:
            g=json.loads((a.out/'goal.json').read_text());pos=PositionConstraint();pos.header.frame_id='panda_link0';pos.link_name='panda_tcp';pos.weight=1.
            ball=SolidPrimitive();ball.type=SolidPrimitive.SPHERE;ball.dimensions=[.008];pose=Pose();pose.position.x,pose.position.y,pose.position.z=g['target_position'];pose.orientation.w=1.;pos.constraint_region.primitives=[ball];pos.constraint_region.primitive_poses=[pose];con.position_constraints=[pos]
            ori=OrientationConstraint();ori.header.frame_id='panda_link0';ori.link_name='panda_tcp';ori.orientation.x,ori.orientation.y,ori.orientation.z,ori.orientation.w=g['target_orientation'];ori.absolute_x_axis_tolerance=.10;ori.absolute_y_axis_tolerance=.10;ori.absolute_z_axis_tolerance=.15;ori.weight=1.;con.orientation_constraints=[ori]
        r.goal_constraints=[con];goal.planning_options.plan_only=False;goal.planning_options.planning_scene_diff.is_diff=True
        handle=wait(client.send_goal_async(goal))
        if not handle.accepted:raise RuntimeError('MoveGroup rejected')
        response=wait(handle.get_result_async(),150);v=response.result
        result.update(moveit_error_code=v.error_code.val,action_status=response.status,planned_joint_names=list(v.planned_trajectory.joint_trajectory.joint_names),planned_points=[list(p.positions) for p in v.planned_trajectory.joint_trajectory.points],executed_points=len(v.executed_trajectory.joint_trajectory.points),final_state=call(a.socket,'arm_state'))
        if v.error_code.val!=1:raise RuntimeError(f'MoveIt execution returned {v.error_code.val}')
        result['success']=True
    except Exception as e:result['success']=False;result['error']=repr(e)
    finally:
        (a.out/'moveit_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);n.destroy_node();rclpy.shutdown()
    return 0 if result.get('success') else 1
if __name__=='__main__':sys.exit(main())
