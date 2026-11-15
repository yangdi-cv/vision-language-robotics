"""Installed MoveIt/KDL/OMPL with an external physics-backed action controller.
No mock hardware, fake controllers or second simulator is started.
"""
from pathlib import Path
import sys,yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kinematics import robot_xml,semantic_xml

def generate_launch_description():
    share=Path(get_package_share_directory('moveit_resources_panda_moveit_config'))/'config'
    kin=yaml.safe_load((share/'kinematics.yaml').read_text());kin['panda_arm']['kinematics_solver_timeout']=.3
    ompl=yaml.safe_load((share/'ompl_planning.yaml').read_text())
    limits=yaml.safe_load((share/'joint_limits.yaml').read_text())
    for value in limits['joint_limits'].values():
        if value.get('has_acceleration_limits'):value['max_acceleration']=float(value['max_acceleration'])
    desc={'robot_description':robot_xml()}
    params={**desc,'robot_description_semantic':semantic_xml(),'robot_description_kinematics':kin,'robot_description_planning':limits,
            'planning_pipelines':['ompl'],'default_planning_pipeline':'ompl','ompl':ompl,
            'moveit_controller_manager':'moveit_simple_controller_manager/MoveItSimpleControllerManager',
            'moveit_manage_controllers':False,
            'moveit_simple_controller_manager':{'controller_names':['panda_arm_controller'],'panda_arm_controller':{'type':'FollowJointTrajectory','action_ns':'follow_joint_trajectory','default':True,'joints':[f'panda_joint{i}' for i in range(1,8)]}},
            'trajectory_execution.allowed_execution_duration_scaling':2.,'trajectory_execution.allowed_goal_duration_margin':3.,'trajectory_execution.allowed_start_tolerance':.03,
            'publish_robot_description_semantic':True,'use_sim_time':False}
    return LaunchDescription([Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[desc],output='screen'),
                             Node(package='moveit_ros_move_group',executable='move_group',parameters=[params],output='screen')])
