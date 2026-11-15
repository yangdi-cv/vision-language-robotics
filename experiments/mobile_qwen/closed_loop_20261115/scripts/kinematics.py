"""Read the installed upstream Panda URDF; apply the measured gripper fixed frames.
No inverse-kinematics/planner implementation is provided here: MoveIt/KDL owns IK.
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

URDF=Path('/opt/ros/humble/share/moveit_resources_panda_description/urdf/panda.urdf')
SRDF=Path('/opt/ros/humble/share/moveit_resources_panda_moveit_config/config/panda.srdf')
HAND_YAW=2*math.atan2(-.382911,.923785)

def robot_xml():
    r=ET.fromstring(URDF.read_text())
    r.find("joint[@name='panda_joint8']/origin").set('xyz','0 0 0.1065')
    r.find("joint[@name='panda_hand_joint']/origin").set('rpy',f'0 0 {HAND_YAW}')
    for name in ['panda_finger_joint1','panda_finger_joint2']:
        r.find(f"joint[@name='{name}']/origin").set('xyz','0 0 0.0524')
    ET.SubElement(r,'link',name='panda_tcp')
    j=ET.SubElement(r,'joint',name='panda_tcp_joint',type='fixed')
    ET.SubElement(j,'parent',link='panda_hand');ET.SubElement(j,'child',link='panda_tcp')
    ET.SubElement(j,'origin',xyz='0 0 0.097',rpy=f'0 0 {-math.pi/2}')
    return ET.tostring(r,encoding='unicode')

def semantic_xml():
    r=ET.fromstring(SRDF.read_text())
    for j in list(r.findall('virtual_joint')):r.remove(j)
    r.find("group[@name='panda_arm']/chain").set('tip_link','panda_tcp')
    return ET.tostring(r,encoding='unicode')

def forward(q,tip='panda_tcp'):
    transforms={'panda_link0':np.eye(4)}
    vals={f'panda_joint{i+1}':v for i,v in enumerate(q)}
    for joint in ET.fromstring(robot_xml()).findall('joint'):
        parent=joint.find('parent').get('link');child=joint.find('child').get('link')
        if parent not in transforms:continue
        origin=joint.find('origin'); t=np.eye(4)
        if origin is not None:
            t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        if joint.get('type')=='revolute':
            a=np.fromstring(joint.find('axis').get('xyz'),sep=' ');v=vals[joint.get('name')]
            turn=np.eye(4);turn[:3,:3]=Rotation.from_rotvec(a*v).as_matrix();t=t@turn
        transforms[child]=transforms[parent]@t
    return transforms[tip]
