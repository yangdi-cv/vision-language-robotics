from pathlib import Path
import argparse,gymnasium as gym,gym_aloha
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();env=gym.make('gym_aloha/AlohaTransferCube-v0',obs_type='pixels_agent_pos');obs,_=env.reset(seed=1003);Image.fromarray(obs['pixels']['top']).save(a.out);env.close()
