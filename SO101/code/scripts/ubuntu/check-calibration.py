from pathlib import Path
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
root = Path(__file__).resolve().parents[3]
for role, cls, config, serial, directory in [
 ('black leader', SO101Leader, SO101LeaderConfig, '5CE5121669', 'teleoperators/so_leader'),
 ('white follower', SO101Follower, SO101FollowerConfig, '5CE5122996', 'robots/so_follower'),
]:
 device=cls(config(port=f'/dev/serial/by-id/usb-1a86_USB_Single_Serial_{serial}-if00',id=f'my_awesome_{"leader" if cls is SO101Leader else "follower"}_arm',calibration_dir=root/'calibration'/directory,use_degrees=True))
 try:
  device.bus.connect()
  print(role, 'calibration_matches:', device.bus.is_calibrated, flush=True)
  print('positions:',device.bus.sync_read('Present_Position', normalize=False),flush=True)
  print('torque:',device.bus.sync_read('Torque_Enable',normalize=False),flush=True)
 finally:
  if device.bus.is_connected: device.bus.disconnect(disable_torque=False)
