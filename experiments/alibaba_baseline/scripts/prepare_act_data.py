"""Decode public episodes into a memory-mapped RGB cache, preserving episode boundaries."""
import json
from pathlib import Path
import av,numpy as np,pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_baseline';data=TASK/'data/aloha_transfer';cache=TASK/'data/act_cache';cache.mkdir(exist_ok=True)
info=json.loads((data/'meta/info.json').read_text());total=info['total_frames']
images=np.lib.format.open_memmap(cache/'images.npy',mode='w+',dtype=np.uint8,shape=(total,480,640,3))
states=[];actions=[];episodes=[];offset=0;boundaries=[]
for episode in range(50):
 file=data/'data/chunk-000'/f'episode_{episode:06d}.parquet';rows=pq.read_table(file).to_pylist();length=len(rows)
 video=data/'videos/chunk-000/observation.images.top'/f'episode_{episode:06d}.mp4'
 count=0
 with av.open(str(video)) as container:
  container.streams.video[0].thread_type='AUTO'
  for frame in container.decode(video=0):
   if count>=length:raise ValueError('Video has more frames than action episode')
   image=frame.to_ndarray(format='rgb24');assert image.shape==(480,640,3)
   images[offset+count]=image;count+=1
 assert count==length,(episode,count,length)
 states.extend(r['observation.state'] for r in rows);actions.extend(r['action'] for r in rows);episodes.extend([episode]*length)
 boundaries.append([offset,offset+length]);offset+=length
 print('decoded episode',episode,'frames',count,flush=True)
assert offset==total;images.flush()
np.save(cache/'states.npy',np.asarray(states,dtype=np.float32));np.save(cache/'actions.npy',np.asarray(actions,dtype=np.float32));np.save(cache/'episodes.npy',np.asarray(episodes,dtype=np.int32))
statistics=[json.loads(line) for line in (data/'meta/episodes_stats.jsonl').read_text().splitlines()]
train_stats=aggregate_stats([{key:{name:np.asarray(value) for name,value in fields.items()} for key,fields in row['stats'].items()} for row in statistics if row['episode_index']<40])
(cache/'train_stats.json').write_text(json.dumps({key:{name:value.tolist() for name,value in fields.items()} for key,fields in train_stats.items()},indent=2))
(cache/'manifest.json').write_text(json.dumps({'source':'lerobot/aloha_sim_transfer_cube_human','revision':'8f81f044cbc65d9c1efbe5a302e8f9f4c16a86ed','frames':total,'boundaries':boundaries,'train_episodes':list(range(40)),'validation_episodes':list(range(40,45)),'test_episodes':list(range(45,50)),'image_shape':[480,640,3]},indent=2))
print('Cache complete',flush=True)
