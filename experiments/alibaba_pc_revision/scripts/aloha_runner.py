"""One ALOHA execution/video loop shared by evaluation and language dispatch."""

import time
import av, numpy as np, torch


class VideoWriter:
    def __init__(self, path):
        self.file = av.open(str(path), "w")
        self.stream = self.file.add_stream("libx264", rate=25)
        self.stream.width = 640
        self.stream.height = 480
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "22", "preset": "fast"}

    def write(self, image):
        for p in self.stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            self.file.mux(p)

    def close(self):
        for p in self.stream.encode():
            self.file.mux(p)
        self.file.close()


def rollout_episode(policy, env, obs, video_path=None):
    policy.reset()
    video = VideoWriter(video_path) if video_path else None
    rewards = []
    actions = []
    timing = []
    begin = time.perf_counter()
    try:
        for step in range(400):
            rgb = np.asarray(obs["pixels"]["top"])
            if video and step % 2 == 0:
                video.write(rgb)
            batch = {
                "observation.state": torch.as_tensor(
                    obs["agent_pos"], dtype=torch.float32, device="cuda"
                )[None],
                "observation.images.top": torch.from_numpy(rgb.copy())
                .permute(2, 0, 1)[None]
                .to("cuda", dtype=torch.float32)
                / 255,
            }
            full = len(policy._action_queue) == 0
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)[0].cpu().numpy()
            torch.cuda.synchronize()
            timing.append([step, full, time.perf_counter() - tick])
            assert np.isfinite(action).all()
            obs, r, terminated, truncated, info = env.step(action)
            rewards.append(float(r))
            actions.append(action.tolist())
            if terminated or truncated:
                break
    finally:
        if video:
            video.close()
    return {
        "max_reward": max(rewards),
        "success": max(rewards) >= 4,
        "steps": len(rewards),
        "wall_seconds": time.perf_counter() - begin,
        "rewards": rewards,
        "actions": actions,
        "timing": timing,
    }
