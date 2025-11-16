import os
from datetime import datetime
import cv2
from collections import deque

from .utils import preprocess_frame, stack_frames


def record_episode(agent, email="estudiante@uvg.edu.gt", output_dir="videos", env_name='ALE/Galaxian-v5'):
    os.makedirs(output_dir, exist_ok=True)
    import gymnasium as gym

    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    truncated = False
    frames = []
    total_reward = 0

    stacked_frames = deque(maxlen=4)
    processed = preprocess_frame(state)
    state_stack = stack_frames(stacked_frames, processed, True)

    while not (done or truncated):
        frame = env.render()
        frames.append(frame)

        action = agent.select_action(state_stack, training=False)
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        processed = preprocess_frame(next_state)
        state_stack = stack_frames(stacked_frames, processed, False)

    env.close()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    email_prefix = email.split('@')[0]
    score = int(total_reward)
    filename = f"{email_prefix}_{timestamp}_{score}.mp4"
    filepath = os.path.join(output_dir, filename)

    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 30.0, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()

    print(f"Video saved: {filepath}")
    print(f"Score: {score} | Frames: {len(frames)}")
    return filepath
