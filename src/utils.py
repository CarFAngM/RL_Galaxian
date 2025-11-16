import os
from collections import deque
import numpy as np
import cv2
import matplotlib.pyplot as plt


def preprocess_frame(frame):
    """Convert RGB frame to grayscale 84x84 normalized float32."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def stack_frames(frames: deque, new_frame, is_new_episode: bool):
    if is_new_episode:
        frames.clear()
        for _ in range(4):
            frames.append(new_frame)
    else:
        frames.append(new_frame)

    return np.stack(frames, axis=0)


def plot_training_metrics(rewards, losses, durations, email_prefix, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Métricas de Entrenamiento DQN - Galaxian', fontsize=14)

    axes[0].plot(rewards, label='Reward')
    axes[0].set_title('Recompensa por episodio')
    axes[0].grid(True)
    if len(rewards) >= 10:
        ma = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        axes[0].plot(range(9, len(rewards)), ma, label='MA(10)')
        axes[0].legend()

    if losses:
        axes[1].plot(losses, color='orange')
        axes[1].set_title('Pérdida')
        axes[1].grid(True)

    axes[2].plot(durations, color='green')
    axes[2].set_title('Duración')
    axes[2].grid(True)

    plt.tight_layout()
    filename = os.path.join(output_dir, f'training_metrics_{email_prefix}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename
