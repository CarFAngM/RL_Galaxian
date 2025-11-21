import os
from collections import deque
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from .agent import DQNAgent
from .utils import preprocess_frame, stack_frames, plot_training_metrics

def train_agent(episodes=100, email="estudiante@uvg.edu.gt", checkpoint_dir='checkpoints',
                env_name='ALE/Galaxian-v5', early_stop_patience=50, ma_window=10,
                save_every=50, device=None, save_buffer=False, max_buffer_size=10000):
    """Train DQN agent on Galaxian."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = gym.make(env_name)
    n_actions = env.action_space.n
    state_shape = (4, 84, 84)

    agent = DQNAgent(state_shape, n_actions, device=device)
    stacked = deque(maxlen=4)

    # Cambio: mejor modelo basado en REWARD (puntos), no en loss
    best_avg_reward = -np.inf
    no_improve = 0

    rewards = []
    losses = []
    durations = []

    steps = 0
    update_target_frequency = 10000

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        processed = preprocess_frame(state)
        state_stack = stack_frames(stacked, processed, True)

        done = False
        truncated = False
        ep_reward = 0
        ep_loss_sum = 0
        ep_loss_count = 0
        ep_steps = 0

        while not (done or truncated):
            action = agent.select_action(state_stack, training=True)
            next_state, reward, done, truncated, _ = env.step(action)

            processed = preprocess_frame(next_state)
            next_stack = stack_frames(stacked, processed, False)

            clipped = np.clip(reward, -1, 1)
            agent.memory.push(state_stack, action, clipped, next_stack, float(done))

            loss = agent.train_step()
            if loss is not None:
                ep_loss_sum += loss
                ep_loss_count += 1

            ep_reward += reward
            ep_steps += 1
            steps += 1
            state_stack = next_stack

            if steps % update_target_frequency == 0:
                agent.update_target_network()

        agent.decay_epsilon()

        rewards.append(ep_reward)
        durations.append(ep_steps)
        avg_loss = (ep_loss_sum / ep_loss_count) if ep_loss_count > 0 else 0
        losses.append(avg_loss)

        # Calcular promedio móvil de REWARDS (puntos)
        if len(rewards) >= ma_window:
            ma_reward = np.mean(rewards[-ma_window:])
        else:
            ma_reward = np.mean(rewards) if rewards else -np.inf

        print(f"Ep {ep}/{episodes} | Reward: {ep_reward:.1f} | MA_Reward: {ma_reward:.1f} | Loss: {avg_loss:.4f} | ε: {agent.epsilon:.3f}")

        # Guardar mejor modelo basado en REWARD (mayor es mejor)
        if ma_reward > best_avg_reward:
            best_avg_reward = ma_reward
            no_improve = 0
            
            email_prefix = email.split('@')[0]
            best_path = os.path.join(checkpoint_dir, f'best_model_{email_prefix}.pth')
            agent.save(best_path)
            print(f"  ✓ Nuevo mejor MA_Reward: {ma_reward:.1f} → Modelo guardado")
        else:
            no_improve += 1

        if ep % save_every == 0:
            cp_path = os.path.join(checkpoint_dir, f'checkpoint_ep{ep}_{email.split("@")[0]}.pth')
            agent.save(cp_path)  # Checkpoints sin buffer por defecto
            print(f"  → Checkpoint guardado: ep{ep}")

        if no_improve >= early_stop_patience:
            print(f"Early stopping: {no_improve} episodios sin mejora")
            break

    env.close()

    # Guardar modelo final con buffer
    email_prefix = email.split('@')[0]
    final_path = os.path.join(checkpoint_dir, f'final_model_{email_prefix}.pth')
    agent.save(final_path, save_buffer=save_buffer, max_buffer_size=max_buffer_size)
    print(f"\n✓ Modelo final guardado: {final_path}")
    if save_buffer:
        print(f"  Buffer guardado: {min(len(agent.memory.buffer), max_buffer_size)} experiencias")

    # Guardar gráficas
    plot_path = plot_training_metrics(rewards, losses, durations, email_prefix, output_dir=checkpoint_dir)
    print(f"✓ Métricas guardadas: {plot_path}")

    return agent, {'rewards': rewards, 'losses': losses, 'durations': durations}