import os
from collections import deque
import numpy as np
import gymnasium as gym

from .ac_agent import ActorCriticAgent
from .utils import preprocess_frame, stack_frames, plot_training_metrics


def train_ac_agent(episodes=100, email="estudiante@uvg.edu.gt", checkpoint_dir='checkpoints_ac',
                   env_name='ALE/Galaxian-v5', early_stop_patience=50, ma_window=10,
                   save_every=50, device=None):
    """Train Actor-Critic agent on Galaxian.
    
    Early stopping: stops if average reward doesn't improve for `early_stop_patience` episodes.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = gym.make(env_name)
    n_actions = env.action_space.n
    state_shape = (4, 84, 84)

    agent = ActorCriticAgent(state_shape, n_actions, device=device)
    stacked = deque(maxlen=4)

    # Track best average reward (higher is better)
    best_avg_reward = -np.inf
    no_improve = 0

    rewards = []
    actor_losses = []
    critic_losses = []
    entropies = []
    durations = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        processed = preprocess_frame(state)
        state_stack = stack_frames(stacked, processed, True)

        done = False
        truncated = False
        ep_reward = 0
        ep_steps = 0

        # Reset trajectory for new episode
        agent.reset_trajectory()

        while not (done or truncated):
            action = agent.select_action(state_stack, training=True)
            next_state, reward, done, truncated, _ = env.step(action)

            processed = preprocess_frame(next_state)
            next_stack = stack_frames(stacked, processed, False)

            # NO clipear reward - queremos que el agente aprenda valores reales
            agent.store_reward_done(reward, done or truncated)

            ep_reward += reward
            ep_steps += 1
            state_stack = next_stack

        # Train after episode ends
        actor_loss, critic_loss, entropy = agent.train_step()

        rewards.append(ep_reward)
        durations.append(ep_steps)
        
        if actor_loss is not None:
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropies.append(entropy)

        # Calculate moving average of rewards
        if len(rewards) >= ma_window:
            ma_reward = np.mean(rewards[-ma_window:])
        else:
            ma_reward = np.mean(rewards) if rewards else -np.inf

        # Format losses for printing
        a_loss_str = f"{actor_loss:.4f}" if actor_loss is not None else "0.0000"
        c_loss_str = f"{critic_loss:.4f}" if critic_loss is not None else "0.0000"
        entropy_str = f"{entropy:.4f}" if entropy is not None else "0.0000"
        
        print(f"Ep {ep}/{episodes} | Reward: {ep_reward:.1f} | MA_Reward: {ma_reward:.1f} | "
              f"A_Loss: {a_loss_str} | C_Loss: {c_loss_str} | Entropy: {entropy_str}")

        # Guardar mejor modelo basado en REWARD (mayor es mejor)
        if ma_reward > best_avg_reward:
            best_avg_reward = ma_reward
            no_improve = 0
            
            email_prefix = email.split('@')[0]
            best_path = os.path.join(checkpoint_dir, f'best_model_ac_{email_prefix}.pth')
            agent.save(best_path)
            print(f"  ✓ Nuevo mejor MA_Reward: {ma_reward:.1f} → Modelo guardado")
        else:
            no_improve += 1

        # Periodic checkpoint
        if ep % save_every == 0:
            cp_path = os.path.join(checkpoint_dir, f'checkpoint_ac_ep{ep}_{email.split("@")[0]}.pth')
            agent.save(cp_path)
            print(f"  → Checkpoint guardado: ep{ep}")

        # Early stopping based on reward
        if no_improve >= early_stop_patience:
            print(f"Early stopping: {no_improve} episodios sin mejora")
            break

    env.close()

    # Guardar modelo final
    email_prefix = email.split('@')[0]
    final_path = os.path.join(checkpoint_dir, f'final_model_ac_{email_prefix}.pth')
    agent.save(final_path)
    print(f"\n✓ Modelo final guardado: {final_path}")

    # Guardar gráficas
    plot_path = plot_training_metrics(
        rewards, actor_losses, durations, 
        email_prefix + '_ac', 
        output_dir=checkpoint_dir
    )
    print(f"✓ Métricas guardadas: {plot_path}")

    return agent, {
        'rewards': rewards, 
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'entropies': entropies,
        'durations': durations
    }
