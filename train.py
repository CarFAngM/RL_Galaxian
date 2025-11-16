"""CLI entrypoint to train a DQN agent on Galaxian."""
import argparse
from src.train import train_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--email', type=str, default='estudiante@uvg.edu.gt')
    parser.add_argument('--env', type=str, default='ALE/Galaxian-v5')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--ma_window', type=int, default=10)
    args = parser.parse_args()

    train_agent(episodes=args.episodes, email=args.email, env_name=args.env,
                checkpoint_dir=args.checkpoint_dir, early_stop_patience=args.patience,
                ma_window=args.ma_window)


if __name__ == '__main__':
    main()
