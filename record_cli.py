"""CLI to record a single episode using a saved model."""
import argparse
from src.agent import DQNAgent
from src.record import record_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model .pth')
    parser.add_argument('--email', type=str, default='estudiante@uvg.edu.gt')
    parser.add_argument('--env', type=str, default='ALE/Galaxian-v5')
    parser.add_argument('--output_dir', type=str, default='videos')
    args = parser.parse_args()

    state_shape = (4, 84, 84)
    n_actions = 6
    agent = DQNAgent(state_shape, n_actions)
    agent.load(args.model, map_location='cpu')

    record_episode(agent, email=args.email, output_dir=args.output_dir, env_name=args.env)


if __name__ == '__main__':
    main()
