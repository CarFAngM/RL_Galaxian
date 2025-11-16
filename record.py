
from src.record import record_episode as _record_episode


def record_episode(agent, email="estudiante@uvg.edu.gt", output_dir="videos"):
	"""Wrapper to call the src.record.record_episode function.

	Keeps compatibility with previous usage.
	"""
	return _record_episode(agent, email=email, output_dir=output_dir)


if __name__ == '__main__':
	# Minimal CLI for quick use
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', required=True)
	parser.add_argument('--email', default='estudiante@uvg.edu.gt')
	parser.add_argument('--output_dir', default='videos')
	args = parser.parse_args()

	from src.agent import DQNAgent

	agent = DQNAgent((4, 84, 84), 6)
	agent.load(args.model, map_location='cpu')
	_record_episode(agent, email=args.email, output_dir=args.output_dir)
