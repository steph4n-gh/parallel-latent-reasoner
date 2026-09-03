"""Command-line interface dispatch for Parallel Latent Reasoner."""
import sys
from pathlib import Path

def main_demo():
    from parallel_latent_reasoner.visualizer import run_visualizer_demo
    import argparse
    parser = argparse.ArgumentParser(description='Parallel Latent Reasoner (PRLR) Demo')
    parser.add_argument('--prompt', type=str, default='What is 25 * 14?', help='Prompt text')
    parser.add_argument('--preset', type=str, default='compact_test', help='Model scale preset')
    parser.add_argument('--slots', type=int, default=16, help='Working memory slots M')
    parser.add_argument('--steps', type=int, default=8, help='Recurrent unroll steps T')
    args = parser.parse_args()
    run_visualizer_demo(prompt=args.prompt, preset=args.preset, num_slots=args.slots, num_steps=args.steps)

def main_benchmark():
    from parallel_latent_reasoner.benchmark import MultiScaleBenchmarkSuite
    suite = MultiScaleBenchmarkSuite()
    suite.run()

if __name__ == '__main__':
    main_demo()
