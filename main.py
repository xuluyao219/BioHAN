"""
BioHAN - Main entry point for experiments
"""

import argparse
import yaml
from pathlib import Path
import torch
import warnings

warnings.filterwarnings('ignore')

from experiments.run_experiments import ExperimentRunner
from experiments.cross_dataset_consistency import run_consistency_analysis
from visualization.paper_figures import FigureGenerator


def load_config(config_path: str = 'configs/default_config.yaml') -> dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='BioHAN - Biologically-Informed Hierarchical Attention Networks')

    parser.add_argument('--mode', type=str, default='standard',
                        choices=['quick', 'standard', 'full', 'consistency', 'figures'],
                        help='Experiment mode')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Specific methods to run')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set GPU
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Override config with command line arguments
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = config['experiments']['modes'][args.mode]['datasets']
        if datasets == 'all':
            datasets = config['datasets']['available']

    if args.methods:
        methods = args.methods
    else:
        methods = config['experiments']['modes'][args.mode]['methods']
        if methods == 'all':
            methods = config['experiments']['methods']

    if args.epochs:
        n_epochs = args.epochs
    else:
        n_epochs = config['experiments']['modes'][args.mode]['n_epochs']

    print(f"\nRunning in {args.mode} mode")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Epochs: {n_epochs}\n")

    if args.mode == 'consistency':
        # Run cross-dataset consistency analysis
        print("Running cross-dataset consistency analysis...")
        results = run_consistency_analysis()

    elif args.mode == 'figures':
        # Generate figures from existing results
        if args.results_dir:
            fig_gen = FigureGenerator(args.results_dir)
            fig_gen.generate_all_figures()
        else:
            print("Please specify --results_dir for figure generation")

    else:
        # Run standard experiments
        runner = ExperimentRunner(
            output_dir=args.results_dir or config['paths']['results_dir']
        )

        results = runner.run_all_experiments(
            datasets=datasets,
            methods=methods,
            n_epochs=n_epochs
        )

        # Generate figures
        print("\nGenerating figures...")
        fig_gen = FigureGenerator(runner.output_dir)
        fig_gen.generate_all_figures()

    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    main()