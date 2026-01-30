"""run_benchmark.py

Main experiment runner for Decision-Theoretic Choice Complexity benchmark.
Executes full evaluation pipeline with multiple controller strategies and generates
comprehensive metrics and visualizations.

Usage:
    python experiments/run_benchmark.py --config configs/default.yaml
    python experiments/run_benchmark.py --config configs/ablation.yaml --output results/ablation_v1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import SyntheticChoiceDataset, ConsumerChoiceDataset
from src.cci import ChoiceComplexityIndex
from src.ildc import ILDCComputer
from src.controller import ChoiceController
from src.llm_adapter import LLMAdapter
from src.utils import set_seed, save_results, load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates full benchmark evaluation pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        set_seed(config.get('seed', 42))
        
        # Initialize components
        self.cci_calculator = ChoiceComplexityIndex(**config.get('cci', {}))
        self.ildc_calculator = ILDCComputer(**config.get('ildc', {}))
        self.llm = LLMAdapter(**config.get('llm', {}))
        
        # Results storage
        self.results = []
        self.metrics_summary = {}
        
    def generate_datasets(self) -> Dict[str, List[Dict]]:
        """Generate all benchmark datasets."""
        logger.info("Generating benchmark datasets...")
        datasets = {}
        
        # Dataset A: Synthetic with controlled complexity
        if self.config.get('datasets', {}).get('synthetic', True):
            synthetic_config = self.config.get('synthetic_config', {})
            synthetic_gen = SyntheticChoiceDataset(**synthetic_config)
            datasets['synthetic'] = synthetic_gen.generate_dataset(
                n_samples=self.config.get('n_samples_per_dataset', 100)
            )
            logger.info(f"Generated {len(datasets['synthetic'])} synthetic samples")
        
        # Dataset B: Consumer choice
        if self.config.get('datasets', {}).get('consumer', True):
            consumer_config = self.config.get('consumer_config', {})
            consumer_gen = ConsumerChoiceDataset(**consumer_config)
            datasets['consumer'] = consumer_gen.generate_dataset(
                n_samples=self.config.get('n_samples_per_dataset', 100)
            )
            logger.info(f"Generated {len(datasets['consumer'])} consumer choice samples")
        
        return datasets
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        controller_strategy: str = 'none'
    ) -> Dict[str, Any]:
        """Evaluate a single choice problem.
        
        Args:
            sample: Choice problem with options and ground truth
            controller_strategy: Which controller to use ('none', 'cci_only', 'ildc_only', 'two_tier', etc.)
        
        Returns:
            Dictionary with metrics for this sample
        """
        options = sample['options']
        ground_truth = sample.get('ground_truth_choice')
        
        # Step 1: Compute CCI
        cci_result = self.cci_calculator.compute(options)
        cci_score = cci_result['cci_score']
        cci_features = cci_result['features']
        
        # Step 2: Apply controller (may modify options)
        controller = ChoiceController(strategy=controller_strategy)
        controlled_options, controller_action = controller.apply(
            options=options,
            cci_score=cci_score,
            cci_features=cci_features
        )
        
        # Step 3: LLM makes decision (with multiple samples for ILDC)
        n_samples = self.config.get('ildc', {}).get('n_samples', 5)
        choices = []
        for _ in range(n_samples):
            response = self.llm.choose(
                options=controlled_options,
                context=sample.get('context', ''),
                temperature=0.7
            )
            choices.append(response)
        
        # Step 4: Compute ILDC
        ildc_result = self.ildc_calculator.compute(
            options=controlled_options,
            choices=choices,
            llm=self.llm
        )
        ildc_score = ildc_result['ildc_score']
        ildc_features = ildc_result['features']
        
        # Step 5: Compute metrics
        primary_choice = choices[0]['choice']
        
        # Accuracy (if ground truth available)
        accuracy = None
        if ground_truth is not None:
            # Map chosen option back to original if controller modified
            original_choice = self._map_to_original(primary_choice, controlled_options, options)
            accuracy = 1.0 if original_choice == ground_truth else 0.0
        
        # Volatility
        choice_ids = [c['choice'] for c in choices]
        volatility = 1.0 - (choice_ids.count(choice_ids[0]) / len(choice_ids))
        
        # Token usage
        total_tokens = sum(c.get('tokens_used', 0) for c in choices)
        
        # Latency
        total_latency = sum(c.get('latency', 0) for c in choices)
        
        return {
            'sample_id': sample.get('id'),
            'dataset': sample.get('dataset'),
            'n_options': len(options),
            'n_options_shown': len(controlled_options),
            'controller_strategy': controller_strategy,
            'controller_action': controller_action,
            'cci_score': cci_score,
            'ildc_score': ildc_score,
            'accuracy': accuracy,
            'volatility': volatility,
            'total_tokens': total_tokens,
            'avg_latency': total_latency / n_samples,
            'primary_choice': primary_choice,
            **cci_features,
            **ildc_features
        }
    
    def _map_to_original(self, choice_idx, controlled_options, original_options):
        """Map choice in controlled set back to original option set."""
        if len(controlled_options) == len(original_options):
            return choice_idx
        
        # Find which original option this corresponds to
        chosen_option = controlled_options[choice_idx]
        for i, orig_option in enumerate(original_options):
            if orig_option['id'] == chosen_option['id']:
                return i
        return choice_idx  # Fallback
    
    def run_experiments(self, datasets: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Run experiments across all datasets and controller strategies."""
        strategies = self.config.get('controller_strategies', [
            'none',
            'naive_topk',
            'cci_only',
            'ildc_only',
            'two_tier'
        ])
        
        logger.info(f"Running experiments with strategies: {strategies}")
        
        all_results = []
        
        for dataset_name, samples in datasets.items():
            logger.info(f"\nProcessing dataset: {dataset_name}")
            
            for strategy in strategies:
                logger.info(f"  Strategy: {strategy}")
                
                for sample in tqdm(samples, desc=f"{dataset_name}-{strategy}"):
                    try:
                        result = self.evaluate_sample(sample, strategy)
                        result['dataset'] = dataset_name
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing sample {sample.get('id')}: {e}")
                        continue
        
        return pd.DataFrame(all_results)
    
    def compute_summary_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate metrics across experiments."""
        summary = {}
        
        # Group by controller strategy
        for strategy in results_df['controller_strategy'].unique():
            strategy_df = results_df[results_df['controller_strategy'] == strategy]
            
            summary[strategy] = {
                'mean_accuracy': strategy_df['accuracy'].mean(),
                'mean_volatility': strategy_df['volatility'].mean(),
                'mean_cci': strategy_df['cci_score'].mean(),
                'mean_ildc': strategy_df['ildc_score'].mean(),
                'mean_tokens': strategy_df['total_tokens'].mean(),
                'mean_latency': strategy_df['avg_latency'].mean(),
                'mean_options_shown': strategy_df['n_options_shown'].mean(),
            }
        
        # Correlations
        summary['correlations'] = {
            'cci_vs_ildc': results_df[['cci_score', 'ildc_score']].corr().iloc[0, 1],
            'cci_vs_volatility': results_df[['cci_score', 'volatility']].corr().iloc[0, 1],
            'ildc_vs_volatility': results_df[['ildc_score', 'volatility']].corr().iloc[0, 1],
        }
        
        if 'accuracy' in results_df.columns and results_df['accuracy'].notna().any():
            summary['correlations']['cci_vs_accuracy'] = results_df[['cci_score', 'accuracy']].corr().iloc[0, 1]
            summary['correlations']['ildc_vs_accuracy'] = results_df[['ildc_score', 'accuracy']].corr().iloc[0, 1]
        
        return summary
    
    def save_results(self, results_df: pd.DataFrame, summary: Dict, output_dir: Path):
        """Save all results and metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_path = output_dir / 'results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")
        
        # Save summary metrics
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")
        
        # Save config for reproducibility
        config_path = output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f"Saved config to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Decision-Theoretic Choice Complexity benchmark'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results/run_TIMESTAMP)'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results') / f'run_{timestamp}'
    
    logger.info(f"Starting benchmark with config: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize runner
    runner = BenchmarkRunner(config)
    
    # Generate datasets
    datasets = runner.generate_datasets()
    
    # Run experiments
    results_df = runner.run_experiments(datasets)
    
    # Compute summary metrics
    summary = runner.compute_summary_metrics(results_df)
    
    # Save results
    runner.save_results(results_df, summary, output_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*80)
    logger.info(f"Total samples evaluated: {len(results_df)}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nKey Findings:")
    for strategy, metrics in summary.items():
        if strategy != 'correlations':
            logger.info(f"\n{strategy}:")
            logger.info(f"  Accuracy: {metrics.get('mean_accuracy', 'N/A'):.3f}")
            logger.info(f"  Volatility: {metrics['mean_volatility']:.3f}")
            logger.info(f"  CCI: {metrics['mean_cci']:.3f}")
            logger.info(f"  ILDC: {metrics['mean_ildc']:.3f}")
    
    logger.info("\nCorrelations:")
    for k, v in summary['correlations'].items():
        logger.info(f"  {k}: {v:.3f}")
    
    logger.info("\nNext steps:")
    logger.info(f"  1. Generate plots: python experiments/plotting.py --input {output_dir}")
    logger.info(f"  2. Run ablations: python experiments/ablation_study.py --baseline {output_dir}")


if __name__ == '__main__':
    main()
