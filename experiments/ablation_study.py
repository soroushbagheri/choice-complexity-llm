"""ablation_study.py

Ablation study comparing different controller components and configurations.

Usage:
    python experiments/ablation_study.py --baseline results/run_baseline
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from experiments.run_benchmark import BenchmarkRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationStudy:
    """Systematic ablation of controller components."""
    
    def __init__(self, baseline_config: Dict):
        self.baseline_config = baseline_config
        self.results = {}
    
    def run_ablations(self) -> Dict[str, pd.DataFrame]:
        """Run systematic ablations."""
        ablation_configs = [
            ('baseline', self.baseline_config),
            ('no_redundancy_removal', self._ablate_redundancy()),
            ('no_clustering', self._ablate_clustering()),
            ('no_satisficing', self._ablate_satisficing()),
            ('cci_only', self._cci_only()),
            ('ildc_only', self._ildc_only()),
            ('naive_topk', self._naive_topk()),
        ]
        
        for name, config in ablation_configs:
            logger.info(f"\nRunning ablation: {name}")
            runner = BenchmarkRunner(config)
            datasets = runner.generate_datasets()
            results_df = runner.run_experiments(datasets)
            self.results[name] = results_df
        
        return self.results
    
    def _ablate_redundancy(self) -> Dict:
        config = self.baseline_config.copy()
        config['controller']['remove_redundancy'] = False
        return config
    
    def _ablate_clustering(self) -> Dict:
        config = self.baseline_config.copy()
        config['controller']['use_clustering'] = False
        return config
    
    def _ablate_satisficing(self) -> Dict:
        config = self.baseline_config.copy()
        config['controller']['satisficing_threshold'] = None
        return config
    
    def _cci_only(self) -> Dict:
        config = self.baseline_config.copy()
        config['controller_strategies'] = ['cci_only']
        return config
    
    def _ildc_only(self) -> Dict:
        config = self.baseline_config.copy()
        config['controller_strategies'] = ['ildc_only']
        return config
    
    def _naive_topk(self) -> Dict:
        config = self.baseline_config.copy()
        config['controller_strategies'] = ['naive_topk']
        return config
    
    def compare_ablations(self) -> pd.DataFrame:
        """Compare metrics across ablations."""
        comparison = []
        
        for name, df in self.results.items():
            comparison.append({
                'ablation': name,
                'mean_volatility': df['volatility'].mean(),
                'mean_accuracy': df['accuracy'].mean() if 'accuracy' in df else None,
                'mean_cci': df['cci_score'].mean(),
                'mean_ildc': df['ildc_score'].mean(),
                'mean_tokens': df['total_tokens'].mean(),
            })
        
        return pd.DataFrame(comparison)


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Baseline results directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/ablations',
        help='Output directory'
    )
    args = parser.parse_args()
    
    # Load baseline config
    baseline_dir = Path(args.baseline)
    config_path = baseline_dir / 'config.yaml'
    config = load_config(str(config_path))
    
    # Run ablations
    study = AblationStudy(config)
    results = study.run_ablations()
    
    # Compare results
    comparison = study.compare_ablations()
    
    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / 'ablation_comparison.csv', index=False)
    
    logger.info(f"\nAblation results saved to {output_dir}")
    logger.info("\nComparison:")
    print(comparison.to_string())


if __name__ == '__main__':
    main()
