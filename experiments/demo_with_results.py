"""demo_with_results.py

Comprehensive demonstration of the Decision-Theoretic Choice Complexity framework.
Generates synthetic results for quick testing and demonstration without requiring LLM API access.

This script:
1. Generates synthetic choice problems with controlled complexity
2. Computes CCI (Choice Complexity Index) for each problem
3. Simulates LLM decisions with realistic volatility patterns
4. Computes ILDC (Internal LLM Decision Complexity)
5. Applies different controller strategies
6. Generates comprehensive metrics and visualizations
7. Produces publication-ready plots

Usage:
    python experiments/demo_with_results.py
    python experiments/demo_with_results.py --n-samples 200 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import SyntheticChoiceDataset
from src.cci import ChoiceComplexityIndex
from src.utils import set_seed

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)


class SyntheticLLMSimulator:
    """Simulates LLM behavior with realistic patterns based on complexity."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def make_decision(
        self,
        options: List[Dict],
        cci_score: float,
        ground_truth: int,
        temperature: float = 0.7
    ) -> Dict:
        """Simulate LLM decision with complexity-dependent error rate.
        
        Higher CCI → higher error probability and volatility
        """
        n_options = len(options)
        
        # Error probability increases with CCI
        # CCI 0.0 → 5% error, CCI 1.0 → 50% error
        base_error_rate = 0.05 + (cci_score * 0.45)
        
        # Add temperature effect
        error_rate = min(0.95, base_error_rate * (1 + temperature))
        
        # Decide: correct or random
        if self.rng.rand() < (1 - error_rate):
            choice = ground_truth
        else:
            # Sample from non-ground-truth options
            others = [i for i in range(n_options) if i != ground_truth]
            choice = self.rng.choice(others)
        
        # Generate confidence (inversely related to CCI)
        confidence = max(0.1, 1.0 - cci_score * 0.8 + self.rng.normal(0, 0.1))
        confidence = np.clip(confidence, 0.1, 1.0)
        
        return {
            'choice': choice,
            'confidence': confidence,
            'reasoning': f"Selected option {choice} based on attributes."
        }
    
    def generate_samples(
        self,
        options: List[Dict],
        cci_score: float,
        ground_truth: int,
        n_samples: int = 10,
        temperature: float = 0.7
    ) -> List[Dict]:
        """Generate multiple decision samples."""
        return [
            self.make_decision(options, cci_score, ground_truth, temperature)
            for _ in range(n_samples)
        ]


def compute_ildc(choices: List[Dict]) -> Tuple[float, Dict]:
    """Compute ILDC from choice samples."""
    choice_ids = [c['choice'] for c in choices]
    confidences = [c['confidence'] for c in choices]
    
    # Volatility: fraction of times choice changes
    most_common_choice = max(set(choice_ids), key=choice_ids.count)
    volatility = 1.0 - (choice_ids.count(most_common_choice) / len(choice_ids))
    
    # Confidence metrics
    mean_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)
    
    # Disagreement: pairwise different choices
    n_unique = len(set(choice_ids))
    disagreement = n_unique / len(choice_ids)
    
    # ILDC score: weighted combination
    ildc_score = (
        0.4 * volatility +
        0.3 * (1 - mean_confidence) +
        0.2 * std_confidence +
        0.1 * disagreement
    )
    
    features = {
        'volatility': volatility,
        'mean_confidence': mean_confidence,
        'std_confidence': std_confidence,
        'n_unique_choices': n_unique,
        'disagreement': disagreement
    }
    
    return ildc_score, features


def apply_controller(
    options: List[Dict],
    cci_score: float,
    ildc_score: float,
    strategy: str
) -> Tuple[List[Dict], str, Dict]:
    """Apply controller strategy."""
    n_original = len(options)
    
    if strategy == 'none':
        return options, 'none', {'n_pruned': 0}
    
    elif strategy == 'naive_topk':
        # Always show top 5
        k = min(5, len(options))
        return options[:k], 'prune_topk', {'n_pruned': n_original - k}
    
    elif strategy == 'cci_only':
        # Prune if CCI > 0.6
        if cci_score > 0.6:
            k = max(3, int(len(options) * 0.3))
            return options[:k], 'prune_cci', {'n_pruned': n_original - k}
        return options, 'none', {'n_pruned': 0}
    
    elif strategy == 'ildc_only':
        # Not applicable without first seeing ILDC (chicken-egg problem)
        # In practice, would need iterative approach
        return options, 'none', {'n_pruned': 0}
    
    elif strategy == 'two_tier':
        # Combined threshold
        complexity_score = 0.6 * cci_score + 0.4 * ildc_score
        if complexity_score > 0.5:
            # Aggressive pruning for high complexity
            k = max(3, int(len(options) * (1 - complexity_score)))
            return options[:k], 'prune_two_tier', {'n_pruned': n_original - k}
        return options, 'none', {'n_pruned': 0}
    
    return options, 'unknown', {'n_pruned': 0}


def run_demo(args):
    """Run complete demonstration."""
    set_seed(args.seed)
    
    print("="*80)
    print("Decision-Theoretic Choice Complexity in LLMs - Demonstration")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.n_samples}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output}")
    print()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("[1/6] Initializing components...")
    dataset_gen = SyntheticChoiceDataset(seed=args.seed)
    cci_calculator = ChoiceComplexityIndex()
    llm_sim = SyntheticLLMSimulator(seed=args.seed)
    
    # Generate dataset
    print(f"[2/6] Generating {args.n_samples} synthetic choice problems...")
    samples = dataset_gen.generate_dataset(n_samples=args.n_samples)
    print(f"  Generated {len(samples)} samples with complexity range [0.0, 1.0]")
    
    # Run experiments
    print("[3/6] Running experiments with all controller strategies...")
    strategies = ['none', 'naive_topk', 'cci_only', 'two_tier']
    
    all_results = []
    
    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        
        for sample in tqdm(samples, desc=f"  Processing"):
            # Compute CCI
            cci_result = cci_calculator.compute(sample['options'])
            cci_score = cci_result['cci_score']
            
            # Generate initial decisions for ILDC
            initial_choices = llm_sim.generate_samples(
                sample['options'],
                cci_score,
                sample['ground_truth_choice'],
                n_samples=10
            )
            
            # Compute ILDC
            ildc_score, ildc_features = compute_ildc(initial_choices)
            
            # Apply controller
            controlled_options, action, controller_info = apply_controller(
                sample['options'],
                cci_score,
                ildc_score,
                strategy
            )
            
            # Make final decision on controlled set
            final_choices = llm_sim.generate_samples(
                controlled_options,
                cci_score,
                sample['ground_truth_choice'],
                n_samples=10
            )
            
            # Map choice back to original if needed
            primary_choice = final_choices[0]['choice']
            if len(controlled_options) < len(sample['options']):
                # Simplified mapping (assumes sorted by quality)
                mapped_choice = primary_choice
            else:
                mapped_choice = primary_choice
            
            # Compute accuracy
            accuracy = 1.0 if mapped_choice == sample['ground_truth_choice'] else 0.0
            
            # Volatility of final choices
            final_ildc, final_ildc_features = compute_ildc(final_choices)
            
            # Store result
            result = {
                'sample_id': sample['id'],
                'n_options': len(sample['options']),
                'n_options_shown': len(controlled_options),
                'controller_strategy': strategy,
                'controller_action': action,
                'cci_score': cci_score,
                'ildc_score': ildc_score,
                'ildc_score_final': final_ildc,
                'accuracy': accuracy,
                'volatility': ildc_features['volatility'],
                'volatility_final': final_ildc_features['volatility'],
                'mean_confidence': ildc_features['mean_confidence'],
                'n_unique_choices': ildc_features['n_unique_choices'],
                **cci_result['features']
            }
            
            all_results.append(result)
    
    # Create DataFrame
    print("\n[4/6] Computing summary statistics...")
    results_df = pd.DataFrame(all_results)
    
    # Summary by strategy
    summary = results_df.groupby('controller_strategy').agg({
        'accuracy': ['mean', 'std'],
        'volatility_final': ['mean', 'std'],
        'cci_score': 'mean',
        'ildc_score': 'mean',
        'n_options_shown': 'mean'
    }).round(4)
    
    print("\nSummary by Controller Strategy:")
    print(summary)
    
    # Correlations
    print("\nKey Correlations:")
    print(f"  CCI vs ILDC: {results_df['cci_score'].corr(results_df['ildc_score']):.3f}")
    print(f"  CCI vs Accuracy: {results_df['cci_score'].corr(results_df['accuracy']):.3f}")
    print(f"  ILDC vs Volatility: {results_df['ildc_score'].corr(results_df['volatility']):.3f}")
    print(f"  CCI vs Volatility: {results_df['cci_score'].corr(results_df['volatility_final']):.3f}")
    
    # Save results
    print(f"\n[5/6] Saving results to {output_dir}...")
    results_df.to_csv(output_dir / 'demo_results.csv', index=False)
    
    summary_dict = {
        'summary_by_strategy': summary.to_dict(),
        'correlations': {
            'cci_vs_ildc': float(results_df['cci_score'].corr(results_df['ildc_score'])),
            'cci_vs_accuracy': float(results_df['cci_score'].corr(results_df['accuracy'])),
            'ildc_vs_volatility': float(results_df['ildc_score'].corr(results_df['volatility'])),
            'cci_vs_volatility': float(results_df['cci_score'].corr(results_df['volatility_final']))
        },
        'timestamp': datetime.now().isoformat(),
        'config': vars(args)
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    # Generate plots
    print("[6/6] Generating visualizations...")
    generate_plots(results_df, plots_dir)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")
    print("\nKey Files:")
    print(f"  - demo_results.csv: Full results data")
    print(f"  - summary.json: Aggregate metrics")
    print(f"  - plots/: All visualizations")
    print("\nNext Steps:")
    print("  1. Review plots in plots/ directory")
    print("  2. Analyze demo_results.csv for detailed patterns")
    print("  3. Run with different seeds to verify robustness")
    print("  4. Integrate with real LLM via llm_adapter.py")
    print()


def generate_plots(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive visualizations."""
    
    # 1. CCI vs ILDC scatter
    plt.figure(figsize=(10, 6))
    for strategy in df['controller_strategy'].unique():
        strategy_df = df[df['controller_strategy'] == strategy]
        plt.scatter(
            strategy_df['cci_score'],
            strategy_df['ildc_score'],
            label=strategy,
            alpha=0.6,
            s=50
        )
    plt.xlabel('CCI (Choice Complexity Index)', fontsize=12)
    plt.ylabel('ILDC (Internal LLM Decision Complexity)', fontsize=12)
    plt.title('External vs Internal Complexity', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cci_vs_ildc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. CCI vs Accuracy
    plt.figure(figsize=(10, 6))
    for strategy in df['controller_strategy'].unique():
        strategy_df = df[df['controller_strategy'] == strategy]
        plt.scatter(
            strategy_df['cci_score'],
            strategy_df['accuracy'],
            label=strategy,
            alpha=0.6,
            s=50
        )
    plt.xlabel('CCI (Choice Complexity Index)', fontsize=12)
    plt.ylabel('Decision Accuracy', fontsize=12)
    plt.title('Choice Complexity vs Decision Quality', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cci_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Controller effects
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy by strategy
    strategy_means = df.groupby('controller_strategy')['accuracy'].mean().sort_values()
    axes[0].barh(range(len(strategy_means)), strategy_means.values, color='steelblue')
    axes[0].set_yticks(range(len(strategy_means)))
    axes[0].set_yticklabels(strategy_means.index)
    axes[0].set_xlabel('Mean Accuracy', fontsize=11)
    axes[0].set_title('(A) Decision Accuracy by Strategy', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Volatility by strategy
    strategy_vol = df.groupby('controller_strategy')['volatility_final'].mean().sort_values()
    axes[1].barh(range(len(strategy_vol)), strategy_vol.values, color='coral')
    axes[1].set_yticks(range(len(strategy_vol)))
    axes[1].set_yticklabels(strategy_vol.index)
    axes[1].set_xlabel('Mean Volatility', fontsize=11)
    axes[1].set_title('(B) Decision Volatility by Strategy', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Options shown
    strategy_opts = df.groupby('controller_strategy')['n_options_shown'].mean().sort_values()
    axes[2].barh(range(len(strategy_opts)), strategy_opts.values, color='mediumseagreen')
    axes[2].set_yticks(range(len(strategy_opts)))
    axes[2].set_yticklabels(strategy_opts.index)
    axes[2].set_xlabel('Mean Options Shown', fontsize=11)
    axes[2].set_title('(C) Cognitive Load (Options Presented)', fontsize=12, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'controller_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Complexity distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['cci_score'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('CCI Score', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution of Choice Complexity Index', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(df['ildc_score'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    plt.xlabel('ILDC Score', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution of Internal Decision Complexity', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_vars = ['cci_score', 'ildc_score', 'accuracy', 'volatility_final', 
                 'n_options', 'n_options_shown', 'mean_confidence']
    corr_matrix = df[corr_vars].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Key Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Generated 5 visualizations in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Demonstration of Decision-Theoretic Choice Complexity framework'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of choice problems to generate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/demo',
        help='Output directory'
    )
    
    args = parser.parse_args()
    run_demo(args)


if __name__ == '__main__':
    main()
