"""plotting.py

Visualization module for Decision-Theoretic Choice Complexity experiments.
Generates publication-quality plots for analysis and reporting.

Usage:
    python experiments/plotting.py --input results/run_20260121_120000
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.3)
plt.rcParams['figure.dpi'] = 150


class ExperimentPlotter:
    """Generate comprehensive visualizations from experiment results."""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Path):
        self.df = results_df
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_all(self):
        """Generate all standard plots."""
        logger.info("Generating all plots...")
        
        self.plot_cci_vs_ildc()
        self.plot_complexity_vs_metrics()
        self.plot_controller_comparison()
        self.plot_volatility_analysis()
        self.plot_correlation_matrix()
        self.plot_complexity_distributions()
        
        if 'accuracy' in self.df.columns and self.df['accuracy'].notna().any():
            self.plot_accuracy_by_complexity()
        
        logger.info(f"All plots saved to {self.output_dir}")
    
    def plot_cci_vs_ildc(self):
        """Scatter plot of CCI vs ILDC with controller strategies."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for strategy in self.df['controller_strategy'].unique():
            subset = self.df[self.df['controller_strategy'] == strategy]
            ax.scatter(
                subset['cci_score'],
                subset['ildc_score'],
                label=strategy,
                alpha=0.6,
                s=50
            )
        
        # Add regression line for 'none' strategy
        none_df = self.df[self.df['controller_strategy'] == 'none']
        if len(none_df) > 0:
            z = np.polyfit(none_df['cci_score'], none_df['ildc_score'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(none_df['cci_score'].min(), none_df['cci_score'].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.5, label='Trend (no controller)')
        
        ax.set_xlabel('External Choice Complexity (CCI)')
        ax.set_ylabel('Internal Decision Complexity (ILDC)')
        ax.set_title('CCI vs ILDC: Two-Tier Complexity Architecture')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cci_vs_ildc.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: cci_vs_ildc.png")
    
    def plot_complexity_vs_metrics(self):
        """Plot CCI/ILDC vs key outcome metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # CCI vs Volatility
        for strategy in self.df['controller_strategy'].unique():
            subset = self.df[self.df['controller_strategy'] == strategy]
            axes[0, 0].scatter(subset['cci_score'], subset['volatility'], label=strategy, alpha=0.6)
        axes[0, 0].set_xlabel('CCI Score')
        axes[0, 0].set_ylabel('Choice Volatility')
        axes[0, 0].set_title('CCI vs Volatility')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ILDC vs Volatility
        for strategy in self.df['controller_strategy'].unique():
            subset = self.df[self.df['controller_strategy'] == strategy]
            axes[0, 1].scatter(subset['ildc_score'], subset['volatility'], label=strategy, alpha=0.6)
        axes[0, 1].set_xlabel('ILDC Score')
        axes[0, 1].set_ylabel('Choice Volatility')
        axes[0, 1].set_title('ILDC vs Volatility')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # CCI vs Token Usage
        for strategy in self.df['controller_strategy'].unique():
            subset = self.df[self.df['controller_strategy'] == strategy]
            axes[1, 0].scatter(subset['cci_score'], subset['total_tokens'], label=strategy, alpha=0.6)
        axes[1, 0].set_xlabel('CCI Score')
        axes[1, 0].set_ylabel('Total Tokens Used')
        axes[1, 0].set_title('CCI vs Computational Cost')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of Options vs ILDC
        for strategy in self.df['controller_strategy'].unique():
            subset = self.df[self.df['controller_strategy'] == strategy]
            axes[1, 1].scatter(subset['n_options'], subset['ildc_score'], label=strategy, alpha=0.6)
        axes[1, 1].set_xlabel('Number of Options')
        axes[1, 1].set_ylabel('ILDC Score')
        axes[1, 1].set_title('Option Count vs Internal Complexity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_vs_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: complexity_vs_metrics.png")
    
    def plot_controller_comparison(self):
        """Bar plot comparing controller strategies."""
        metrics = ['volatility', 'total_tokens', 'ildc_score', 'n_options_shown']
        strategy_means = self.df.groupby('controller_strategy')[metrics].mean()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            strategy_means[metric].plot(kind='bar', ax=axes[i], color='steelblue')
            axes[i].set_title(f'Mean {metric.replace("_", " ").title()} by Strategy')
            axes[i].set_xlabel('Controller Strategy')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'controller_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: controller_comparison.png")
    
    def plot_volatility_analysis(self):
        """Detailed volatility analysis across complexity levels."""
        # Bin CCI into low/medium/high
        self.df['cci_level'] = pd.cut(
            self.df['cci_score'],
            bins=3,
            labels=['Low', 'Medium', 'High']
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Volatility by CCI level and strategy
        sns.boxplot(
            data=self.df,
            x='cci_level',
            y='volatility',
            hue='controller_strategy',
            ax=axes[0]
        )
        axes[0].set_xlabel('CCI Level')
        axes[0].set_ylabel('Choice Volatility')
        axes[0].set_title('Volatility Across Complexity Levels')
        axes[0].legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Volatility reduction vs baseline
        baseline = self.df[self.df['controller_strategy'] == 'none'].groupby('cci_level')['volatility'].mean()
        reductions = {}
        for strategy in self.df['controller_strategy'].unique():
            if strategy != 'none':
                strategy_vol = self.df[self.df['controller_strategy'] == strategy].groupby('cci_level')['volatility'].mean()
                reductions[strategy] = ((baseline - strategy_vol) / baseline * 100).values
        
        reduction_df = pd.DataFrame(reductions, index=['Low', 'Medium', 'High'])
        reduction_df.plot(kind='bar', ax=axes[1])
        axes[1].set_xlabel('CCI Level')
        axes[1].set_ylabel('Volatility Reduction (%)')
        axes[1].set_title('Volatility Reduction vs Baseline')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
        axes[1].legend(title='Strategy')
        axes[1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: volatility_analysis.png")
    
    def plot_correlation_matrix(self):
        """Correlation heatmap of key variables."""
        corr_vars = [
            'cci_score', 'ildc_score', 'volatility',
            'n_options', 'total_tokens', 'avg_latency'
        ]
        
        if 'accuracy' in self.df.columns:
            corr_vars.append('accuracy')
        
        corr_matrix = self.df[corr_vars].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        ax.set_title('Correlation Matrix: Key Metrics')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: correlation_matrix.png")
    
    def plot_complexity_distributions(self):
        """Distribution plots for CCI and ILDC."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # CCI distribution by strategy
        for strategy in self.df['controller_strategy'].unique():
            subset = self.df[self.df['controller_strategy'] == strategy]
            axes[0].hist(subset['cci_score'], bins=20, alpha=0.5, label=strategy)
        axes[0].set_xlabel('CCI Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of External Choice Complexity (CCI)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ILDC distribution by strategy
        for strategy in self.df['controller_strategy'].unique():
            subset = self.df[self.df['controller_strategy'] == strategy]
            axes[1].hist(subset['ildc_score'], bins=20, alpha=0.5, label=strategy)
        axes[1].set_xlabel('ILDC Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Internal Decision Complexity (ILDC)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: complexity_distributions.png")
    
    def plot_accuracy_by_complexity(self):
        """Accuracy analysis across complexity levels (if ground truth available)."""
        self.df['cci_level'] = pd.cut(
            self.df['cci_score'],
            bins=3,
            labels=['Low', 'Medium', 'High']
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=self.df,
            x='cci_level',
            y='accuracy',
            hue='controller_strategy',
            ax=ax
        )
        ax.set_xlabel('CCI Level')
        ax.set_ylabel('Accuracy')
        ax.set_title('Decision Accuracy Across Complexity Levels')
        ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_by_complexity.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: accuracy_by_complexity.png")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from experiment results')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with results.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for plots (default: INPUT/plots)'
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / 'plots'
    
    # Load results
    results_path = input_dir / 'results.csv'
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return
    
    logger.info(f"Loading results from {results_path}")
    df = pd.read_csv(results_path)
    
    # Generate plots
    plotter = ExperimentPlotter(df, output_dir)
    plotter.plot_all()
    
    logger.info(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
