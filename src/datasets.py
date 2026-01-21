"""Dataset generators for choice complexity experiments.

Provides:
- SyntheticChoiceDataset: Controlled complexity with ground truth
- ConsumerChoiceDataset: Product-like scenarios
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Literal
from dataclasses import dataclass
import logging
import itertools

logger = logging.getLogger(__name__)


@dataclass
class ChoiceProblem:
    """A single choice problem with options and metadata."""
    
    problem_id: int
    options: List[Dict[str, Any]]
    ground_truth: Optional[int]  # Index of correct/best option
    decision_rule: Optional[str]  # Description of decision rule
    complexity_params: Dict[str, Any]  # Parameters used to generate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'problem_id': self.problem_id,
            'n_options': len(self.options),
            'ground_truth': self.ground_truth,
            'decision_rule': self.decision_rule,
            **self.complexity_params
        }


class SyntheticChoiceDataset:
    """Generate synthetic choice problems with controlled complexity.
    
    Creates option sets with manipulated:
    - Number of options
    - Redundancy ratio (near-duplicates)
    - Attribute conflict (trade-offs)
    - Pareto front size
    - Decoy options (attraction effect)
    
    Args:
        n_options_range: Range or list of option counts
        redundancy_ratios: List of redundancy ratios to test
        conflict_levels: List of conflict levels
        n_attributes: Number of attributes per option
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_options_range: List[int] = None,
        redundancy_ratios: List[float] = None,
        conflict_levels: List[str] = None,
        n_attributes: int = 5,
        seed: int = 42
    ):
        if n_options_range is None:
            n_options_range = [3, 5, 10, 20, 50]
        
        if redundancy_ratios is None:
            redundancy_ratios = [0.0, 0.3, 0.6]
        
        if conflict_levels is None:
            conflict_levels = ['low', 'medium', 'high']
        
        self.n_options_range = n_options_range
        self.redundancy_ratios = redundancy_ratios
        self.conflict_levels = conflict_levels
        self.n_attributes = n_attributes
        self.seed = seed
        
        self.rng = np.random.RandomState(seed)
        
        logger.info(f"Initialized SyntheticChoiceDataset with seed={seed}")
    
    def generate(
        self,
        n_problems_per_config: int = 10
    ) -> Tuple[List[ChoiceProblem], pd.DataFrame]:
        """Generate full dataset.
        
        Args:
            n_problems_per_config: Number of problems per configuration
        
        Returns:
            (problems_list, metadata_df)
        """
        problems = []
        problem_id = 0
        
        # Generate problems for all configurations
        configs = itertools.product(
            self.n_options_range,
            self.redundancy_ratios,
            self.conflict_levels
        )
        
        for n_opts, redundancy, conflict in configs:
            for rep in range(n_problems_per_config):
                # Generate problem
                options, ground_truth = self._generate_options(
                    n_options=n_opts,
                    redundancy_ratio=redundancy,
                    conflict_level=conflict
                )
                
                decision_rule = self._get_decision_rule()
                
                problem = ChoiceProblem(
                    problem_id=problem_id,
                    options=options,
                    ground_truth=ground_truth,
                    decision_rule=decision_rule,
                    complexity_params={
                        'n_options': n_opts,
                        'redundancy_ratio': redundancy,
                        'conflict_level': conflict
                    }
                )
                
                problems.append(problem)
                problem_id += 1
        
        # Create metadata DataFrame
        metadata = pd.DataFrame([p.to_dict() for p in problems])
        
        logger.info(f"Generated {len(problems)} choice problems")
        
        return problems, metadata
    
    def _generate_options(
        self,
        n_options: int,
        redundancy_ratio: float,
        conflict_level: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Generate option set with specified complexity.
        
        Returns:
            (options, ground_truth_index)
        """
        options = []
        
        # Determine number of unique vs redundant options
        n_redundant = int(n_options * redundancy_ratio)
        n_unique = n_options - n_redundant
        
        # Generate unique options
        unique_options = self._generate_unique_options(n_unique, conflict_level)
        options.extend(unique_options)
        
        # Add redundant (near-duplicate) options
        if n_redundant > 0:
            redundant_options = self._generate_redundant_options(
                unique_options, 
                n_redundant
            )
            options.extend(redundant_options)
        
        # Shuffle options
        self.rng.shuffle(options)
        
        # Ground truth: best option according to simple rule
        # (e.g., highest sum of normalized attributes)
        ground_truth = self._determine_ground_truth(options)
        
        return options, ground_truth
    
    def _generate_unique_options(
        self,
        n_options: int,
        conflict_level: str
    ) -> List[Dict[str, Any]]:
        """Generate diverse unique options."""
        options = []
        
        for i in range(n_options):
            option = {'id': i}
            
            if conflict_level == 'low':
                # Aligned attributes (positive correlation)
                base_quality = self.rng.uniform(0, 1)
                for j in range(self.n_attributes):
                    # Add noise around base quality
                    option[f'attr_{j}'] = np.clip(
                        base_quality + self.rng.normal(0, 0.1),
                        0, 1
                    )
            
            elif conflict_level == 'medium':
                # Independent attributes
                for j in range(self.n_attributes):
                    option[f'attr_{j}'] = self.rng.uniform(0, 1)
            
            else:  # high conflict
                # Trade-offs (negative correlation)
                for j in range(self.n_attributes):
                    if j % 2 == 0:
                        option[f'attr_{j}'] = self.rng.uniform(0.7, 1.0)
                    else:
                        option[f'attr_{j}'] = self.rng.uniform(0.0, 0.3)
            
            options.append(option)
        
        return options
    
    def _generate_redundant_options(
        self,
        base_options: List[Dict[str, Any]],
        n_redundant: int
    ) -> List[Dict[str, Any]]:
        """Generate near-duplicates of existing options."""
        redundant = []
        
        for i in range(n_redundant):
            # Pick random base option
            base = self.rng.choice(base_options)
            
            # Create near-duplicate with small noise
            duplicate = {'id': f"dup_{i}"}
            
            for key, value in base.items():
                if isinstance(value, (int, float)) and key != 'id':
                    # Add small noise (±5%)
                    noise = self.rng.normal(0, 0.05)
                    duplicate[key] = np.clip(value + noise, 0, 1)
                else:
                    duplicate[key] = value
            
            redundant.append(duplicate)
        
        return redundant
    
    def _determine_ground_truth(self, options: List[Dict[str, Any]]) -> int:
        """Determine best option by simple rule.
        
        Ground truth = option with highest average attribute value.
        """
        scores = []
        
        for opt in options:
            attr_values = [v for k, v in opt.items() 
                          if k.startswith('attr_') and isinstance(v, (int, float))]
            scores.append(np.mean(attr_values) if attr_values else 0)
        
        return int(np.argmax(scores))
    
    def _get_decision_rule(self) -> str:
        """Get description of decision rule."""
        return "maximize: average of all attributes"


class ConsumerChoiceDataset:
    """Generate consumer-choice-like problems (e.g., product selection).
    
    Creates product-like options with realistic attributes:
    - Price (lower is better)
    - Rating (higher is better)
    - Shipping time (lower is better)
    - Brand reputation (higher is better)
    - Warranty (higher is better)
    
    Args:
        n_problems: Number of problems to generate
        n_options_range: Range of option counts
        user_profiles: Types of user preferences
        seed: Random seed
    """
    
    def __init__(
        self,
        n_problems: int = 100,
        n_options_range: Tuple[int, int] = (5, 20),
        user_profiles: List[str] = None,
        seed: int = 42
    ):
        if user_profiles is None:
            user_profiles = ['budget', 'balanced', 'quality']
        
        self.n_problems = n_problems
        self.n_options_range = n_options_range
        self.user_profiles = user_profiles
        self.seed = seed
        
        self.rng = np.random.RandomState(seed)
        
        logger.info(f"Initialized ConsumerChoiceDataset with seed={seed}")
    
    def generate(self) -> Tuple[List[ChoiceProblem], pd.DataFrame]:
        """Generate consumer choice problems.
        
        Returns:
            (problems_list, metadata_df)
        """
        problems = []
        
        for i in range(self.n_problems):
            # Random number of options
            n_opts = self.rng.randint(*self.n_options_range)
            
            # Random user profile
            profile = self.rng.choice(self.user_profiles)
            
            # Generate options
            options = self._generate_products(n_opts)
            
            # Determine ground truth based on profile
            ground_truth = self._determine_best_product(options, profile)
            
            decision_rule = self._get_decision_rule(profile)
            
            problem = ChoiceProblem(
                problem_id=i,
                options=options,
                ground_truth=ground_truth,
                decision_rule=decision_rule,
                complexity_params={
                    'n_options': n_opts,
                    'user_profile': profile
                }
            )
            
            problems.append(problem)
        
        metadata = pd.DataFrame([p.to_dict() for p in problems])
        
        logger.info(f"Generated {len(problems)} consumer choice problems")
        
        return problems, metadata
    
    def _generate_products(self, n_products: int) -> List[Dict[str, Any]]:
        """Generate product options."""
        products = []
        
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
        
        for i in range(n_products):
            product = {
                'id': i,
                'name': f'Product_{i}',
                'price': self.rng.uniform(20, 200),  # $20-200
                'rating': self.rng.uniform(3.0, 5.0),  # 3.0-5.0 stars
                'shipping_days': self.rng.randint(1, 15),  # 1-14 days
                'brand': self.rng.choice(brands),
                'warranty_months': self.rng.choice([0, 6, 12, 24, 36]),
                'brand_reputation': self.rng.uniform(0.5, 1.0)  # 0.5-1.0
            }
            
            products.append(product)
        
        return products
    
    def _determine_best_product(
        self, 
        products: List[Dict[str, Any]], 
        profile: str
    ) -> int:
        """Determine best product based on user profile."""
        scores = []
        
        for product in products:
            if profile == 'budget':
                # Minimize price, rating > 4.0
                if product['rating'] >= 4.0:
                    score = 1.0 / product['price']  # Lower price = higher score
                else:
                    score = -1  # Disqualified
            
            elif profile == 'balanced':
                # Balance price and quality
                # Normalize and compute weighted sum
                price_norm = 1.0 - (product['price'] - 20) / 180  # Invert: lower is better
                rating_norm = (product['rating'] - 3.0) / 2.0
                score = 0.4 * price_norm + 0.6 * rating_norm
            
            else:  # quality
                # Maximize rating and warranty, price < $150
                if product['price'] <= 150:
                    rating_norm = (product['rating'] - 3.0) / 2.0
                    warranty_norm = product['warranty_months'] / 36
                    score = 0.7 * rating_norm + 0.3 * warranty_norm
                else:
                    score = -1
            
            scores.append(score)
        
        # Return index of best option
        valid_indices = [i for i, s in enumerate(scores) if s >= 0]
        
        if len(valid_indices) == 0:
            return 0  # Fallback
        
        best_idx = valid_indices[np.argmax([scores[i] for i in valid_indices])]
        return best_idx
    
    def _get_decision_rule(self, profile: str) -> str:
        """Get decision rule description."""
        rules = {
            'budget': 'minimize price with rating >= 4.0',
            'balanced': 'balance price (40%) and rating (60%)',
            'quality': 'maximize rating (70%) and warranty (30%) with price <= $150'
        }
        return rules.get(profile, 'unknown')
