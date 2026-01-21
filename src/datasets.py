"""Dataset Generation Module for Choice Complexity Research.

Generates synthetic choice sets with controlled complexity parameters:
- Number of options (n)
- Redundancy ratio (near-duplicates)
- Attribute conflict levels
- Decoy options
- Pareto front structure

Author: Soroush Bagheri
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import hashlib


@dataclass
class Option:
    """Represents a single choice option with attributes."""
    id: str
    name: str
    attributes: Dict[str, float]
    description: Optional[str] = None
    is_decoy: bool = False
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert option to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'attributes': self.attributes,
            'description': self.description,
            'is_decoy': self.is_decoy,
            'is_duplicate': self.is_duplicate,
            'duplicate_of': self.duplicate_of
        }
    
    def __str__(self) -> str:
        """String representation for LLM prompts."""
        attr_str = ", ".join([f"{k}: {v:.2f}" for k, v in self.attributes.items()])
        return f"{self.name} ({attr_str})"


@dataclass
class ChoiceSet:
    """Represents a complete choice set with metadata."""
    id: str
    options: List[Option]
    ground_truth: Optional[str] = None  # ID of optimal choice
    decision_rule: Optional[str] = None  # Description of decision rule
    complexity_params: Dict[str, Any] = field(default_factory=dict)
    task_description: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.options)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert choice set to dictionary."""
        return {
            'id': self.id,
            'options': [opt.to_dict() for opt in self.options],
            'ground_truth': self.ground_truth,
            'decision_rule': self.decision_rule,
            'complexity_params': self.complexity_params,
            'task_description': self.task_description
        }
    
    def to_prompt(self, format_type: str = "numbered_list") -> str:
        """Convert choice set to LLM prompt format."""
        if format_type == "numbered_list":
            prompt_lines = []
            for i, opt in enumerate(self.options, 1):
                prompt_lines.append(f"{i}. {opt}")
            return "\n".join(prompt_lines)
        elif format_type == "markdown_table":
            if not self.options:
                return ""
            # Get all attribute names
            attrs = list(self.options[0].attributes.keys())
            header = "| Option | " + " | ".join(attrs) + " |\n"
            separator = "|" + "---|".join(["" for _ in range(len(attrs) + 2)])
            rows = []
            for opt in self.options:
                row = f"| {opt.name} | " + " | ".join([f"{opt.attributes[a]:.2f}" for a in attrs]) + " |"
                rows.append(row)
            return header + separator + "\n" + "\n".join(rows)
        else:
            return "\n".join([str(opt) for opt in self.options])


class SyntheticChoiceGenerator:
    """Generates synthetic choice sets with controlled complexity."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate_dataset(
        self,
        n_choice_sets: int = 100,
        n_options_range: Tuple[int, int] = (3, 20),
        n_attributes_range: Tuple[int, int] = (2, 5),
        redundancy_ratios: List[float] = [0.0, 0.3, 0.6],
        conflict_levels: List[str] = ["low", "medium", "high"],
        include_decoys: bool = True,
        save_path: Optional[Path] = None
    ) -> List[ChoiceSet]:
        """Generate a complete dataset of choice sets.
        
        Args:
            n_choice_sets: Number of choice sets to generate
            n_options_range: Range for number of options per set
            n_attributes_range: Range for number of attributes per option
            redundancy_ratios: List of redundancy ratios to sample from
            conflict_levels: List of conflict levels ("low", "medium", "high")
            include_decoys: Whether to include decoy options
            save_path: Optional path to save dataset as JSON
            
        Returns:
            List of ChoiceSet objects
        """
        choice_sets = []
        
        for i in range(n_choice_sets):
            # Sample complexity parameters
            n_options = self.rng.randint(*n_options_range)
            n_attributes = self.rng.randint(*n_attributes_range)
            redundancy_ratio = self.rng.choice(redundancy_ratios)
            conflict_level = self.rng.choice(conflict_levels)
            
            # Generate choice set
            choice_set = self.generate_choice_set(
                n_options=n_options,
                n_attributes=n_attributes,
                redundancy_ratio=redundancy_ratio,
                conflict_level=conflict_level,
                include_decoy=include_decoys and self.rng.rand() > 0.5,
                set_id=f"set_{i:04d}"
            )
            
            choice_sets.append(choice_set)
        
        # Save if path provided
        if save_path:
            self._save_dataset(choice_sets, save_path)
        
        return choice_sets
    
    def generate_choice_set(
        self,
        n_options: int,
        n_attributes: int,
        redundancy_ratio: float = 0.0,
        conflict_level: str = "medium",
        include_decoy: bool = False,
        set_id: Optional[str] = None
    ) -> ChoiceSet:
        """Generate a single choice set with specified complexity.
        
        Args:
            n_options: Number of options
            n_attributes: Number of attributes per option
            redundancy_ratio: Fraction of near-duplicate options (0.0-1.0)
            conflict_level: Attribute conflict level ("low", "medium", "high")
            include_decoy: Whether to include a decoy option
            set_id: Unique identifier for this choice set
            
        Returns:
            ChoiceSet object
        """
        if set_id is None:
            set_id = hashlib.md5(str(self.rng.rand()).encode()).hexdigest()[:8]
        
        # Generate attribute names
        attribute_names = [f"attr_{i+1}" for i in range(n_attributes)]
        
        # Determine correlation structure based on conflict level
        correlation = self._get_correlation_matrix(n_attributes, conflict_level)
        
        # Generate base options
        n_unique = int(n_options * (1 - redundancy_ratio))
        n_duplicates = n_options - n_unique
        
        options = []
        
        # Generate unique options
        base_attributes = self._generate_correlated_attributes(
            n_unique, n_attributes, correlation
        )
        
        for i in range(n_unique):
            attrs = {name: float(base_attributes[i, j]) 
                    for j, name in enumerate(attribute_names)}
            
            option = Option(
                id=f"opt_{set_id}_{i:03d}",
                name=f"Option {chr(65+i)}",  # A, B, C, ...
                attributes=attrs,
                description=None,
                is_decoy=False,
                is_duplicate=False
            )
            options.append(option)
        
        # Add near-duplicates
        for i in range(n_duplicates):
            # Pick a random base option to duplicate
            base_idx = self.rng.randint(0, n_unique)
            base_option = options[base_idx]
            
            # Add small noise to attributes
            noisy_attrs = {}
            for name, value in base_option.attributes.items():
                noise = self.rng.normal(0, 0.05)  # 5% std noise
                noisy_attrs[name] = np.clip(value + noise, 0, 1)
            
            duplicate = Option(
                id=f"opt_{set_id}_{n_unique+i:03d}",
                name=f"Option {chr(65+n_unique+i)}",
                attributes=noisy_attrs,
                description=None,
                is_decoy=False,
                is_duplicate=True,
                duplicate_of=base_option.id
            )
            options.append(duplicate)
        
        # Add decoy option if requested
        if include_decoy and n_unique > 0:
            decoy = self._create_decoy_option(options[0], set_id, len(options))
            options.append(decoy)
        
        # Shuffle options
        self.rng.shuffle(options)
        
        # Determine ground truth (highest utility option)
        ground_truth, decision_rule = self._compute_ground_truth(
            options, attribute_names
        )
        
        # Create complexity parameters record
        complexity_params = {
            'n_options': n_options,
            'n_attributes': n_attributes,
            'redundancy_ratio': redundancy_ratio,
            'conflict_level': conflict_level,
            'has_decoy': include_decoy,
            'n_unique': n_unique,
            'n_duplicates': n_duplicates
        }
        
        choice_set = ChoiceSet(
            id=set_id,
            options=options,
            ground_truth=ground_truth,
            decision_rule=decision_rule,
            complexity_params=complexity_params,
            task_description="Select the best option based on the given attributes."
        )
        
        return choice_set
    
    def _get_correlation_matrix(
        self, 
        n_attributes: int, 
        conflict_level: str
    ) -> np.ndarray:
        """Generate correlation matrix for attributes based on conflict level.
        
        Args:
            n_attributes: Number of attributes
            conflict_level: "low", "medium", or "high"
            
        Returns:
            Correlation matrix
        """
        if conflict_level == "low":
            # Highly correlated attributes (easier decisions)
            base_corr = 0.7
        elif conflict_level == "medium":
            # Moderate correlation
            base_corr = 0.3
        else:  # high
            # Negative correlation (conflicting attributes)
            base_corr = -0.5
        
        # Create correlation matrix
        corr = np.eye(n_attributes)
        for i in range(n_attributes):
            for j in range(i+1, n_attributes):
                corr[i, j] = corr[j, i] = base_corr + self.rng.normal(0, 0.1)
        
        # Ensure positive semi-definite
        corr = self._nearest_psd(corr)
        
        return corr
    
    def _nearest_psd(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive semi-definite matrix."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_psd(A3):
            return A3
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_psd(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        
        return A3
    
    def _is_psd(self, A: np.ndarray) -> bool:
        """Check if matrix is positive semi-definite."""
        return np.all(np.linalg.eigvals(A) >= -1e-8)
    
    def _generate_correlated_attributes(
        self,
        n_samples: int,
        n_attributes: int,
        correlation: np.ndarray
    ) -> np.ndarray:
        """Generate correlated attribute values.
        
        Args:
            n_samples: Number of options
            n_attributes: Number of attributes
            correlation: Correlation matrix
            
        Returns:
            Array of shape (n_samples, n_attributes) with values in [0, 1]
        """
        # Generate correlated normal variables
        mean = np.zeros(n_attributes)
        samples = self.rng.multivariate_normal(mean, correlation, size=n_samples)
        
        # Transform to [0, 1] using CDF
        from scipy.stats import norm
        uniform_samples = norm.cdf(samples)
        
        return uniform_samples
    
    def _create_decoy_option(
        self, 
        target: Option, 
        set_id: str, 
        index: int
    ) -> Option:
        """Create a decoy option (asymmetrically dominated).
        
        The decoy is slightly worse than the target on all attributes.
        
        Args:
            target: Target option to create decoy for
            set_id: Choice set ID
            index: Index for option ID
            
        Returns:
            Decoy option
        """
        decoy_attrs = {}
        for name, value in target.attributes.items():
            # Make slightly worse (5-15% lower)
            reduction = self.rng.uniform(0.05, 0.15)
            decoy_attrs[name] = max(0, value * (1 - reduction))
        
        decoy = Option(
            id=f"opt_{set_id}_{index:03d}",
            name=f"Option {chr(65+index)}",
            attributes=decoy_attrs,
            description="Decoy option",
            is_decoy=True,
            is_duplicate=False
        )
        
        return decoy
    
    def _compute_ground_truth(
        self, 
        options: List[Option], 
        attribute_names: List[str]
    ) -> Tuple[str, str]:
        """Compute ground truth optimal choice.
        
        Uses simple additive utility: sum of all attributes.
        
        Args:
            options: List of options
            attribute_names: List of attribute names
            
        Returns:
            Tuple of (ground_truth_id, decision_rule)
        """
        utilities = []
        for opt in options:
            # Equal-weight additive utility
            utility = sum(opt.attributes.values()) / len(opt.attributes)
            utilities.append(utility)
        
        best_idx = np.argmax(utilities)
        ground_truth = options[best_idx].id
        decision_rule = "Maximize average attribute value (equal weights)"
        
        return ground_truth, decision_rule
    
    def _save_dataset(
        self, 
        choice_sets: List[ChoiceSet], 
        save_path: Path
    ):
        """Save dataset to JSON file.
        
        Args:
            choice_sets: List of choice sets
            save_path: Path to save file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset_dict = {
            'metadata': {
                'n_choice_sets': len(choice_sets),
                'generator_seed': self.seed,
                'generation_date': pd.Timestamp.now().isoformat()
            },
            'choice_sets': [cs.to_dict() for cs in choice_sets]
        }
        
        with open(save_path, 'w') as f:
            json.dump(dataset_dict, f, indent=2)
        
        print(f"Dataset saved to {save_path}")


class ConsumerChoiceGenerator:
    """Generates realistic consumer choice scenarios (products with attributes)."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator.
        
        Args:
            seed: Random seed
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Product categories and their typical attributes
        self.categories = {
            'laptop': ['price', 'performance', 'battery_life', 'weight', 'screen_quality'],
            'phone': ['price', 'camera_quality', 'battery_life', 'storage', 'brand_reputation'],
            'hotel': ['price', 'rating', 'location_score', 'amenities', 'cleanliness'],
            'restaurant': ['price', 'food_rating', 'service_rating', 'ambiance', 'distance']
        }
        
        # Brand names for realism
        self.brands = {
            'laptop': ['TechPro', 'CompuMax', 'EliteBook', 'PowerLap', 'SwiftTech'],
            'phone': ['TechPhone', 'SmartX', 'MobilePro', 'PhoneMax', 'ConnectPlus'],
            'hotel': ['Grand Hotel', 'Comfort Inn', 'Luxury Stay', 'Budget Lodge', 'Elite Suites'],
            'restaurant': ['Bella Italia', 'Sushi Master', 'The Steakhouse', 'Vegan Delight', 'Fast Bites']
        }
    
    def generate_product_choice_set(
        self,
        category: str,
        n_options: int = 10,
        user_profile: Optional[Dict[str, float]] = None,
        set_id: Optional[str] = None
    ) -> ChoiceSet:
        """Generate a realistic product choice set.
        
        Args:
            category: Product category (laptop, phone, hotel, restaurant)
            n_options: Number of options
            user_profile: User preference weights for attributes
            set_id: Unique identifier
            
        Returns:
            ChoiceSet object
        """
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}")
        
        if set_id is None:
            set_id = f"{category}_{hashlib.md5(str(self.rng.rand()).encode()).hexdigest()[:8]}"
        
        attributes = self.categories[category]
        brands = self.brands[category]
        
        options = []
        for i in range(n_options):
            # Generate attributes (normalized to [0,1])
            attrs = {}
            for attr in attributes:
                if attr == 'price':
                    # Price: lower is better, but we normalize
                    attrs[attr] = self.rng.uniform(0.2, 1.0)
                else:
                    # Other attributes: higher is better
                    attrs[attr] = self.rng.uniform(0.3, 1.0)
            
            # Pick a brand
            brand = self.rng.choice(brands)
            
            option = Option(
                id=f"prod_{set_id}_{i:03d}",
                name=f"{brand} {category.title()} {i+1}",
                attributes=attrs,
                description=f"{brand} {category}",
                is_decoy=False,
                is_duplicate=False
            )
            options.append(option)
        
        # Determine ground truth based on user profile
        if user_profile is None:
            # Default: equal weights
            user_profile = {attr: 1.0 / len(attributes) for attr in attributes}
        
        ground_truth, decision_rule = self._compute_utility_based_truth(
            options, user_profile
        )
        
        complexity_params = {
            'category': category,
            'n_options': n_options,
            'user_profile': user_profile
        }
        
        choice_set = ChoiceSet(
            id=set_id,
            options=options,
            ground_truth=ground_truth,
            decision_rule=decision_rule,
            complexity_params=complexity_params,
            task_description=f"Select the best {category} based on your preferences."
        )
        
        return choice_set
    
    def _compute_utility_based_truth(
        self,
        options: List[Option],
        user_profile: Dict[str, float]
    ) -> Tuple[str, str]:
        """Compute ground truth using weighted utility.
        
        Args:
            options: List of options
            user_profile: Attribute weights
            
        Returns:
            Tuple of (ground_truth_id, decision_rule)
        """
        utilities = []
        for opt in options:
            utility = sum(opt.attributes.get(attr, 0) * weight 
                         for attr, weight in user_profile.items())
            utilities.append(utility)
        
        best_idx = np.argmax(utilities)
        ground_truth = options[best_idx].id
        
        # Format decision rule
        top_prefs = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)[:2]
        decision_rule = f"Maximize utility (top priorities: {top_prefs[0][0]}, {top_prefs[1][0]})"
        
        return ground_truth, decision_rule


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Choice Complexity Dataset Generator - Test Run")
    print("=" * 80)
    
    # Test synthetic generator
    print("\n1. Generating synthetic dataset...")
    syn_gen = SyntheticChoiceGenerator(seed=42)
    
    # Generate a single example
    example_set = syn_gen.generate_choice_set(
        n_options=10,
        n_attributes=3,
        redundancy_ratio=0.3,
        conflict_level="high",
        include_decoy=True,
        set_id="example_001"
    )
    
    print(f"\nGenerated choice set: {example_set.id}")
    print(f"Number of options: {len(example_set)}")
    print(f"Complexity params: {example_set.complexity_params}")
    print(f"\nGround truth: {example_set.ground_truth}")
    print(f"Decision rule: {example_set.decision_rule}")
    print(f"\nOptions (first 3):")
    for i, opt in enumerate(example_set.options[:3]):
        print(f"  {i+1}. {opt}")
    
    # Generate full dataset
    print("\n2. Generating full synthetic dataset (100 choice sets)...")
    dataset = syn_gen.generate_dataset(
        n_choice_sets=100,
        save_path=Path("data/synthetic_dataset.json")
    )
    print(f"Generated {len(dataset)} choice sets")
    
    # Test consumer choice generator
    print("\n3. Generating consumer choice scenario...")
    consumer_gen = ConsumerChoiceGenerator(seed=42)
    
    user_profile = {
        'price': 0.4,  # Price-sensitive user
        'performance': 0.3,
        'battery_life': 0.2,
        'weight': 0.05,
        'screen_quality': 0.05
    }
    
    laptop_set = consumer_gen.generate_product_choice_set(
        category='laptop',
        n_options=8,
        user_profile=user_profile,
        set_id="laptop_001"
    )
    
    print(f"\nGenerated product choice set: {laptop_set.id}")
    print(f"Number of laptops: {len(laptop_set)}")
    print(f"\nUser profile: {user_profile}")
    print(f"Ground truth (best laptop): {laptop_set.ground_truth}")
    print(f"\nFirst 3 laptops:")
    for i, opt in enumerate(laptop_set.options[:3]):
        print(f"  {i+1}. {opt}")
    
    print("\n" + "=" * 80)
    print("Dataset generation test completed successfully!")
    print("=" * 80)
