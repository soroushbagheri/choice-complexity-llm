"""Choice Complexity Index (CCI) - Tier A: External Choice Complexity.

Measures the complexity of a choice set based on:
- Number of options
- Redundancy and similarity
- Attribute conflicts and trade-offs
- Dominance structure (Pareto front)
- Option-set entropy and dispersion
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

from .utils import (
    compute_similarity_matrix,
    detect_redundancy,
    compute_pareto_front,
    compute_attribute_conflict,
    compute_option_entropy,
    safe_divide,
    normalize_score
)

logger = logging.getLogger(__name__)


@dataclass
class CCIFeatures:
    """Features contributing to Choice Complexity Index."""
    
    # Basic features
    n_options: int
    n_attributes: int
    
    # Redundancy features
    redundancy_ratio: float  # [0, 1]
    avg_similarity: float  # [0, 1]
    
    # Conflict features
    attribute_conflict: float  # [0, 1]
    
    # Dominance features
    pareto_front_size: int
    pareto_ratio: float  # pareto_size / n_options
    
    # Diversity features
    option_entropy: float  # [0, 1]
    
    # Normalized complexity score
    cci_score: float  # [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_options': self.n_options,
            'n_attributes': self.n_attributes,
            'redundancy_ratio': self.redundancy_ratio,
            'avg_similarity': self.avg_similarity,
            'attribute_conflict': self.attribute_conflict,
            'pareto_front_size': self.pareto_front_size,
            'pareto_ratio': self.pareto_ratio,
            'option_entropy': self.option_entropy,
            'cci_score': self.cci_score
        }


class ChoiceComplexityIndex:
    """Computes external choice complexity from option sets.
    
    The CCI is a weighted combination of multiple complexity indicators:
    - Option count (more options = higher complexity)
    - Redundancy (more duplicates = paradox of choice)
    - Attribute conflicts (trade-offs increase difficulty)
    - Dominance structure (fewer Pareto-optimal = easier)
    - Entropy (dispersion increases complexity)
    
    Args:
        weights: Dictionary of feature weights (must sum to 1.0)
        redundancy_threshold: Similarity threshold for redundancy detection
        normalize: Whether to normalize final score to [0, 1]
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        redundancy_threshold: float = 0.8,
        normalize: bool = True
    ):
        # Default weights (sum to 1.0)
        if weights is None:
            self.weights = {
                'n_options': 0.30,
                'redundancy': 0.25,
                'conflict': 0.20,
                'pareto': 0.15,
                'entropy': 0.10
            }
        else:
            self.weights = weights
            
        # Validate weights
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        self.redundancy_threshold = redundancy_threshold
        self.normalize = normalize
        
        logger.info(f"Initialized CCI with weights: {self.weights}")
    
    def compute(
        self,
        options: List[Dict[str, Any]],
        attribute_keys: Optional[List[str]] = None,
        objective_keys: Optional[List[str]] = None
    ) -> Tuple[float, CCIFeatures]:
        """Compute Choice Complexity Index for an option set.
        
        Args:
            options: List of option dictionaries
            attribute_keys: Keys to use for similarity/conflict analysis
                          (if None, auto-detect numeric keys)
            objective_keys: Keys for Pareto dominance analysis
                          (if None, use attribute_keys)
        
        Returns:
            (cci_score, cci_features)
        """
        if len(options) == 0:
            raise ValueError("Option set cannot be empty")
        
        # Auto-detect attribute keys if not provided
        if attribute_keys is None:
            attribute_keys = [k for k in options[0].keys() 
                            if isinstance(options[0][k], (int, float))]
        
        if objective_keys is None:
            objective_keys = attribute_keys
        
        n_options = len(options)
        n_attributes = len(attribute_keys)
        
        logger.debug(f"Computing CCI for {n_options} options with {n_attributes} attributes")
        
        # 1. Compute similarity and redundancy
        similarity_matrix = compute_similarity_matrix(options, attribute_keys)
        redundancy_ratio, redundant_pairs = detect_redundancy(
            similarity_matrix, 
            threshold=self.redundancy_threshold
        )
        
        # Average pairwise similarity (excluding diagonal)
        mask = ~np.eye(n_options, dtype=bool)
        avg_similarity = similarity_matrix[mask].mean() if n_options > 1 else 0.0
        
        # 2. Compute attribute conflict
        attribute_conflict = compute_attribute_conflict(options, attribute_keys)
        
        # 3. Compute Pareto front
        pareto_indices, pareto_size = compute_pareto_front(
            options, 
            objective_keys,
            maximize=[True] * len(objective_keys)  # Assume all maximize
        )
        pareto_ratio = safe_divide(pareto_size, n_options, default=1.0)
        
        # 4. Compute option entropy
        option_entropy = compute_option_entropy(options, attribute_keys)
        
        # 5. Normalize individual components
        # Option count: logarithmic scaling
        n_options_norm = np.log1p(n_options) / np.log1p(100)  # Scale relative to 100 options
        n_options_norm = min(n_options_norm, 1.0)
        
        # Redundancy: already in [0, 1]
        redundancy_norm = redundancy_ratio
        
        # Conflict: already in [0, 1]
        conflict_norm = attribute_conflict
        
        # Pareto: invert (larger front = lower complexity)
        pareto_complexity = 1.0 - pareto_ratio
        
        # Entropy: already in [0, 1]
        entropy_norm = option_entropy
        
        # 6. Compute weighted CCI score
        cci_score = (
            self.weights['n_options'] * n_options_norm +
            self.weights['redundancy'] * redundancy_norm +
            self.weights['conflict'] * conflict_norm +
            self.weights['pareto'] * pareto_complexity +
            self.weights['entropy'] * entropy_norm
        )
        
        if self.normalize:
            cci_score = normalize_score(cci_score, 0.0, 1.0)
        
        # Create features object
        features = CCIFeatures(
            n_options=n_options,
            n_attributes=n_attributes,
            redundancy_ratio=float(redundancy_ratio),
            avg_similarity=float(avg_similarity),
            attribute_conflict=float(attribute_conflict),
            pareto_front_size=pareto_size,
            pareto_ratio=float(pareto_ratio),
            option_entropy=float(option_entropy),
            cci_score=float(cci_score)
        )
        
        logger.debug(f"CCI computed: {cci_score:.3f}")
        
        return cci_score, features
    
    def batch_compute(
        self,
        option_sets: List[List[Dict[str, Any]]],
        **kwargs
    ) -> Tuple[np.ndarray, List[CCIFeatures]]:
        """Compute CCI for multiple option sets.
        
        Args:
            option_sets: List of option set lists
            **kwargs: Additional arguments for compute()
        
        Returns:
            (cci_scores, cci_features_list)
        """
        scores = []
        features_list = []
        
        for i, options in enumerate(option_sets):
            try:
                score, features = self.compute(options, **kwargs)
                scores.append(score)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error computing CCI for option set {i}: {e}")
                scores.append(np.nan)
                features_list.append(None)
        
        return np.array(scores), features_list
    
    def get_complexity_level(self, cci_score: float) -> str:
        """Categorize complexity level.
        
        Args:
            cci_score: CCI score in [0, 1]
        
        Returns:
            Complexity level: 'low', 'medium', 'high', 'very_high'
        """
        if cci_score < 0.25:
            return 'low'
        elif cci_score < 0.5:
            return 'medium'
        elif cci_score < 0.75:
            return 'high'
        else:
            return 'very_high'
