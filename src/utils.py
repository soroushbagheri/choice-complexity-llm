"""Utility functions for choice complexity analysis."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a score to [min_val, max_val] range.
    
    Args:
        value: Raw score value
        min_val: Minimum of target range
        max_val: Maximum of target range
        
    Returns:
        Normalized score
    """
    return np.clip(value, min_val, max_val)


def compute_similarity_matrix(
    options: List[Dict[str, Any]], 
    feature_keys: Optional[List[str]] = None
) -> np.ndarray:
    """Compute pairwise similarity matrix for options.
    
    Args:
        options: List of option dictionaries
        feature_keys: Keys to use for similarity (if None, use all numeric)
        
    Returns:
        n x n similarity matrix (1 = identical, 0 = completely different)
    """
    n = len(options)
    similarity_matrix = np.zeros((n, n))
    
    # Extract feature vectors
    if feature_keys is None:
        # Auto-detect numeric keys
        feature_keys = [k for k in options[0].keys() 
                       if isinstance(options[0][k], (int, float))]
    
    vectors = []
    for opt in options:
        vec = [opt.get(k, 0) for k in feature_keys]
        vectors.append(vec)
    
    vectors = np.array(vectors, dtype=float)
    
    # Normalize vectors
    if vectors.std(axis=0).sum() > 0:
        vectors = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-8)
    
    # Compute similarities
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Use 1 - normalized Euclidean distance
                dist = euclidean(vectors[i], vectors[j])
                sim = 1.0 / (1.0 + dist)  # Convert distance to similarity
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    return similarity_matrix


def detect_redundancy(similarity_matrix: np.ndarray, threshold: float = 0.8) -> Tuple[float, List[Tuple[int, int]]]:
    """Detect redundant options based on similarity.
    
    Args:
        similarity_matrix: n x n pairwise similarity matrix
        threshold: Similarity threshold for redundancy
        
    Returns:
        (redundancy_ratio, redundant_pairs)
    """
    n = similarity_matrix.shape[0]
    redundant_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                redundant_pairs.append((i, j))
    
    max_possible_pairs = n * (n - 1) / 2
    redundancy_ratio = len(redundant_pairs) / max_possible_pairs if max_possible_pairs > 0 else 0.0
    
    return redundancy_ratio, redundant_pairs


def compute_pareto_front(
    options: List[Dict[str, Any]], 
    objectives: List[str],
    maximize: Optional[List[bool]] = None
) -> Tuple[List[int], int]:
    """Identify Pareto-optimal options.
    
    Args:
        options: List of option dictionaries
        objectives: Keys for objective values
        maximize: Whether to maximize each objective (default: all True)
        
    Returns:
        (indices_of_pareto_optimal, pareto_front_size)
    """
    n = len(options)
    
    if maximize is None:
        maximize = [True] * len(objectives)
    
    # Extract objective vectors
    vectors = []
    for opt in options:
        vec = [opt.get(obj, 0) for obj in objectives]
        vectors.append(vec)
    vectors = np.array(vectors, dtype=float)
    
    # Flip signs for minimization objectives
    for i, should_max in enumerate(maximize):
        if not should_max:
            vectors[:, i] = -vectors[:, i]
    
    # Find Pareto front
    pareto_mask = np.ones(n, dtype=bool)
    
    for i in range(n):
        if pareto_mask[i]:
            # Check if any other point dominates i
            for j in range(n):
                if i != j and pareto_mask[j]:
                    # j dominates i if j is >= i in all objectives and > in at least one
                    dominates = np.all(vectors[j] >= vectors[i]) and np.any(vectors[j] > vectors[i])
                    if dominates:
                        pareto_mask[i] = False
                        break
    
    pareto_indices = np.where(pareto_mask)[0].tolist()
    
    return pareto_indices, len(pareto_indices)


def compute_attribute_conflict(
    options: List[Dict[str, Any]], 
    attributes: List[str]
) -> float:
    """Compute degree of attribute conflict (trade-offs).
    
    High conflict = strong negative correlations between attributes.
    
    Args:
        options: List of option dictionaries
        attributes: Attribute keys to analyze
        
    Returns:
        Conflict score in [0, 1] (1 = maximum conflict)
    """
    # Extract attribute matrix
    matrix = []
    for opt in options:
        row = [opt.get(attr, 0) for attr in attributes]
        matrix.append(row)
    
    matrix = np.array(matrix, dtype=float)
    
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return 0.0
    
    # Compute correlation matrix
    try:
        corr_matrix = np.corrcoef(matrix.T)
        
        # Count strong negative correlations
        n_attrs = len(attributes)
        negative_corrs = []
        
        for i in range(n_attrs):
            for j in range(i + 1, n_attrs):
                if not np.isnan(corr_matrix[i, j]):
                    if corr_matrix[i, j] < -0.3:  # Threshold for conflict
                        negative_corrs.append(abs(corr_matrix[i, j]))
        
        if len(negative_corrs) == 0:
            return 0.0
        
        # Conflict score = average magnitude of negative correlations
        conflict_score = np.mean(negative_corrs)
        return float(np.clip(conflict_score, 0, 1))
    
    except Exception as e:
        logger.warning(f"Error computing conflict: {e}")
        return 0.0


def compute_option_entropy(
    options: List[Dict[str, Any]], 
    attributes: List[str]
) -> float:
    """Compute entropy of option distribution in attribute space.
    
    Higher entropy = more dispersed/diverse options.
    
    Args:
        options: List of option dictionaries
        attributes: Attribute keys
        
    Returns:
        Normalized entropy score in [0, 1]
    """
    # Extract attribute matrix
    matrix = []
    for opt in options:
        row = [opt.get(attr, 0) for attr in attributes]
        matrix.append(row)
    
    matrix = np.array(matrix, dtype=float)
    
    if matrix.shape[0] < 2:
        return 0.0
    
    # Discretize continuous values into bins
    n_bins = min(10, matrix.shape[0])
    
    entropies = []
    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        if values.std() < 1e-8:  # Constant attribute
            entropies.append(0.0)
        else:
            # Create histogram
            hist, _ = np.histogram(values, bins=n_bins)
            hist = hist + 1  # Add-one smoothing
            probs = hist / hist.sum()
            
            # Compute entropy
            ent = entropy(probs, base=2)
            max_ent = np.log2(n_bins)  # Maximum possible entropy
            normalized_ent = ent / max_ent if max_ent > 0 else 0
            entropies.append(normalized_ent)
    
    # Average entropy across attributes
    return float(np.mean(entropies))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value."""
    return numerator / denominator if denominator != 0 else default


def format_option(option: Dict[str, Any], include_keys: Optional[List[str]] = None) -> str:
    """Format option dictionary as readable string.
    
    Args:
        option: Option dictionary
        include_keys: Keys to include (if None, include all)
        
    Returns:
        Formatted string
    """
    if include_keys is None:
        include_keys = list(option.keys())
    
    parts = []
    for key in include_keys:
        if key in option:
            value = option[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
    
    return ", ".join(parts)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_results(results: pd.DataFrame, output_path: str) -> None:
    """Save results DataFrame to CSV.
    
    Args:
        results: Results DataFrame
        output_path: Output CSV path
    """
    results.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
