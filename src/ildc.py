"""Internal LLM Decision Complexity (ILDC) Module.

This module computes internal decision difficulty metrics from LLM inference signals.
ILDC measures how hard it is for the model to make a decision, using proxies like:
- Choice volatility across repeated samples
- Confidence gaps between top options
- Disagreement in pairwise preferences
- Deliberation indicators (rationale complexity)

Author: Soroush Bagheri
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import warnings
from scipy.stats import entropy


class ILDCComputer:
    """Computes Internal LLM Decision Complexity metrics.
    
    Attributes:
        use_logprobs: Whether to use log probabilities if available
        num_samples: Number of samples for volatility estimation
        confidence_method: Method for confidence estimation ('margin', 'entropy', 'voting')
    """
    
    def __init__(
        self,
        use_logprobs: bool = False,
        num_samples: int = 10,
        confidence_method: str = 'voting'
    ):
        """Initialize ILDC computer.
        
        Args:
            use_logprobs: Use log probabilities for confidence (if available)
            num_samples: Number of samples for volatility measurement
            confidence_method: Method for confidence ('margin', 'entropy', 'voting')
        """
        self.use_logprobs = use_logprobs
        self.num_samples = num_samples
        self.confidence_method = confidence_method
    
    def compute_ildc(
        self,
        choices: List[int],
        logprobs: Optional[List[List[float]]] = None,
        rationales: Optional[List[str]] = None,
        num_options: int = None
    ) -> Dict[str, float]:
        """Compute comprehensive ILDC metrics.
        
        Args:
            choices: List of chosen option indices across multiple samples
            logprobs: Optional log probabilities for each option per sample
            rationales: Optional decision rationales/explanations
            num_options: Total number of options in the choice set
            
        Returns:
            Dictionary with ILDC features and overall score
        """
        if not choices:
            raise ValueError("Choices list cannot be empty")
        
        if num_options is None:
            num_options = max(choices) + 1
        
        features = {}
        
        # Core metric: Choice volatility
        features['volatility'] = self._compute_volatility(choices, num_options)
        
        # Confidence-based metrics
        if logprobs is not None and self.use_logprobs:
            features['confidence_gap'] = self._compute_confidence_gap(logprobs)
            features['entropy'] = self._compute_choice_entropy(logprobs)
        else:
            # Use voting-based confidence proxy
            features['confidence_gap'] = self._compute_voting_confidence(choices)
            features['entropy'] = self._compute_empirical_entropy(choices, num_options)
        
        # Disagreement metrics
        features['disagreement'] = self._compute_disagreement(choices)
        features['consistency_rate'] = 1.0 - features['disagreement']
        
        # Rationale complexity (if available)
        if rationales:
            features['rationale_complexity'] = self._compute_rationale_complexity(rationales)
        else:
            features['rationale_complexity'] = 0.0
        
        # Compute overall ILDC score (weighted combination)
        features['ildc_score'] = self._compute_ildc_score(features)
        
        return features
    
    def _compute_volatility(self, choices: List[int], num_options: int) -> float:
        """Compute choice volatility: how often the choice changes across samples.
        
        Args:
            choices: List of chosen options
            num_options: Total number of options
            
        Returns:
            Volatility score in [0, 1]
        """
        if len(choices) <= 1:
            return 0.0
        
        # Count unique choices
        unique_choices = len(set(choices))
        
        # Normalized volatility
        volatility = (unique_choices - 1) / (min(len(choices), num_options) - 1)
        
        return float(volatility)
    
    def _compute_confidence_gap(self, logprobs: List[List[float]]) -> float:
        """Compute average confidence gap between top-2 options.
        
        Args:
            logprobs: Log probabilities for each option per sample
            
        Returns:
            Average margin between top-2 options (smaller = less confident)
        """
        gaps = []
        
        for lp in logprobs:
            if len(lp) < 2:
                continue
            
            # Convert to probabilities and sort
            probs = np.exp(lp)
            probs = probs / probs.sum()  # Normalize
            sorted_probs = np.sort(probs)[::-1]
            
            # Gap between top 2
            gap = sorted_probs[0] - sorted_probs[1]
            gaps.append(gap)
        
        return float(np.mean(gaps)) if gaps else 0.0
    
    def _compute_voting_confidence(self, choices: List[int]) -> float:
        """Compute confidence from voting distribution (no logprobs needed).
        
        Args:
            choices: List of chosen options
            
        Returns:
            Confidence score based on majority vote strength
        """
        if not choices:
            return 0.0
        
        # Count votes
        vote_counts = Counter(choices)
        most_common = vote_counts.most_common(2)
        
        if len(most_common) < 2:
            # Perfect agreement
            return 1.0
        
        # Gap between top 2 choices
        top_count = most_common[0][1]
        second_count = most_common[1][1]
        
        gap = (top_count - second_count) / len(choices)
        
        return float(gap)
    
    def _compute_choice_entropy(self, logprobs: List[List[float]]) -> float:
        """Compute average entropy of choice distribution.
        
        Args:
            logprobs: Log probabilities for each option per sample
            
        Returns:
            Average entropy across samples
        """
        entropies = []
        
        for lp in logprobs:
            probs = np.exp(lp)
            probs = probs / probs.sum()
            ent = entropy(probs)
            entropies.append(ent)
        
        return float(np.mean(entropies)) if entropies else 0.0
    
    def _compute_empirical_entropy(self, choices: List[int], num_options: int) -> float:
        """Compute entropy from empirical choice distribution.
        
        Args:
            choices: List of chosen options
            num_options: Total number of options
            
        Returns:
            Empirical entropy
        """
        # Build probability distribution
        counts = Counter(choices)
        probs = np.zeros(num_options)
        
        for option, count in counts.items():
            probs[option] = count / len(choices)
        
        # Add smoothing to avoid log(0)
        probs = probs + 1e-10
        probs = probs / probs.sum()
        
        return float(entropy(probs))
    
    def _compute_disagreement(self, choices: List[int]) -> float:
        """Compute disagreement rate across samples.
        
        Args:
            choices: List of chosen options
            
        Returns:
            Fraction of samples that disagree with mode
        """
        if not choices:
            return 0.0
        
        # Find most common choice
        mode = Counter(choices).most_common(1)[0][0]
        
        # Count disagreements
        disagreements = sum(1 for c in choices if c != mode)
        
        return disagreements / len(choices)
    
    def _compute_rationale_complexity(self, rationales: List[str]) -> float:
        """Compute complexity/diversity of decision rationales.
        
        Args:
            rationales: List of explanation strings
            
        Returns:
            Rationale complexity score
        """
        if not rationales:
            return 0.0
        
        # Metrics:
        # 1. Average length (normalized)
        avg_length = np.mean([len(r.split()) for r in rationales])
        length_score = min(avg_length / 50, 1.0)  # Cap at 50 words
        
        # 2. Vocabulary diversity
        all_words = set()
        total_words = 0
        for r in rationales:
            words = r.lower().split()
            all_words.update(words)
            total_words += len(words)
        
        diversity = len(all_words) / max(total_words, 1)
        
        # 3. Variation across rationales (Jaccard distance)
        if len(rationales) > 1:
            similarities = []
            for i in range(len(rationales)):
                for j in range(i + 1, len(rationales)):
                    set_i = set(rationales[i].lower().split())
                    set_j = set(rationales[j].lower().split())
                    if set_i or set_j:
                        jaccard = len(set_i & set_j) / len(set_i | set_j)
                        similarities.append(jaccard)
            
            variation = 1.0 - np.mean(similarities) if similarities else 0.0
        else:
            variation = 0.0
        
        # Combined score
        complexity = 0.3 * length_score + 0.3 * diversity + 0.4 * variation
        
        return float(complexity)
    
    def _compute_ildc_score(self, features: Dict[str, float]) -> float:
        """Compute overall ILDC score from component features.
        
        Args:
            features: Dictionary of ILDC feature values
            
        Returns:
            Weighted ILDC score in [0, 1], higher = more decision difficulty
        """
        # Weights for each component
        weights = {
            'volatility': 0.35,
            'disagreement': 0.25,
            'entropy': 0.20,
            'confidence_gap': -0.15,  # Negative: larger gap = easier decision
            'rationale_complexity': 0.05
        }
        
        score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in features:
                if feature == 'confidence_gap':
                    # Invert: high confidence gap = low difficulty
                    score += weight * (1.0 - features[feature])
                else:
                    score += weight * features[feature]
                total_weight += abs(weight)
        
        # Normalize
        if total_weight > 0:
            score = score / total_weight
        
        # Ensure [0, 1] range
        score = np.clip(score, 0.0, 1.0)
        
        return float(score)
    
    def compute_batch_ildc(
        self,
        batch_choices: List[List[int]],
        batch_logprobs: Optional[List[List[List[float]]]] = None,
        batch_rationales: Optional[List[List[str]]] = None,
        num_options_list: Optional[List[int]] = None
    ) -> List[Dict[str, float]]:
        """Compute ILDC for multiple choice problems in batch.
        
        Args:
            batch_choices: List of choice lists
            batch_logprobs: Optional batch of logprobs
            batch_rationales: Optional batch of rationales
            num_options_list: List of option counts per problem
            
        Returns:
            List of ILDC feature dictionaries
        """
        results = []
        
        for i, choices in enumerate(batch_choices):
            logprobs = batch_logprobs[i] if batch_logprobs else None
            rationales = batch_rationales[i] if batch_rationales else None
            num_options = num_options_list[i] if num_options_list else None
            
            ildc = self.compute_ildc(choices, logprobs, rationales, num_options)
            results.append(ildc)
        
        return results


def compute_pairwise_disagreement(
    choices_matrix: np.ndarray,
    method: str = 'kendall'
) -> float:
    """Compute pairwise disagreement between choice rankings.
    
    Args:
        choices_matrix: Matrix of shape (n_samples, n_comparisons) with choice indices
        method: Disagreement metric ('kendall', 'hamming', 'rbo')
        
    Returns:
        Disagreement score
    """
    n_samples = choices_matrix.shape[0]
    
    if n_samples < 2:
        return 0.0
    
    disagreements = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if method == 'hamming':
                # Simple Hamming distance
                dist = np.mean(choices_matrix[i] != choices_matrix[j])
            elif method == 'kendall':
                # Kendall tau distance (rank correlation)
                from scipy.stats import kendalltau
                tau, _ = kendalltau(choices_matrix[i], choices_matrix[j])
                dist = (1 - tau) / 2  # Convert correlation to distance
            else:
                dist = np.mean(choices_matrix[i] != choices_matrix[j])
            
            disagreements.append(dist)
    
    return float(np.mean(disagreements)) if disagreements else 0.0
