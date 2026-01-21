"""Controller Module for Choice Complexity Regulation.

This module implements inference-time control policies that use (CCI, ILDC) to:
- Prune/limit the number of options presented
- Cluster redundant options
- Apply satisficing thresholds
- Ask clarifying questions
- Provide hierarchical option structures

Author: Soroush Bagheri
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import warnings


@dataclass
class ControlAction:
    """Represents a control action taken by the controller.
    
    Attributes:
        action_type: Type of action ('prune', 'cluster', 'hierarchical', 'clarify', 'satisfice', 'none')
        num_options_after: Number of options after action
        selected_indices: Indices of options to present
        clusters: Optional cluster assignments
        clarifying_question: Optional question to ask
        explanation: Human-readable explanation of action
    """
    action_type: str
    num_options_after: int
    selected_indices: List[int]
    clusters: Optional[Dict[int, List[int]]] = None
    clarifying_question: Optional[str] = None
    explanation: str = ""


class ChoiceComplexityController:
    """Controller for regulating choice complexity based on CCI and ILDC.
    
    Attributes:
        cci_threshold: Threshold for external complexity intervention
        ildc_threshold: Threshold for internal complexity intervention
        max_options: Maximum number of options to present
        pruning_strategy: Strategy for option selection ('top_k', 'diverse', 'pareto')
        clustering_method: Clustering approach ('hierarchical', 'kmeans')
        use_hierarchical: Whether to support hierarchical presentation
    """
    
    def __init__(
        self,
        cci_threshold: float = 0.6,
        ildc_threshold: float = 0.5,
        max_options: int = 5,
        pruning_strategy: str = 'diverse',
        clustering_method: str = 'hierarchical',
        use_hierarchical: bool = True,
        use_clarification: bool = True
    ):
        """Initialize controller.
        
        Args:
            cci_threshold: CCI above which to intervene
            ildc_threshold: ILDC above which to intervene
            max_options: Maximum options to present
            pruning_strategy: How to select options
            clustering_method: How to cluster
            use_hierarchical: Enable hierarchical presentation
            use_clarification: Enable clarifying questions
        """
        self.cci_threshold = cci_threshold
        self.ildc_threshold = ildc_threshold
        self.max_options = max_options
        self.pruning_strategy = pruning_strategy
        self.clustering_method = clustering_method
        self.use_hierarchical = use_hierarchical
        self.use_clarification = use_clarification
    
    def decide_action(
        self,
        cci_score: float,
        ildc_score: Optional[float],
        num_options: int,
        cci_features: Optional[Dict[str, float]] = None,
        ildc_features: Optional[Dict[str, float]] = None
    ) -> str:
        """Decide which control action to take.
        
        Args:
            cci_score: External choice complexity score
            ildc_score: Internal decision difficulty score (can be None initially)
            num_options: Current number of options
            cci_features: Optional CCI component features
            ildc_features: Optional ILDC component features
            
        Returns:
            Action type string
        """
        # Decision tree for action selection
        
        # Case 1: Very high external complexity
        if cci_score > 0.8 and num_options > 10:
            # High redundancy? -> Cluster
            if cci_features and cci_features.get('redundancy_ratio', 0) > 0.4:
                return 'cluster'
            # Ask clarifying question to reduce scope
            elif self.use_clarification and cci_features:
                return 'clarify'
            else:
                return 'prune'
        
        # Case 2: High external complexity + high internal difficulty
        if cci_score > self.cci_threshold and ildc_score and ildc_score > self.ildc_threshold:
            # Both are high -> aggressive pruning or hierarchical
            if num_options > self.max_options * 2:
                return 'hierarchical' if self.use_hierarchical else 'prune'
            else:
                return 'prune'
        
        # Case 3: High external complexity alone
        if cci_score > self.cci_threshold:
            if num_options > self.max_options:
                # Check if satisficing is appropriate
                if cci_features and cci_features.get('pareto_front_size', num_options) < num_options * 0.3:
                    return 'satisfice'  # Many dominated options
                else:
                    return 'prune'
            else:
                return 'none'  # Acceptable complexity
        
        # Case 4: High internal difficulty alone
        if ildc_score and ildc_score > self.ildc_threshold:
            # Model is struggling despite moderate external complexity
            # -> Simplify presentation
            if num_options > self.max_options:
                return 'prune'
            elif ildc_features and ildc_features.get('volatility', 0) > 0.7:
                return 'clarify'  # Ask for preference
            else:
                return 'none'
        
        # Case 5: Complexity is manageable
        return 'none'
    
    def apply_action(
        self,
        action_type: str,
        options: List[Dict[str, Any]],
        cci_features: Optional[Dict[str, float]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        utilities: Optional[np.ndarray] = None
    ) -> ControlAction:
        """Apply the selected control action.
        
        Args:
            action_type: Type of action to apply
            options: List of option dictionaries
            cci_features: CCI component features
            similarity_matrix: Pairwise similarity between options
            utilities: Optional utility scores for each option
            
        Returns:
            ControlAction with results
        """
        num_options = len(options)
        
        if action_type == 'none':
            return ControlAction(
                action_type='none',
                num_options_after=num_options,
                selected_indices=list(range(num_options)),
                explanation="No intervention needed - complexity is manageable"
            )
        
        elif action_type == 'prune':
            return self._prune_options(options, utilities, similarity_matrix)
        
        elif action_type == 'cluster':
            return self._cluster_options(options, similarity_matrix)
        
        elif action_type == 'hierarchical':
            return self._create_hierarchical_structure(options, utilities, similarity_matrix)
        
        elif action_type == 'clarify':
            return self._generate_clarifying_question(options, cci_features)
        
        elif action_type == 'satisfice':
            return self._apply_satisficing(options, utilities)
        
        else:
            warnings.warn(f"Unknown action type: {action_type}. Returning all options.")
            return ControlAction(
                action_type='none',
                num_options_after=num_options,
                selected_indices=list(range(num_options)),
                explanation=f"Unknown action '{action_type}'"
            )
    
    def _prune_options(
        self,
        options: List[Dict[str, Any]],
        utilities: Optional[np.ndarray],
        similarity_matrix: Optional[np.ndarray]
    ) -> ControlAction:
        """Prune options to top-k based on strategy.
        
        Args:
            options: List of options
            utilities: Optional utility scores
            similarity_matrix: Optional similarity matrix
            
        Returns:
            ControlAction with pruned option indices
        """
        num_options = len(options)
        target_size = min(self.max_options, num_options)
        
        if self.pruning_strategy == 'top_k':
            # Select top k by utility
            if utilities is not None:
                selected = np.argsort(utilities)[::-1][:target_size].tolist()
            else:
                # Random selection if no utilities
                selected = list(range(target_size))
            
            explanation = f"Selected top {target_size} options by utility"
        
        elif self.pruning_strategy == 'diverse':
            # Maximal marginal relevance: balance utility and diversity
            if utilities is not None and similarity_matrix is not None:
                selected = self._mmr_selection(utilities, similarity_matrix, target_size)
                explanation = f"Selected {target_size} diverse high-utility options"
            else:
                # Fallback to top-k
                selected = list(range(target_size))
                explanation = f"Selected first {target_size} options (no diversity info)"
        
        elif self.pruning_strategy == 'pareto':
            # Select Pareto-optimal options
            if utilities is not None and isinstance(utilities[0], (list, np.ndarray)):
                pareto_indices = self._compute_pareto_front(utilities)
                selected = pareto_indices[:target_size]
                explanation = f"Selected {len(selected)} Pareto-optimal options"
            else:
                selected = list(range(target_size))
                explanation = f"Pareto pruning not applicable - using top {target_size}"
        
        else:
            selected = list(range(target_size))
            explanation = f"Selected first {target_size} options"
        
        return ControlAction(
            action_type='prune',
            num_options_after=len(selected),
            selected_indices=selected,
            explanation=explanation
        )
    
    def _cluster_options(
        self,
        options: List[Dict[str, Any]],
        similarity_matrix: Optional[np.ndarray]
    ) -> ControlAction:
        """Cluster similar options and select representatives.
        
        Args:
            options: List of options
            similarity_matrix: Pairwise similarity
            
        Returns:
            ControlAction with cluster information
        """
        num_options = len(options)
        target_clusters = min(self.max_options, num_options)
        
        if similarity_matrix is None:
            # Can't cluster without similarity
            return ControlAction(
                action_type='cluster',
                num_options_after=target_clusters,
                selected_indices=list(range(target_clusters)),
                explanation="Clustering unavailable - selected first k options"
            )
        
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=target_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Build cluster dictionary
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Select one representative per cluster (most central)
        selected = []
        for cluster_id, member_indices in clusters.items():
            if len(member_indices) == 1:
                selected.append(member_indices[0])
            else:
                # Pick most central (highest avg similarity to others in cluster)
                centralities = []
                for idx in member_indices:
                    avg_sim = np.mean([similarity_matrix[idx, other] 
                                      for other in member_indices if other != idx])
                    centralities.append(avg_sim)
                best = member_indices[np.argmax(centralities)]
                selected.append(best)
        
        explanation = f"Clustered {num_options} options into {target_clusters} groups, showing representatives"
        
        return ControlAction(
            action_type='cluster',
            num_options_after=len(selected),
            selected_indices=selected,
            clusters=clusters,
            explanation=explanation
        )
    
    def _create_hierarchical_structure(
        self,
        options: List[Dict[str, Any]],
        utilities: Optional[np.ndarray],
        similarity_matrix: Optional[np.ndarray]
    ) -> ControlAction:
        """Create hierarchical option presentation (top-3 + drill-down).
        
        Args:
            options: List of options
            utilities: Optional utility scores
            similarity_matrix: Optional similarity
            
        Returns:
            ControlAction with hierarchical structure
        """
        num_options = len(options)
        top_k = min(3, num_options)
        
        if utilities is not None:
            # Top 3 by utility
            top_indices = np.argsort(utilities)[::-1][:top_k].tolist()
        else:
            top_indices = list(range(top_k))
        
        explanation = f"Showing top {top_k} options; {num_options - top_k} others available on request"
        
        return ControlAction(
            action_type='hierarchical',
            num_options_after=top_k,
            selected_indices=top_indices,
            explanation=explanation
        )
    
    def _generate_clarifying_question(
        self,
        options: List[Dict[str, Any]],
        cci_features: Optional[Dict[str, float]]
    ) -> ControlAction:
        """Generate a clarifying question to reduce choice scope.
        
        Args:
            options: List of options
            cci_features: CCI features
            
        Returns:
            ControlAction with clarifying question
        """
        # Analyze attribute variation to generate question
        if not options or 'attributes' not in options[0]:
            question = "Could you specify your most important criterion?"
        else:
            # Find attribute with highest variance
            attributes = options[0]['attributes'].keys()
            variances = {}
            for attr in attributes:
                values = [opt['attributes'][attr] for opt in options if 'attributes' in opt]
                variances[attr] = np.var(values) if len(values) > 0 else 0
            
            max_var_attr = max(variances, key=variances.get) if variances else "price"
            question = f"What is your preference for {max_var_attr}? (e.g., minimize, maximize, or specific range)"
        
        return ControlAction(
            action_type='clarify',
            num_options_after=len(options),
            selected_indices=list(range(len(options))),
            clarifying_question=question,
            explanation="Asking for user preference to reduce complexity"
        )
    
    def _apply_satisficing(
        self,
        options: List[Dict[str, Any]],
        utilities: Optional[np.ndarray]
    ) -> ControlAction:
        """Apply satisficing: pick first option above threshold.
        
        Args:
            options: List of options
            utilities: Optional utility scores
            
        Returns:
            ControlAction with satisficing selection
        """
        if utilities is None:
            # No utility info - can't satisfice
            selected = [0]
            explanation = "Satisficing unavailable - selected first option"
        else:
            # Threshold = 80% of max utility
            threshold = 0.8 * np.max(utilities)
            above_threshold = np.where(utilities >= threshold)[0]
            
            if len(above_threshold) > 0:
                selected = [above_threshold[0]]  # First acceptable
                explanation = f"Satisficing: selected first option above threshold ({threshold:.2f})"
            else:
                selected = [np.argmax(utilities)]
                explanation = "No options above threshold - selected best"
        
        return ControlAction(
            action_type='satisfice',
            num_options_after=1,
            selected_indices=selected,
            explanation=explanation
        )
    
    def _mmr_selection(
        self,
        utilities: np.ndarray,
        similarity_matrix: np.ndarray,
        k: int,
        lambda_param: float = 0.7
    ) -> List[int]:
        """Maximal Marginal Relevance selection for diversity.
        
        Args:
            utilities: Utility scores
            similarity_matrix: Pairwise similarities
            k: Number of items to select
            lambda_param: Trade-off between utility and diversity
            
        Returns:
            List of selected indices
        """
        selected = []
        remaining = list(range(len(utilities)))
        
        # Start with highest utility
        first = int(np.argmax(utilities))
        selected.append(first)
        remaining.remove(first)
        
        # Iteratively add items maximizing MMR score
        while len(selected) < k and remaining:
            mmr_scores = []
            for idx in remaining:
                utility_score = utilities[idx]
                # Max similarity to already selected items
                max_sim = max([similarity_matrix[idx, s] for s in selected])
                mmr = lambda_param * utility_score - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            
            best_idx = remaining[int(np.argmax(mmr_scores))]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected
    
    def _compute_pareto_front(
        self,
        utilities: np.ndarray
    ) -> List[int]:
        """Compute Pareto-optimal front for multi-attribute utilities.
        
        Args:
            utilities: Array of shape (n_options, n_attributes)
            
        Returns:
            Indices of Pareto-optimal options
        """
        n = len(utilities)
        is_pareto = np.ones(n, dtype=bool)
        
        for i in range(n):
            if is_pareto[i]:
                # Check if any other option dominates i
                is_pareto[is_pareto] = np.any(
                    utilities[is_pareto] <= utilities[i], axis=1
                )
                is_pareto[i] = True  # Keep i
        
        return np.where(is_pareto)[0].tolist()
