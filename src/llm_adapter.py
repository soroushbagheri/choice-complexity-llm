"""llm_adapter.py

Adapter for LLM inference supporting multiple backends (OpenAI, local models, mock).
Provides unified interface for choice-making tasks with logging and error handling.
"""

import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMAdapter:
    """Unified adapter for LLM inference across different backends."""
    
    def __init__(
        self,
        backend: str = 'mock',
        model_name: str = 'gpt-3.5-turbo',
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        **kwargs
    ):
        """
        Args:
            backend: 'openai', 'local', or 'mock'
            model_name: Model identifier
            api_key: API key for OpenAI
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        if backend == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            if api_key:
                openai.api_key = api_key
            logger.info(f"Initialized OpenAI backend with model {model_name}")
        
        elif backend == 'mock':
            logger.info("Initialized mock LLM backend for testing")
        
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def choose(
        self,
        options: List[Dict[str, Any]],
        context: str = "",
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make a choice among options.
        
        Args:
            options: List of option dictionaries with 'id', 'description', and optional attributes
            context: Task context/instructions
            temperature: Override default temperature
        
        Returns:
            Dictionary with:
                - choice: Selected option index
                - rationale: Explanation
                - confidence: Confidence score (if available)
                - tokens_used: Token count
                - latency: Response time
        """
        start_time = time.time()
        temp = temperature if temperature is not None else self.temperature
        
        # Build prompt
        prompt = self._build_choice_prompt(options, context)
        
        if self.backend == 'openai':
            result = self._openai_choose(prompt, temp)
        elif self.backend == 'mock':
            result = self._mock_choose(options, temp)
        else:
            raise ValueError(f"Backend {self.backend} not implemented")
        
        result['latency'] = time.time() - start_time
        return result
    
    def _build_choice_prompt(self, options: List[Dict], context: str) -> str:
        """Construct prompt for choice task."""
        prompt = f"{context}\n\n" if context else ""
        prompt += "Please select the best option from the following choices:\n\n"
        
        for i, opt in enumerate(options):
            prompt += f"Option {i}: {opt.get('description', '')}\n"
            
            # Add attributes if present
            if 'attributes' in opt:
                for key, value in opt['attributes'].items():
                    prompt += f"  - {key}: {value}\n"
            prompt += "\n"
        
        prompt += ("\nProvide your choice as 'Option X' where X is the option number, "
                  "followed by a brief rationale (2-3 sentences).")
        return prompt
    
    def _openai_choose(self, prompt: str, temperature: float) -> Dict[str, Any]:
        """OpenAI API inference."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            # Parse choice from response
            choice_idx, rationale = self._parse_choice_response(content)
            
            return {
                'choice': choice_idx,
                'rationale': rationale,
                'raw_response': content,
                'tokens_used': tokens,
                'confidence': None  # Not available from OpenAI
            }
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _mock_choose(self, options: List[Dict], temperature: float) -> Dict[str, Any]:
        """Mock LLM for testing (uses simple heuristics)."""
        # Simulate choice based on temperature
        if temperature < 0.3:  # Deterministic: choose first option
            choice_idx = 0
        elif temperature > 0.7:  # Stochastic: random weighted by inverse order
            weights = np.exp(-np.arange(len(options)) / 2)
            weights /= weights.sum()
            choice_idx = np.random.choice(len(options), p=weights)
        else:  # Semi-stochastic
            choice_idx = np.random.randint(0, min(3, len(options)))
        
        # Generate mock rationale
        rationale = f"Selected Option {choice_idx} based on its attributes and overall suitability."
        
        # Simulate token usage
        tokens = 50 + len(options) * 10
        
        return {
            'choice': choice_idx,
            'rationale': rationale,
            'raw_response': f"Option {choice_idx}\n{rationale}",
            'tokens_used': tokens,
            'confidence': 0.7 + 0.2 * np.random.random()
        }
    
    def _parse_choice_response(self, response: str) -> tuple:
        """Extract choice index and rationale from LLM response."""
        lines = response.strip().split('\n')
        
        # Find line with "Option X"
        choice_idx = 0
        for line in lines:
            if 'Option' in line:
                try:
                    # Extract number after 'Option'
                    parts = line.split('Option')
                    if len(parts) > 1:
                        num_str = ''.join(c for c in parts[1] if c.isdigit())
                        if num_str:
                            choice_idx = int(num_str)
                            break
                except:
                    continue
        
        # Rationale is remaining text
        rationale = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        
        return choice_idx, rationale
    
    def get_pairwise_preferences(
        self,
        option_a: Dict,
        option_b: Dict,
        context: str = ""
    ) -> Dict[str, Any]:
        """Get preference between two options (for ILDC computation)."""
        prompt = f"{context}\n\n" if context else ""
        prompt += "Compare the following two options and state which you prefer:\n\n"
        prompt += f"Option A: {option_a.get('description', '')}\n"
        prompt += f"Option B: {option_b.get('description', '')}\n"
        prompt += "\nWhich option do you prefer? Respond with 'Option A' or 'Option B' and explain why."
        
        if self.backend == 'mock':
            # Mock pairwise comparison
            preference = np.random.choice(['A', 'B'])
            return {
                'preference': preference,
                'rationale': f"Option {preference} is preferable due to its characteristics."
            }
        
        elif self.backend == 'openai':
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=100
            )
            content = response.choices[0].message.content
            preference = 'A' if 'Option A' in content else 'B'
            return {'preference': preference, 'rationale': content}
        
        return {'preference': 'A', 'rationale': ''}
