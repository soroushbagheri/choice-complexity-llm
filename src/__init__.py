"""Decision-Theoretic Choice Complexity in LLMs."""

__version__ = "0.1.0"
__author__ = "Soroush Bagheri"

from .cci import ChoiceComplexityIndex
from .ildc import ILDCComputer
from .controller import ComplexityController
from .llm_adapter import LLMAdapter
from .datasets import SyntheticChoiceDataset, ConsumerChoiceDataset

__all__ = [
    "ChoiceComplexityIndex",
    "ILDCComputer",
    "ComplexityController",
    "LLMAdapter",
    "SyntheticChoiceDataset",
    "ConsumerChoiceDataset",
]
