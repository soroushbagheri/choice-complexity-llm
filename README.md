# Decision-Theoretic Choice Complexity in LLMs

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org)

> **A two-tier framework for measuring and regulating choice-set complexity and internal decision difficulty in Large Language Models**

## 🎯 Overview

This repository implements a novel **Decision-Theoretic Choice Complexity** framework that addresses the "paradox of choice" and bounded rationality in LLM decision-making. We propose a **two-tier architecture**:

### **Tier A: External Choice Complexity Index (CCI)**
Measures the complexity of option sets based on:
- Number of options
- Redundancy and near-duplicates
- Attribute trade-offs and conflicts
- Dominance structure (Pareto optimality)
- Option-set entropy and dispersion

### **Tier B: Internal LLM Decision Complexity (ILDC)**
Measures internal decision difficulty through:
- Choice volatility across repeated samples
- Entropy of model preferences
- Logit margins between top options
- Self-consistency disagreement
- Deliberation indicators

### **Unified Control Policy**
When complexity is high, the system can:
- Prune or cluster options
- Apply satisficing thresholds
- Provide hierarchical structuring
- Ask clarifying questions
- Reduce branching in outputs

## 🆕 Why This is Novel (2025)

While related work exists, **no prior work combines**:
1. ✅ LLM-specific choice-set complexity metrics (CCI)
2. ✅ Internal decision difficulty signals (ILDC)
3. ✅ Unified inference-time control using both tiers
4. ✅ Operationalization of bounded rationality for LLMs

### Related Work Comparison

| Work | Focus | Gap Addressed Here |
|------|-------|-------------------|
| **SITAlign** (Chehade et al., 2025) | Satisficing alignment via reward thresholds | No choice-set complexity, no dual-tier |
| **CLAI** (Zhang et al., 2025) | Cognitive-load-aware inference | Reasoning cost, not choice overload |
| **Behavioral Econ LLMs** (Jia et al., 2024) | Risk preference evaluation | Evaluation only, no control |
| **Consumer Choice AI** (Cherep et al., 2025) | LLM choice architecture sensitivity | Experimental, no metrics + controller |
| **This Work** | **Two-tier complexity + control** | **Complete framework** |

## 🔬 Research Questions

**RQ1 (Metric Validity)**: Can CCI robustly measure choice-set complexity and correlate with decision failures?

**RQ2 (Internal Difficulty)**: Does ILDC correlate with CCI and predict instability/quality issues?

**RQ3 (Control Effectiveness)**: Can (CCI, ILDC)-based control improve stability and usability?

**RQ4 (Two-Tier Benefit)**: Does dual-tier modeling outperform single-tier baselines?

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DECISION PROBLEM                      │
│          (Options with attributes/features)             │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   Module 1: CCI         │
        │   - Compute complexity  │
        │   - Feature extraction  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Module 2: LLM Engine  │
        │   - Generate choices    │
        │   - Self-consistency    │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Module 3: ILDC        │
        │   - Volatility analysis │
        │   - Confidence gaps     │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Module 4: Controller  │
        │   - Prune/cluster       │
        │   - Satisficing         │
        │   - Clarification       │
        └────────────┬────────────┘
                     │
                ┌────▼─────┐
                │  Output  │
                └──────────┘
```

## 📁 Repository Structure

```
choice-complexity-llm/
├── src/
│   ├── cci.py                 # Choice Complexity Index computation
│   ├── ildc.py                # Internal LLM Decision Complexity
│   ├── controller.py          # Control policies
│   ├── llm_adapter.py         # LLM interface (OpenAI/local)
│   ├── datasets.py            # Synthetic data generators
│   └── utils.py               # Helper functions
├── experiments/
│   ├── run_benchmark.py       # Main evaluation script
│   ├── ablation_study.py      # Ablation experiments
│   └── plotting.py            # Visualization utilities
├── configs/
│   ├── default.yaml           # Default configuration
│   └── experiments.yaml       # Experiment-specific configs
├── docs/
│   ├── ARCHITECTURE.md        # System design details
│   ├── METRICS.md             # CCI and ILDC definitions
│   ├── EXPERIMENTS.md         # How to run experiments
│   └── RELATED_WORK.md        # Literature positioning
├── tests/
│   ├── test_cci.py
│   ├── test_ildc.py
│   └── test_controller.py
├── results/
│   ├── metrics/               # CSV outputs
│   └── plots/                 # Generated figures
├── requirements.txt
├── setup.py
└── Makefile
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/soroushbagheri/choice-complexity-llm.git
cd choice-complexity-llm
pip install -r requirements.txt
```

### Run First Experiment

```bash
# Generate synthetic dataset and run benchmark
python experiments/run_benchmark.py --config configs/default.yaml

# Results will be saved to results/
```

### Expected Output

- **Metrics CSV**: `results/metrics/benchmark_results.csv`
- **Plots**:
  - `results/plots/cci_vs_ildc.png` - Correlation between complexity tiers
  - `results/plots/cci_vs_accuracy.png` - External complexity vs decision quality
  - `results/plots/ildc_vs_volatility.png` - Internal complexity vs stability
  - `results/plots/controller_effects.png` - Control policy impact

## 📊 Datasets

### Dataset A: Synthetic Choice Sets (Controlled Complexity)

```python
from src.datasets import SyntheticChoiceDataset

dataset = SyntheticChoiceDataset(
    n_options=[3, 5, 10, 20, 50],
    redundancy_ratio=[0.0, 0.3, 0.6],
    attribute_conflict=['aligned', 'conflicting'],
    pareto_front_size=[1, 3, 5]
)
```

**Features**:
- Varying number of options (3-50)
- Controlled redundancy (0-60% near-duplicates)
- Attribute conflict manipulation
- Decoy options (attraction effect)
- Pareto dominance structure

### Dataset B: Consumer-Choice-Like

Product-like items with attributes:
- Price, rating, shipping time, brand, warranty
- Ground-truth decision rules (e.g., "min price with rating ≥ 4.5")
- User profiles (budget-conscious vs quality-seeking)

## 📈 Evaluation Metrics

### Primary Metrics
- **Decision Accuracy**: Match rate with ground truth
- **Stability/Volatility**: Choice consistency across repeated runs
- **Response Burden**: Option count, explanation length
- **Efficiency**: Latency, token usage

### Correlation Analyses
- CCI ↔ ILDC
- CCI ↔ Error rate
- ILDC ↔ Volatility

### Baselines
1. No controller
2. Controller using only n (#options)
3. Controller using only CCI
4. Controller using only ILDC
5. Random pruning
6. Naive top-k pruning

## 🧪 Example Usage

```python
from src.cci import ChoiceComplexityIndex
from src.ildc import InternalLLMDecisionComplexity
from src.controller import ComplexityController
from src.llm_adapter import LLMAdapter

# Initialize components
cci_module = ChoiceComplexityIndex()
ildc_module = InternalLLMDecisionComplexity(n_samples=10)
controller = ComplexityController(cci_threshold=0.7, ildc_threshold=0.6)
llm = LLMAdapter(model="gpt-4", temperature=0.7)

# Define choice problem
options = [
    {"name": "Product A", "price": 50, "rating": 4.5, "shipping": 2},
    {"name": "Product B", "price": 45, "rating": 4.2, "shipping": 5},
    # ... more options
]

# Compute CCI
cci_score, cci_features = cci_module.compute(options)
print(f"Choice Complexity Index: {cci_score:.3f}")

# Generate LLM decisions
prompt = "Choose the best product considering price, rating, and shipping."
choices = llm.generate_choices(prompt, options, n_samples=10)

# Compute ILDC
ildc_score, ildc_features = ildc_module.compute(choices)
print(f"Internal Decision Complexity: {ildc_score:.3f}")

# Apply control policy
controlled_options, action = controller.control(
    options, cci_score, ildc_score
)
print(f"Control action: {action}")
print(f"Reduced to {len(controlled_options)} options")
```

## 🔧 Configuration

Edit `configs/default.yaml`:

```yaml
model:
  name: "gpt-4"
  temperature: 0.7
  max_tokens: 500

cci:
  weights:
    n_options: 0.3
    redundancy: 0.25
    conflict: 0.25
    entropy: 0.2

ildc:
  n_samples: 10
  volatility_window: 5

controller:
  cci_threshold: 0.7
  ildc_threshold: 0.6
  max_options: 5
  clustering_method: "kmeans"

dataset:
  n_problems: 100
  n_options_range: [3, 50]
  seed: 42
```

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed system design
- **[METRICS.md](docs/METRICS.md)**: CCI and ILDC mathematical definitions
- **[EXPERIMENTS.md](docs/EXPERIMENTS.md)**: Running experiments and interpreting results
- **[RELATED_WORK.md](docs/RELATED_WORK.md)**: Literature review and positioning

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@software{bagheri2026choice_complexity,
  author = {Bagheri, Soroush},
  title = {Decision-Theoretic Choice Complexity in LLMs: A Two-Tier Framework},
  year = {2026},
  url = {https://github.com/soroushbagheri/choice-complexity-llm},
  note = {Under review}
}
```

## 🤝 Contributing

Contributions welcome! Areas for extension:
- Additional complexity metrics
- More controller policies
- Integration with RAG systems
- Multi-agent decision scenarios
- Real-world datasets

## 📧 Contact

Soroush Bagheri - [GitHub](https://github.com/soroushbagheri)

For questions or collaboration: Open an issue or reach out via GitHub.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Keywords**: Large Language Models, Decision Theory, Choice Complexity, Bounded Rationality, Satisficing, Choice Overload, Inference Control, LLM Agents, Behavioral Economics, AI Decision Making
