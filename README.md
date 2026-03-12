# Decision-Theoretic Choice Complexity in LLMs

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soroushbagheri/choice-complexity-llm/blob/main/notebooks/demo_colab.ipynb)

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

See [docs/REFERENCES.md](docs/REFERENCES.md) for comprehensive literature review.

## 🔬 Research Questions

**RQ1 (Metric Validity)**: Can CCI robustly measure choice-set complexity and correlate with decision failures?

**RQ2 (Internal Difficulty)**: Does ILDC correlate with CCI and predict instability/quality issues?

**RQ3 (Control Effectiveness)**: Can (CCI, ILDC)-based control improve stability and usability?

**RQ4 (Two-Tier Benefit)**: Does dual-tier modeling outperform single-tier baselines?

## 🚀 Quick Start

###  Try It in Google Colab

 interactive demo:

(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soroushbagheri/choice-complexity-llm/blob/main/notebooks/demo_colab.ipynb)

The Colab notebook includes:
- ✅ Automated setup and installation
- ✅ Interactive demo with synthetic data (no API keys needed)
- ✅ Visual results and plots
- ✅ Statistical analysis
- ✅ Downloadable results
- ✅ Custom experiment builder

### 💻 Local Installation

```bash
# Clone repository
git clone https://github.com/soroushbagheri/choice-complexity-llm.git
cd choice-complexity-llm

# Install dependencies
pip install -r requirements.txt
```

### Run Demo (No LLM API Required)

The demo uses a **synthetic LLM simulator** to demonstrate the framework without needing API access:

```bash
# Run demo with default settings (100 samples)
python experiments/demo_with_results.py

# Run with custom parameters
python experiments/demo_with_results.py --n-samples 200 --seed 42 --output results/demo_v1
```

**Expected runtime**: ~30 seconds for 100 samples

**Output**:
```
results/demo/
├── demo_results.csv          # Full experimental data
├── summary.json              # Aggregate metrics
└── plots/
    ├── cci_vs_ildc.png       # Correlation between complexity tiers
    ├── cci_vs_accuracy.png   # External complexity vs quality
    ├── controller_effects.png # Controller strategy comparison
    ├── complexity_distributions.png
    └── correlation_heatmap.png
```

### Run with Real LLM

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Run full benchmark
python experiments/run_benchmark.py --config configs/default.yaml

# Results saved to results/run_TIMESTAMP/
```

## 📊 Example Results

### Key Findings from Demo

| Controller Strategy | Accuracy | Volatility | Options Shown | CCI | ILDC |
|---------------------|----------|------------|---------------|-----|------|
| None (baseline)     | 0.687    | 0.421      | 15.3          | 0.58| 0.44 |
| Naive Top-K         | 0.702    | 0.398      | 5.0           | 0.58| 0.41 |
| CCI Only            | 0.734    | 0.362      | 8.2           | 0.58| 0.38 |
| **Two-Tier**        | **0.761**| **0.325**  | 6.7           | 0.58| **0.35** |

**Correlations**:
- CCI ↔ ILDC: **0.73** (strong positive - validates two-tier coupling)
- CCI ↔ Accuracy: **-0.61** (higher complexity → lower accuracy)
- ILDC ↔ Volatility: **0.68** (higher internal complexity → more instability)

### Visualization Preview

![Controller Effects](results/demo/plots/controller_effects.png)
*Figure: Two-tier controller reduces volatility and improves accuracy*

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
│   ├── controller.py          # Control policies (prune/cluster/satisficing)
│   ├── llm_adapter.py         # LLM interface (OpenAI/local models)
│   ├── datasets.py            # Synthetic data generators
│   └── utils.py               # Helper functions
├── experiments/
│   ├── run_benchmark.py       # Main evaluation script
│   ├── demo_with_results.py   # Demo with synthetic LLM (no API needed)
│   ├── ablation_study.py      # Ablation experiments
│   └── plotting.py            # Visualization utilities
├── notebooks/
│   └── demo_colab.ipynb       # 🆕 Interactive Google Colab demo
├── configs/
│   ├── default.yaml           # Default configuration
│   └── experiments.yaml       # Experiment-specific configs
├── docs/
│   ├── REFERENCES.md          # Comprehensive literature review
│   ├── ARCHITECTURE.md        # System design details (TODO)
│   ├── METRICS.md             # CCI and ILDC definitions (TODO)
│   └── EXPERIMENTS.md         # How to run experiments (TODO)
├── tests/
│   ├── test_cci.py           # CCI unit tests
│   ├── test_ildc.py          # ILDC unit tests
│   └── test_controller.py    # Controller tests
├── results/
│   ├── demo/                  # Demo outputs
│   └── experiments/           # Experiment results
├── requirements.txt
└── README.md
```

## 📚 Documentation

- **[REFERENCES.md](docs/REFERENCES.md)**: Comprehensive literature review (2024-2025 papers)
- **[Google Colab Demo](notebooks/demo_colab.ipynb)**: Interactive demo notebook
- **ARCHITECTURE.md**: Detailed system design (coming soon)
- **METRICS.md**: Mathematical definitions of CCI and ILDC (coming soon)
- **EXPERIMENTS.md**: Running experiments and interpreting results (coming soon)

## 📊 Datasets

### Dataset A: Synthetic Choice Sets (Controlled Complexity)

```python
from src.datasets import SyntheticChoiceDataset

dataset = SyntheticChoiceDataset(
    n_options_range=[3, 5, 10, 20, 50],
    redundancy_ratios=[0.0, 0.3, 0.6],
    attribute_conflict_types=['aligned', 'conflicting'],
    seed=42
)

samples = dataset.generate_dataset(n_samples=100)
```

**Features**:
- Varying number of options (3-50)
- Controlled redundancy (0-60% near-duplicates)
- Attribute conflict manipulation
- Decoy options (attraction effect)
- Pareto dominance structure
- Ground-truth optimal choices

### Dataset B: Consumer-Choice-Like

Product-like items with attributes:
- Price, rating, shipping time, brand, warranty
- Ground-truth decision rules (e.g., "min price with rating ≥ 4.5")
- User profiles (budget-conscious vs quality-seeking)

## 🧪 Example Usage

```python
from src.cci import ChoiceComplexityIndex
from src.ildc import ILDCComputer
from src.controller import ChoiceComplexityController
from src.llm_adapter import LLMAdapter

# Initialize components
cci_calculator = ChoiceComplexityIndex()
ildc_calculator = ILDCComputer(n_samples=10)
controller = ChoiceComplexityController()
llm = LLMAdapter(model="gpt-4", temperature=0.7)

# Define choice problem
options = [
    {"id": 0, "name": "Product A", "attributes": {"price": 50, "rating": 4.5, "shipping": 2}},
    {"id": 1, "name": "Product B", "attributes": {"price": 45, "rating": 4.2, "shipping": 5}},
    {"id": 2, "name": "Product C", "attributes": {"price": 55, "rating": 4.8, "shipping": 1}},
    # ... more options
]

# Step 1: Compute CCI
cci_result = cci_calculator.compute(options)
print(f"Choice Complexity Index: {cci_result['cci_score']:.3f}")
print(f"Features: n={cci_result['features']['n_options']}, "
      f"redundancy={cci_result['features']['redundancy_ratio']:.2f}")

# Step 2: Generate LLM decisions (multiple samples for ILDC)
prompt = "Choose the best product considering price, rating, and shipping."
choices = []
for _ in range(10):
    response = llm.choose(options=options, context=prompt)
    choices.append(response)

# Step 3: Compute ILDC
ildc_result = ildc_calculator.compute(choices)
print(f"Internal Decision Complexity: {ildc_result['ildc_score']:.3f}")
print(f"Volatility: {ildc_result['features']['volatility']:.2f}")

# Step 4: Apply controller
action_type = controller.decide_action(
    cci_score=cci_result['cci_score'],
    ildc_score=ildc_result['ildc_score'],
    num_options=len(options)
)
control_action = controller.apply_action(action_type, options)
print(f"Controller action: {control_action.action_type}")
print(f"Options reduced from {len(options)} to {control_action.num_options_after}")

# Step 5: Make final decision with controlled set
controlled_options = [options[i] for i in control_action.selected_indices]
final_response = llm.choose(options=controlled_options, context=prompt)
print(f"Final choice: {final_response['choice']} - {final_response['reasoning']}")
```

## 🔧 Configuration

Edit `configs/default.yaml`:

```yaml
model:
  name: "gpt-4"
  temperature: 0.7
  max_tokens: 500
  api_key: null  # Or set OPENAI_API_KEY env variable

cci:
  weights:
    n_options: 0.3
    redundancy: 0.25
    conflict: 0.25
    entropy: 0.2
  normalize: true

ildc:
  n_samples: 10
  volatility_window: 5
  confidence_method: "self_eval"  # or "logprobs"

controller:
  cci_threshold: 0.6
  ildc_threshold: 0.5
  max_options: 5
  pruning_strategy: "diverse"
  clustering_method: "hierarchical"

dataset:
  n_samples: 100
  n_options_range: [3, 50]
  seed: 42
```

## 📈 Evaluation Metrics

### Primary Metrics
- **Decision Accuracy**: Match rate with ground truth
- **Stability/Volatility**: Choice consistency across repeated runs
- **Response Burden**: Option count, explanation length
- **Efficiency**: Latency, token usage

### Correlation Analyses
- CCI ↔ ILDC (two-tier coupling validation)
- CCI ↔ Error rate (complexity impact on quality)
- ILDC ↔ Volatility (internal difficulty and instability)

### Baselines
1. **None**: No controller
2. **Naive Top-K**: Always show top-k options
3. **CCI Only**: Controller using only external complexity
4. **ILDC Only**: Controller using only internal complexity
5. **Two-Tier**: Combined CCI + ILDC controller
6. **Random Pruning**: Random option removal

## 🧬 Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_cci.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

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
- Additional complexity metrics (e.g., temporal dynamics, hierarchical structure)
- More controller policies (e.g., active learning, clarification questions)
- Integration with RAG systems
- Multi-agent decision scenarios
- Real-world datasets (e-commerce, healthcare, legal)
- Theoretical analysis (sample complexity, PAC bounds)

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Format code
black src/ experiments/ tests/

# Type checking
mypy src/
```

## 📧 Contact

**Soroush Bagheri**
- GitHub: [@soroushbagheri](https://github.com/soroushbagheri)
- Email: [via GitHub profile]

For questions or collaboration:
- Open an issue for bugs/feature requests
- Start a discussion for conceptual questions
- Reach out via GitHub for research collaboration

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by behavioral economics work of Schwartz, Simon, Iyengar & Lepper
- Built on LLM agent research (Wang et al., 2024)
- Related to satisficing alignment (Chehade et al., 2025) and cognitive load inference (Zhang et al., 2025)

---

**Keywords**: Large Language Models, Decision Theory, Choice Complexity, Bounded Rationality, Satisficing, Choice Overload, Inference Control, LLM Agents, Behavioral Economics, AI Decision Making, RAG, Multi-Agent Systems

---

⭐ **Star this repo** if you find it useful for your research!
