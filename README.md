# Choice Complexity in LLMs

> **A Decision-Theoretic Framework for Controlling the Cognitive Cost of LLM-Generated Option Sets**

[
[
[

***

## What is this project?

When a large language model responds to a question, it often generates multiple candidate options — recommendations, suggestions, next steps, or alternative answers. More options can feel helpful, but beyond a certain point they increase cognitive burden: the user slows down, second-guesses themselves, or disengages entirely.

This project studies that problem. Inspired by Barry Schwartz's **Paradox of Choice**, we ask:

> *Can we formally measure the cognitive complexity of an LLM-generated option set, and can we use that measure to control what the model produces?*

The answer is what we are building here.

***

## Research Questions

**RQ1.** Can we define a formal, computable **Choice Complexity Index (CCI)** that captures how cognitively demanding an LLM-generated option set is for a user?

**RQ2.** Does the CCI correlate with observable proxies of decision difficulty — such as set redundancy, entropy, and regret gap between options?

**RQ3.** Can a symbolic scorer module use the CCI to prune or rerank LLM-generated options without materially reducing task quality or coverage?

**RQ4.** Does complexity-aware generation improve decision clarity across general NLP domains and transfer to high-stakes settings such as clinical decision support?

***

## Hypotheses

| ID | Hypothesis |
|---|---|
| H1 | Larger and more redundant option sets will produce higher CCI scores and higher perceived difficulty |
| H2 | CCI — as a combination of set size, semantic redundancy, utility entropy, and regret gap — will correlate with decision difficulty more reliably than set size alone |
| H3 | Complexity-aware pruning or reranking will reduce CCI without materially reducing coverage of the correct or most useful option |
| H4 | A domain-general CCI framework will generalise across NLP tasks and be reproducible by independent researchers without task-specific engineering |

***

## Proposed Method

The system works in three stages:

```
Input Query
    │
    ▼
┌─────────────────────────────┐
│  LLM Option Generator       │  ← standard LLM, top-k or beam search
│  produces N candidate opts  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Symbolic CCI Scorer        │  ← computes CCI for the candidate set
│  CCI = f(|S|, H, R, D)     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Complexity-Aware Selector  │  ← prunes/reranks to satisfy CCI budget C*
│  outputs final option set   │
└─────────────────────────────┘
             │
             ▼
        User Output
```

***

## Choice Complexity Index (CCI)

We define the CCI for a generated option set **S** as:

$$CCI(S) = w_1 \cdot |S| + w_2 \cdot H(S) + w_3 \cdot R(S) + w_4 \cdot D(S)$$

Where:

| Term | Symbol | Description |
|---|---|---|
| Set size | `|S|` | Raw number of options |
| Utility entropy | `H(S)` | Dispersion of model confidence or utility scores across options — high entropy = hard to discriminate |
| Redundancy | `R(S)` | Average pairwise semantic similarity — high similarity = overlapping options |
| Regret gap | `D(S)` | Distance between best option and mean — large gap = the model is hedging |

- **Low CCI** → few clear, non-overlapping options → easy to choose
- **High CCI** → many similar, competing options → likely to overwhelm

Weights *w₁–w₄* are calibrated empirically via ablation or correlation with human decision difficulty proxies.

***

## Theoretical Basis

This project draws on three bodies of work:

- **Barry Schwartz, *The Paradox of Choice* (2004)** — the core motivation: more options can reduce decision quality and satisfaction. We operationalise this into a formal metric.
- **Random Utility Models (Dean et al., 2025)** — statistical tests for choice overload as violations of monotonicity. Used to validate that CCI correlates with real overload effects.
- **Neurosymbolic AI** — the symbolic CCI scorer acts as an interpretable constraint on top of a neural generator, in the spirit of neurosymbolic integration (cf. CLAI, Zhang 2025; neurosymbolic reasoning, EMNLP 2025).

***

## Scope

This project is framed as a **general NLP / ML method**, not a domain-specific system.

- **Primary scope:** open-domain recommendation, multi-answer QA, and general decision-support tasks
- **Demonstration domain:** clinical triage / next-step recommendation in medicine — a high-stakes setting where choice overload has clear consequences
- This separation gives the framework broader impact, better reproducibility, and a clearer path to being followed by other researchers

***

## Datasets

> 🔲 **TODO: Dataset loading scripts not yet implemented**

Planned evaluation settings:

| Dataset | Task | Status |
|---|---|---|
| AmbigNQ / ASQA | Multi-answer open-domain QA | 🔲 Planned |
| MovieLens / Amazon Reviews | Open-domain recommendation | 🔲 Planned |
| ER-Reason | Clinical triage / next-step recommendation | 🔲 Planned |
| Custom vignette set | Controlled pilot with known complexity levels | 🔲 Planned |

***

## Baselines

> 🔲 **TODO: Baseline implementations not yet added**

| Baseline | Description |
|---|---|
| Standard top-k | LLM top-k generation, no complexity control |
| Diverse beam search | Maximises output diversity |
| Confidence-ranked top-k | Select by model probability only |
| Unconstrained generation | Full candidate set, no pruning |
| Single-answer prompting | Prompt the model for exactly one answer |

***

## Evaluation

> 🔲 **TODO: Evaluation pipeline not yet implemented**

### Automatic Metrics

| Metric | What it measures |
|---|---|
| CCI before/after pruning | Core complexity reduction |
| Semantic redundancy (avg pairwise cosine similarity) | Redundancy in the output set |
| Coverage@k | Does the final set still contain the correct or reference answer? |
| Task accuracy / F1 | Does pruning hurt correctness? |
| Calibration | Do lower-CCI sets correspond to more confident decisions? |

### Human Evaluation (Planned)

If feasible, collect ratings from annotators or domain experts on:
- Perceived clarity
- Decision ease
- Overload rating
- Trust in the recommendation

### Main Success Criterion

> Lower CCI with no major loss in task quality, and ideally improved user clarity.

***

## Publishable Claims

This project targets a publishable contribution if it can demonstrate:

1. LLM-generated option sets have **measurable cognitive complexity** via a formal CCI.
2. The CCI **predicts or approximates** user decision difficulty better than set size alone.
3. A **complexity-aware pruning method** reduces overload without sacrificing task quality.
4. The framework **generalises across tasks and domains**, including high-stakes settings.

**Target venues:** ACL / EMNLP 2026–2027, or SIGCHI / CSCW for the human-centred version.

***

## What differentiates this from existing work?

| Related work | Their focus | Our focus |
|---|---|---|
| CLAI (Zhang, 2025) | LLM token efficiency (model-side cognitive load) | **User-side choice overload** in generated option sets |
| NeurIPS 2024 Decision Framework | Evaluating LLM decision-making under uncertainty | **Controlling** complexity of LLM outputs for users |
| Diverse beam search | Maximising output diversity | **Minimising cognitive cost** while preserving quality |
| Top-k sampling | Controlling output quantity | **Formal metric** for option-set complexity |

***

## Current Status

| Component | Status |
|---|---|
| Research framing and RQs | ✅ Complete |
| CCI formal definition | ✅ Defined (see above) |
| Theoretical basis | ✅ Complete |
| Demo / prototype notebook | ✅ Exists (see `/demo`) |
| CCI scorer implementation | 🔲 Not yet implemented |
| Dataset loaders | 🔲 Not yet implemented |
| Baseline comparison scripts | 🔲 Not yet implemented |
| Evaluation pipeline | 🔲 Not yet implemented |
| Human annotation protocol | 🔲 Not yet planned |
| Paper draft | 🔲 Not yet started |

***

## Next Steps

1. Implement the first version of the CCI scorer (sentence-transformers for similarity, entropy from model logits)
2. Run a small pilot on AmbigNQ or ASQA to validate the index
3. Add baseline comparison scripts
4. Design the evaluation pipeline and ablation plan
5. Use clinical triage (ER-Reason) as one evaluation domain
6. Begin paper draft structure

***

## Repository Structure

```
choice-complexity-llm/
├── README.md                  ← this file
├── demo/                      ← prototype notebook (exists)
├── src/
│   ├── cci_scorer.py          ← 🔲 TODO: symbolic CCI module
│   ├── generator.py           ← 🔲 TODO: LLM option generator wrapper
│   ├── selector.py            ← 🔲 TODO: complexity-aware selector
│   └── evaluate.py            ← 🔲 TODO: evaluation pipeline
├── data/
│   └── README.md              ← 🔲 TODO: dataset download instructions
├── experiments/
│   └── pilot/                 ← 🔲 TODO: first pilot experiment
└── paper/                     ← 🔲 TODO: paper draft
```

***

## Citation

If you use or build on this work, please cite:

```bibtex
@misc{bagheri2026cci,
  title   = {Choice Complexity in LLMs: A Decision-Theoretic Framework for
             Controlling the Cognitive Cost of LLM-Generated Option Sets},
  author  = {Bagheri, Soroush},
  year    = {2026},
  url     = {https://github.com/soroushbagheri/choice-complexity-llm}
}
```

***

## License

MIT License. See `LICENSE` for details.
