# Choice Complexity in LLMs

A decision-theoretic framework for controlling the cognitive cost of LLM-generated option sets

---

## What problem does this project study?

Large language models often generate multiple candidate outputs: recommendations, next steps, alternative answers, or possible actions. More options can look helpful, but beyond a certain point they can make the final choice harder for the user.

This repository studies that problem directly.

The central question is not only whether an LLM can generate good options, but whether it can generate a set of options that is easier for a human to choose from.

In other words:

> Can we formally measure the complexity of an LLM-generated option set, and can we use that signal to control what the user finally sees?

This project treats that as a human-centered inference-time control problem.

---

## Main idea

The framework has three parts:

1. An LLM generates a candidate set of options.
2. A symbolic scorer estimates how difficult that set is to choose from.
3. A selector prunes or reranks the candidate set to satisfy a target complexity budget before the final options are shown to the user.

This makes the contribution different from standard top-k generation, confidence ranking, or diversity-only decoding. The objective here is not just more answers, or more diverse answers, but lower human decision burden while preserving usefulness.

---

## What this repository aims to contribute

This project aims to contribute:

- a formal and interpretable set-level metric for choice complexity,
- a complexity-aware pruning or reranking mechanism for LLM outputs,
- an empirical evaluation showing whether complexity-aware selection helps compared with simple baselines,
- and a human-centered framing for multi-option LLM outputs in NLP tasks.

The intended first paper is deliberately narrow and testable. It is not framed as a universal theory of human choice. It is framed as an interpretable control method for multi-option LLM outputs.

---

## First-paper scope

The first paper is planned around one clean benchmark setting before expanding to broader domains.

### Primary benchmark

- AmbigNQ
- Task type: ambiguous open-domain question answering with multiple plausible answers or interpretations
- Why this setting: ambiguity is intrinsic to the dataset, which makes it a strong and publishable first testbed for set-level complexity control without adding the extra burden of long-form synthesis

### Core experimental design

For each input:

1. Generate a candidate set with the base LLM.
2. Compute a Choice Complexity Index over the set.
3. Apply a complexity-aware selector.
4. Compare the final fixed-size set against strong but simple baselines.

### Critical design principle

The most important evaluation constraint is:

> hold the final number of shown options constant whenever possible

This matters because otherwise a reviewer can always argue that the gains came from merely showing fewer options.

### Baselines for the first paper

- confidence-ranked top-k
- diversity-based pruning
- random pruning
- set-size-only control
- unconstrained candidate set
- single-answer prompting

### Success condition for the first paper

A strong first-paper result would be:

- lower measured complexity,
- equal or better perceived clarity,
- and minimal loss in task quality or answer coverage.

---

## Research questions

- RQ1. Can we define a formal, computable Choice Complexity Index (CCI) for LLM-generated option sets?
- RQ2. Does CCI correlate with decision-difficulty proxies better than simple baselines such as set size alone?
- RQ3. Can a complexity-aware selector reduce option-set burden without materially harming task quality?
- RQ4. Do human users prefer lower-CCI final sets when quality is held approximately constant?

---

## Why this may be novel

The main novelty is the target variable.

Most related work focuses on one of the following:

- answer quality,
- uncertainty,
- calibration,
- decoding diversity,
- or model-side reasoning cost.

This project focuses on something different:

> user-side choice complexity in the final option set

That shift is important. The project is not mainly about whether the model is uncertain internally. It is about whether the final set presented to a user is unnecessarily hard to choose from.

A second source of novelty is that the metric is set-level and interpretable. Instead of a black-box learned score, the current formulation is decomposable into understandable components that can be ablated and challenged individually.

A third source of novelty is that the control mechanism is inference-time and modular. The framework can sit on top of an existing LLM generator rather than requiring a new model architecture.

---

## Proposed method

```text
AmbigNQ question
    │
    ▼
LLM option generator
    │
    ▼
Candidate answer set S
    │
    ▼
Symbolic CCI scorer
    │
    ▼
Complexity-aware selector
    │
    ▼
Final fixed-size answer set
```

The selector may prune, rerank, or cluster options depending on the final implementation.

---

## Choice Complexity Index (CCI)

A current working formulation is:

CCI(S) = w1 * N(S) + w2 * H_u(S) + w3 * R(S) + w4 * A(S)

Where:

- N(S): set size
- H_u(S): utility entropy or uncertainty over option utilities
- R(S): semantic redundancy within the option set
- A(S): top-option ambiguity

### Why use top-option ambiguity instead of a simple regret-gap term?

One reviewer-style concern with the earlier formulation is that a large gap between the best option and the average option may sometimes make the decision easier, not harder.

To address that, this repository now treats the fourth term more carefully as top-option ambiguity rather than a naive regret-gap signal.

In practice, A(S) can be implemented using one of the following:

- inverse margin between the best and second-best option,
- concentration of utility mass among near-top options,
- or another measure of how hard it is to distinguish the leading candidates.

This makes the index more consistent with the intuition that burden increases when the top options are hard to tell apart.

### Interpretation

Low CCI usually means:

- fewer options,
- clearer separation,
- less overlap,
- easier final choice.

High CCI usually means:

- more competing options,
- more redundancy,
- flatter preference structure,
- harder final choice.

### Important note

The exact formulation of CCI is still an active research object. Part of the empirical contribution of this project will be to test whether each component really helps and whether the full index predicts difficulty better than simpler alternatives.

---

## Hypotheses

| ID | Hypothesis |
|---|---|
| H1 | Larger and more redundant option sets will tend to have higher decision burden |
| H2 | CCI will correlate with decision-difficulty proxies better than set size alone |
| H3 | Complexity-aware selection will reduce burden with limited loss in answer quality or coverage |
| H4 | Human raters will prefer lower-CCI final sets when final set size is controlled |

---

## Evaluation plan

### Automatic evaluation

The first automatic evaluation will focus on:

- CCI before and after selection
- semantic redundancy
- coverage of reference answers
- task accuracy or semantic correctness
- calibration or confidence alignment
- ablations over each CCI component

### Human evaluation

A small human study is planned for the first paper.

Target questions include:

- Which final set is easier to choose from?
- Which final set is clearer?
- Which final set feels less overwhelming?
- Does the lower-complexity set still feel useful enough?

If possible, the study will compare pairs of final sets while keeping the number of shown options fixed.

---

## What this project does not claim yet

This repository does not yet claim:

- a universal cognitive theory of human choice,
- a complete multi-domain benchmark result,
- or a final validated formulation of choice complexity.

The first goal is narrower:

> show that a structured set-level complexity signal can improve the presentation of multi-option LLM outputs on at least one real task

That claim is much more defensible for a first publication.

---

## Dataset

The official first-paper dataset is:

- AmbigNQ

Why AmbigNQ:

- ambiguity is built into the dataset itself,
- multiple plausible answers are expected by construction,
- it is scientifically strong without being too complicated,
- and it lets the first paper focus on candidate-set selection rather than long-form synthesis.

Planned dataset roles:

| Dataset | Task | Role |
|---|---|---|
| AmbigNQ | ambiguous open-domain QA | first-paper benchmark |
| Controlled vignette set | synthetic or semi-synthetic pilot | metric sanity check |
| ASQA | long-form ambiguity resolution | later extension |
| recommendation dataset | recommendation or ranked suggestion task | later extension |
| clinical triage / next-step recommendation | high-stakes decision support | later extension |

The clinical domain and long-form ambiguity datasets are intentionally positioned as later extensions rather than the center of the first paper.

### AmbigNQ loading plan

The recommended source for the first implementation is the Hugging Face dataset version of AmbigNQ.

Initial implementation target:

- load the `light` version first,
- inspect `id`, `question`, `annotations`, and `qaPairs`,
- then build the generation and fixed-size selection pipeline on top of that structure.

---

## Baselines

Planned baselines include:

| Baseline | Purpose |
|---|---|
| confidence-ranked top-k | strong simple baseline |
| diversity-based pruning | tests whether diversity alone is enough |
| random pruning | lower bound |
| set-size-only controller | tests whether CCI adds value beyond fewer options |
| unconstrained candidate set | no control |
| single-answer prompting | extreme low-complexity baseline |

---

## Theoretical grounding

The project is motivated by three broad traditions:

- work on choice overload and the paradox of choice,
- decision-theoretic and random-utility perspectives on hard choices,
- neurosymbolic AI, where an interpretable symbolic layer constrains or guides a neural model.

The intended contribution is operational rather than purely philosophical: translate these ideas into a measurable and testable LLM control framework.

---

## Current status

| Component | Status |
|---|---|
| problem framing | done |
| first-pass CCI formulation | done |
| reviewer-aware narrowing of first paper | done |
| first-paper dataset decision (AmbigNQ) | done |
| prototype / demo | exists |
| dataset loader | planned |
| CCI scorer implementation | planned |
| selector baselines | planned |
| evaluation pipeline | planned |
| human study protocol | planned |
| paper draft | planned |

---

## Immediate next steps

1. implement the AmbigNQ loader
2. inspect and print 5 AmbigNQ examples cleanly
3. implement the first CCI scorer
4. implement constant-size baseline comparisons
5. run a pilot on AmbigNQ
6. draft the first paper around AmbigNQ before expanding to other domains

---

## Repository structure

```text
choice-complexity-llm/
├── README.md
├── PROJECT_ROADMAP.md
├── notebooks/
├── src/
├── data/
├── experiments/
└── paper/
```

---

## License

MIT License
