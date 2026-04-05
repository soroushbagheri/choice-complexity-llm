# Choice Complexity in LLMs - 3-Month Research Roadmap

## Overview

This document outlines a focused 12-week plan for the first paper on Choice Complexity in LLMs.

The goal of this roadmap is not to solve the full long-term research agenda. The goal is to produce a strong and defensible first paper built around:

- one primary benchmark,
- clear baselines,
- one working CCI formulation,
- a small human evaluation,
- and a complete paper draft.

This roadmap is intentionally narrow so it remains achievable within about 3 months.

---

## Overall objective for the first paper

Deliver a first-paper submission showing that:

1. a set-level Choice Complexity Index can be computed for LLM-generated option sets,
2. complexity-aware selection improves the final option set compared with simple baselines,
3. these gains are not explained only by showing fewer options,
4. and human raters prefer the lower-complexity sets when quality is approximately preserved.

---

## Scope of the first paper

### In scope

- one main benchmark: AmbigNQ
- one clean LLM generation pipeline
- one CCI scorer
- several simple but strong baselines
- component ablations
- a small human study
- a complete paper draft

### Out of scope for this 3-month plan

- multiple benchmark families
- function calling benchmark
- RAG benchmark
- clinical or other high-stakes domains
- full theoretical analysis
- large-scale user study
- multiple-model benchmark suite

These are good extensions after the first paper, not before it.

---

## Dataset decision

The official first-paper dataset is:

- AmbigNQ

Why this is the right first choice:

- ambiguity is intrinsic to the task,
- multiple plausible answers are expected by construction,
- it is strong enough for publication but still manageable for a first end-to-end pipeline,
- and it avoids the extra complexity of long-form synthesis required by ASQA.

Implementation decision:

- start with the Hugging Face version of AmbigNQ,
- use the `light` version first for the initial loader and inspection,
- then scale to the fuller setup after the pipeline is stable.

---

## 12-week execution plan

## Month 1 - Build and validate the core pipeline

### Week 1 - Lock the formulation and dataset pipeline

Goals:
- freeze the first-paper research question and claim
- finalize AmbigNQ as the benchmark
- define the first CCI formulation
- define the evaluation protocol
- inspect the dataset structure

Deliverables:
- final benchmark decision
- final list of baselines
- fixed experimental protocol
- clean issue list and task breakdown
- first dataset inspection notebook or script

Exit criterion:
- no more expansion of scope before core implementation starts

### Week 2 - Implement the AmbigNQ loader and base generation pipeline

Goals:
- implement dataset loading
- parse question, annotations, and qaPairs cleanly
- implement candidate generation
- save intermediate outputs for analysis

Deliverables:
- working AmbigNQ loader
- working generator
- saved outputs for a small development split

Exit criterion:
- can run end-to-end on a small subset without manual intervention

### Week 3 - Implement the CCI scorer and baselines

Goals:
- implement the first CCI scorer
- support set size, redundancy, utility entropy, and top-option ambiguity
- implement confidence-ranked top-k
- implement diversity-based pruning
- implement random pruning
- implement set-size-only control
- ensure final shown set size can be held constant across methods

Deliverables:
- working scorer
- baseline implementations
- evaluation script for fixed-size comparisons

Exit criterion:
- all methods produce comparable final sets for the same inputs

### Week 4 - Pilot experiment and metric sanity check

Goals:
- run a pilot on a manageable AmbigNQ subset
- inspect whether CCI behaves sensibly
- check whether each term contributes meaningful signal
- identify broken assumptions early

Deliverables:
- pilot results
- first plots
- note on whether the current CCI formulation is stable

Decision point:
- if the ambiguity term is unstable or unhelpful, revise it now
- if CCI adds no value beyond set size, revise the formulation before scaling up

---

## Month 2 - Run the main experiments and ablations

### Week 5 - Main benchmark run

Goals:
- run the full benchmark or a substantial representative subset
- collect outputs for all baselines and the complexity-aware selector
- record quality, coverage, and complexity metrics

Deliverables:
- main result tables
- raw outputs archived for analysis

Exit criterion:
- first complete benchmark pass finished successfully

### Week 6 - Error analysis and ablations

Goals:
- compare CCI against set size alone
- run leave-one-component-out ablations
- inspect failure cases
- identify when the method helps and when it does not

Deliverables:
- ablation tables
- error analysis notes
- shortlist of illustrative examples

Exit criterion:
- clear understanding of which components matter most

### Week 7 - Refine the selector and rerun targeted experiments

Goals:
- improve the selector if needed
- rerun only the targeted experiments that matter
- avoid endless iteration or overfitting to noise

Deliverables:
- final experiment configuration
- final automatic evaluation tables

Exit criterion:
- automatic results are stable enough to support writing

### Week 8 - Prepare the human evaluation

Goals:
- design a small pairwise comparison study
- prepare evaluation items from actual model outputs
- define clarity, ease-of-choice, and overload questions
- check whether a lightweight ethics or advisor review is needed

Deliverables:
- human-study protocol
- evaluation form
- selected example pairs

Exit criterion:
- study materials ready for data collection

---

## Month 3 - Human study and paper writing

### Week 9 - Run the human study

Goals:
- collect ratings on selected output pairs
- compare complexity-aware outputs against strong baselines
- keep the study focused and manageable

Deliverables:
- completed response set
- cleaned study data

Exit criterion:
- enough responses collected for a meaningful small-scale analysis

### Week 10 - Analyze human results and finalize figures

Goals:
- analyze study results
- connect human findings to automatic metrics
- finalize tables, figures, and example cases

Deliverables:
- final plots
- final result summaries
- final interpretation notes

Exit criterion:
- empirical story is complete

### Week 11 - Draft the paper

Goals:
- write introduction and related work
- write method section
- write experiments and results
- write limitations and future work

Deliverables:
- full first draft

Exit criterion:
- complete draft exists, even if rough

### Week 12 - Revise and prepare submission

Goals:
- revise for clarity and reviewer defensibility
- tighten contribution claims
- verify tables, figures, and references
- prepare submission package

Deliverables:
- polished paper draft
- appendix or supplementary notes if needed
- venue-ready submission package

Exit criterion:
- paper is ready for submission or advisor review

---

## Key milestones

### Milestone 1 - End of Week 4
Core pipeline works on AmbigNQ and the CCI formulation looks sensible.

### Milestone 2 - End of Week 8
Main automatic experiments and ablations are complete.

### Milestone 3 - End of Week 10
Human study is complete and integrated into the empirical story.

### Milestone 4 - End of Week 12
A full first-paper draft is ready for submission or final review.

---

## Go / no-go checkpoints

### Checkpoint A - End of Week 4
Question:
Does the metric behave sensibly enough on AmbigNQ to justify the main experiment?

Go if:
- CCI is stable,
- the components are interpretable,
- and it appears more meaningful than set size alone.

If not:
- simplify the formulation,
- reduce the claim,
- and continue with a cleaner metric.

### Checkpoint B - End of Week 6
Question:
Does the complexity-aware selector show promising automatic gains?

Go if:
- complexity is reduced,
- quality loss is small,
- and the method is competitive with simple baselines.

If not:
- reposition the paper around analysis and negative findings,
- or simplify the selector further.

### Checkpoint C - End of Week 10
Question:
Do humans show at least some preference for the lower-complexity outputs?

Go if:
- users find them clearer or easier without major quality concerns.

If not:
- keep the paper focused on automatic set-level control,
- and present the human evidence as mixed or limited.

---

## Practical resource assumptions

### Compute
- API budget or model access sufficient for one benchmark and several reruns
- local or hosted environment for storing outputs and embeddings

### Human evaluation
- small and manageable study, not a large-scale HCI project
- enough participants for directional evidence, not necessarily a large formal user study

### Time commitment
This roadmap assumes focused work over about 12 weeks.

A reasonable interpretation is:
- full-time research over 3 months, or
- part-time but disciplined work with weekly milestones

---

## Risks and mitigation

### Risk 1 - Scope expansion
Mitigation:
- keep only AmbigNQ in scope for the first paper
- postpone extra domains and benchmarks until after the first draft

### Risk 2 - Metric instability
Mitigation:
- test the metric early in Week 4
- allow simplification of the ambiguity term if needed

### Risk 3 - Gains only come from fewer options
Mitigation:
- use constant-size comparisons as a core design principle
- include set-size-only control as a baseline

### Risk 4 - Human study is too ambitious
Mitigation:
- keep the study small
- use pairwise judgments on real outputs
- focus on clarity and ease rather than complex cognitive measures

### Risk 5 - Writing starts too late
Mitigation:
- draft notes from Week 5 onward
- begin related work and method notes before all experiments are finished

---

## Definition of success for this 3-month phase

A successful 3-month phase means:

- one benchmark completed,
- one stable CCI formulation evaluated,
- one set of meaningful baselines compared,
- one small human study completed,
- and one full paper draft ready.

That is already a strong result for a first paper.

---

## What comes after this roadmap

After the first paper, the project can expand to:

- ASQA as a longer-form ambiguity extension,
- recommendation benchmarks,
- RAG or function-calling settings,
- high-stakes domains such as clinical decision support,
- deeper theoretical analysis,
- and larger human studies.

Those should be treated as phase two, not as prerequisites for the first submission.

---

## Immediate next actions

1. implement the AmbigNQ loader
2. inspect and print 5 AmbigNQ examples cleanly
3. finalize the current CCI formula for implementation
4. implement the base scorer and baseline selectors
5. run the first pilot by the end of Week 4

---

**Last updated:** April 2, 2026  
**Current status:** focused 3-month first-paper plan with AmbigNQ  
**Planning horizon:** 12 weeks
