# Decision-Theoretic Choice Complexity in LLMs - Project Roadmap

## Overview
This document outlines the complete development and publication roadmap for the Choice Complexity research project.

## Project Timeline (January - August 2026)

```
Jan 2026  |  Feb 2026  |  Mar 2026  |  Apr 2026  |  May 2026  |  Jun 2026  |  Jul 2026  |  Aug 2026
    |          |          |          |          |          |          |          |
    ├─ #1 Real LLM Integration
    |          ├─ #2 Function Calling Benchmark
    |          |          ├─ #4 Human Evaluation Study
    ├──── #7 Write Intro/Related Work
               |          ├─ #8 Write Method Section
               ├─ #5,#6 Ablation Studies
                         |          ├─ #9 Write Results
                         ├─ #14 Venue Decision
                                    |          ├─ EMNLP Submission
                                               |          ├─ Workshop Backup
                                                          |          ├─ Revisions
```

## Phase 1: Publication-Ready Research (Weeks 1-24)

### 🔴 Critical Path (Must Complete)

#### Month 1-2: Real LLM Integration & Benchmarks
- **#1: Integrate Real LLM APIs** → [Issue #1](https://github.com/soroushbagheri/choice-complexity-llm/issues/1)
  - Week 1-2: GPT-4 and Claude integration
  - Week 2-3: Run on 100 synthetic problems
  - Week 3-4: Validate CCI/ILDC correlations

- **#2: Function Calling Benchmark** → [Issue #2](https://github.com/soroushbagheri/choice-complexity-llm/issues/2)
  - Week 3-5: Berkeley benchmark setup
  - Week 5-7: Run all controller strategies
  - Week 7-8: Analysis and comparison

#### Month 2-3: Human Evaluation
- **#4: Human Evaluation Study** → [Issue #4](https://github.com/soroushbagheri/choice-complexity-llm/issues/4)
  - Week 5-6: Study design + IRB (if needed)
  - Week 7-8: Participant recruitment
  - Week 9-10: Data collection
  - Week 11-12: Analysis

#### Month 3-4: Ablations & Paper Writing
- **#5: CCI Ablations** → [Issue #5](https://github.com/soroushbagheri/choice-complexity-llm/issues/5)
  - Week 9-10: Feature importance studies
  - Week 10-11: Weight optimization

- **#6: ILDC Ablations** → [Issue #6](https://github.com/soroushbagheri/choice-complexity-llm/issues/6)
  - Week 10-11: Formulation comparison
  - Week 11-12: Efficiency analysis

- **#7: Introduction & Related Work** → [Issue #7](https://github.com/soroushbagheri/choice-complexity-llm/issues/7)
  - Week 1-4: Can start in parallel
  - Draft and revise with advisor

- **#8: Method Section** → [Issue #8](https://github.com/soroushbagheri/choice-complexity-llm/issues/8)
  - Week 9-12: After experiments are defined

#### Month 5-6: Results & Submission
- **#9: Results Section** → [Issue #9](https://github.com/soroushbagheri/choice-complexity-llm/issues/9)
  - Week 13-16: Comprehensive results
  - Week 16-18: Analysis and discussion

- **#14: Venue Selection & Submission** → [Issue #14](https://github.com/soroushbagheri/choice-complexity-llm/issues/14)
  - Week 12: Decision point (EMNLP vs Workshop)
  - Week 20-22: Final polishing
  - Week 24: **EMNLP 2026 Submission** (June deadline)

### 🟡 Supporting Tasks (Important)

- **#3: RAG Benchmark** → [Issue #3](https://github.com/soroushbagheri/choice-complexity-llm/issues/3)
  - Optional but strengthens paper
  - Can substitute for #2 if function calling fails

- **#10: Documentation** → [Issue #10](https://github.com/soroushbagheri/choice-complexity-llm/issues/10)
  - Ongoing throughout project
  - Finalize before paper submission

- **#15: Funding Applications** → [Issue #15](https://github.com/soroushbagheri/choice-complexity-llm/issues/15)
  - Parallel track
  - Submit 2-3 applications by April

### 🟢 Optional Tasks (Nice to Have)

- **#11: Unit Tests** → [Issue #11](https://github.com/soroushbagheri/choice-complexity-llm/issues/11)
  - Good engineering practice
  - Not critical for publication

## Phase 2: Extended Research (Month 7+)

### Post-Publication Work

- **#12: High-Stakes Domain Application** → [Issue #12](https://github.com/soroushbagheri/choice-complexity-llm/issues/12)
  - After Phase 1 publication accepted
  - 6-12 month project
  - Requires industry partner

- **#13: Theoretical Analysis** → [Issue #13](https://github.com/soroushbagheri/choice-complexity-llm/issues/13)
  - PhD-level work
  - Could be separate paper
  - Seek theory collaborators

## Key Milestones & Decision Points

### Milestone 1: Real LLM Validation (End of February)
**Go/No-Go Decision**: Do real LLMs show same patterns as synthetic?
- ✅ **GO**: CCI correlates with errors, ILDC detects instability → Continue to benchmarks
- ❌ **NO-GO**: No correlation → Pivot to pure synthetic study or revisit metrics

### Milestone 2: Benchmark Results (End of March)
**Go/No-Go Decision**: Does two-tier beat baselines on real tasks?
- ✅ **GO**: Significant improvements → Target EMNLP main or findings
- ⚠️ **PARTIAL**: Modest improvements → Target EMNLP findings or workshop
- ❌ **NO-GO**: No improvement → Target workshop, focus on analysis

### Milestone 3: Human Evaluation (Mid-April)
**Go/No-Go Decision**: Do humans prefer controller outputs?
- ✅ **GO**: Significant preference + lower cognitive load → Major paper strength
- ⚠️ **PARTIAL**: Preference but not load → Still publishable, discuss limitations
- ❌ **NO-GO**: No preference → Reframe as optimization problem, not UX

### Milestone 4: Venue Selection (Early May)
**Final Decision**: Where to submit?
- **Option A**: EMNLP Main (if all milestones strong GO)
- **Option B**: EMNLP Findings (if 2/3 milestones GO)
- **Option C**: NeurIPS Workshop (if 1/3 milestones GO or timeline slips)

## Resource Requirements

### Compute
- **LLM API Credits**: ~$500-1000 (GPT-4, Claude)
- **Local GPU** (optional): For open models

### Human Resources
- **Human evaluation participants**: $200-500 budget
- **Advisor time**: Weekly meetings
- **Collaborators** (optional): Statistics expert for analysis

### Time Commitment
- **Full-time equivalent**: 6 months (January - June 2026)
- **Part-time (50%)**: 12 months (if combining with PhD coursework)

## Risk Mitigation

### Risk 1: Real LLMs Don't Show Expected Patterns
**Mitigation**: 
- Have synthetic results as fallback
- Pivot to "synthetic LLM as model organism" framing
- Focus on framework contribution, not empirical findings

### Risk 2: Benchmark Results Underwhelming
**Mitigation**:
- Try multiple benchmarks (function calling, RAG, consumer choice)
- Find niche where it works well
- Honest discussion of when/where it helps

### Risk 3: Human Evaluation Doesn't Show Preference
**Mitigation**:
- Reframe as efficiency optimization (fewer tokens, faster)
- Target systems/optimization community instead of HCI
- Emphasize computational benefits over user experience

### Risk 4: Timeline Slips
**Mitigation**:
- Built-in 2-week buffer before each deadline
- Workshop backup plan (shorter format, easier acceptance)
- Can always submit to next cycle (EMNLP → NeurIPS → ICLR)

## Success Criteria

### Minimum Viable Publication (Workshop)
- Real LLM experiments on 1 benchmark
- Synthetic dataset results
- Clear positioning vs related work
- 4-6 page workshop paper

### Good Publication (Conference Findings)
- Real LLM experiments on 2 benchmarks
- Human evaluation with statistical significance
- Comprehensive ablations
- 8-page findings paper

### Excellent Publication (Conference Main)
- Real LLM experiments on 3+ benchmarks
- Large-scale human evaluation (100+ participants)
- Theoretical insight or strong empirical surprise
- Domain application case study
- 8-page main conference paper + appendix

## Next Immediate Actions (This Week)

1. **Start #1 (Real LLM Integration)**
   - Set up OpenAI API key
   - Modify `llm_adapter.py`
   - Run first test experiments

2. **Draft #7 (Introduction)**
   - 1-page motivation
   - Position against SITAlign, CLAI

3. **Plan #4 (Human Evaluation)**
   - Draft survey questions
   - Check if IRB needed
   - Estimate budget

4. **Decide on #14 (Venue)**
   - Review EMNLP 2026 call for papers
   - Mark deadline on calendar (early June)

## Project Management

### Weekly Rhythm
- **Monday**: Plan week, review GitHub issues
- **Wednesday**: Mid-week check-in, unblock issues
- **Friday**: Week review, update issue status, plan next week

### Monthly Rhythm
- **Month end**: Milestone review
- **Go/No-Go decision** if applicable
- **Adjust timeline** based on progress

### Communication
- **GitHub Issues**: All tasks and progress
- **This file**: Strategic overview
- **Weekly meetings**: Sync with advisor/collaborators

---

**Last Updated**: January 24, 2026  
**Status**: Phase 1 - Week 0 (Planning Complete, Ready to Execute)  
**Next Review**: February 1, 2026
