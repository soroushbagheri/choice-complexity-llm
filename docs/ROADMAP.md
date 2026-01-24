# Project Roadmap: Decision-Theoretic Choice Complexity in LLMs

**Version:** 1.0  
**Last Updated:** January 24, 2026  
**Target Publication:** EMNLP 2026 or NeurIPS Workshop 2026

---

## Project Vision

Build and validate a two-tier framework that measures and regulates choice complexity in LLM decision-making, demonstrating improved decision quality, consistency, and usability compared to baseline approaches.

**Success Metrics:**
- ✅ Publication at tier-1 conference or top workshop
- ✅ 20%+ improvement in decision consistency (volatility reduction)
- ✅ Maintain 90%+ accuracy while reducing cognitive load by 40%+
- ✅ Open-source tool with 100+ GitHub stars
- ✅ (Stretch) Industry partnership or deployment

---

## Phase 1: Foundation & Validation (Weeks 1-12) ⏰ CURRENT PHASE

**Goal:** Transform prototype into publication-ready research with real LLM validation.

### Week 1-2: Real LLM Integration 🔴 HIGH PRIORITY

**Tasks:**
- [ ] Implement OpenAI API integration in `llm_adapter.py`
  - GPT-4-turbo, GPT-4o
  - Support for logprobs extraction (for confidence)
  - Rate limiting and error handling
- [ ] Add Anthropic Claude integration
  - Claude-3.5-Sonnet
  - Claude-3-Opus
- [ ] Test on pilot dataset (20 samples)
  - Verify CCI computation correctness
  - Validate ILDC calculation from real outputs
  - Debug any API issues

**Deliverables:**
- ✅ Working `llm_adapter.py` with GPT-4 and Claude
- ✅ Pilot results CSV with 20 samples × 6 strategies
- ✅ Cost estimation for full experiments (~$50-200)

**Owner:** You  
**Blockers:** API access, budget approval

---

### Week 3-4: Benchmark Testing 🟡 MEDIUM PRIORITY

**Tasks:**
- [ ] **Berkeley Function Calling Leaderboard**
  - Download benchmark dataset
  - Adapt to choice complexity framework
  - Run all 6 controller strategies
  - Compare accuracy, latency, token usage
  
- [ ] **RAG Benchmark (MS MARCO or BEIR)**
  - Use existing retrieval results as options
  - Test top-5, top-10, top-20, top-50 scenarios
  - Measure answer quality with/without controller
  
- [ ] **Consumer Choice Benchmark**
  - Create realistic product scenarios (laptops, hotels)
  - Define ground truth based on user profiles
  - Test with multiple user preference configurations

**Deliverables:**
- ✅ Benchmark results tables for paper
- ✅ Comparison with baseline methods
- ✅ Statistical significance tests (t-tests, Wilcoxon)

**Owner:** You  
**Dependencies:** Week 1-2 completion

---

### Week 5-6: Human Evaluation Study 🟢 CRITICAL FOR PUBLICATION

**Tasks:**
- [ ] **Study Design**
  - IRB exemption application (if needed)
  - Create survey instrument (Qualtrics/Google Forms)
  - Define evaluation criteria:
    - Preference (A vs B)
    - Cognitive load (7-point Likert scale)
    - Perceived decision quality
    - Time to decision
  
- [ ] **Participant Recruitment**
  - Target: 50-100 participants
  - Platforms: Prolific, MTurk, or university panel
  - Screening criteria: English fluency, 18+
  - Compensation: $10-15 per 20 minutes
  
- [ ] **Data Collection**
  - Show 10 choice scenarios per participant
  - Randomize order (within-subject design)
  - Counterbalance conditions
  - Collect demographics
  
- [ ] **Analysis**
  - Preference rates (binomial test)
  - Cognitive load scores (paired t-test)
  - Time-to-decision (Kaplan-Meier analysis)
  - Qualitative feedback coding

**Deliverables:**
- ✅ Human evaluation section in paper (1-2 pages)
- ✅ Statistical validation of user preference
- ✅ Supplementary materials with full survey

**Owner:** You (can help with study design)  
**Budget:** $500-1500 for participants  
**Timeline:** 2 weeks (1 week prep, 1 week collection/analysis)

---

### Week 7: Ablation Studies & Analysis 🟡 MEDIUM PRIORITY

**Tasks:**
- [ ] **CCI Weight Ablation**
  - Test all 32 combinations of on/off features
  - Find optimal weights via grid search
  - Report sensitivity analysis
  
- [ ] **ILDC Component Ablation**
  - Volatility only
  - Confidence only
  - Disagreement only
  - All combinations
  
- [ ] **Controller Strategy Variants**
  - Different thresholds (0.3, 0.5, 0.7)
  - Hierarchical presentation (top-3 + clusters)
  - Clarifying question generation
  
- [ ] **Correlation Analysis**
  - CCI vs ILDC scatter plots
  - CCI vs accuracy
  - ILDC vs volatility
  - Feature importance (SHAP values)

**Deliverables:**
- ✅ Ablation tables and figures for paper
- ✅ Sensitivity analysis section
- ✅ Feature importance visualization

**Owner:** You  
**Dependencies:** Week 3-4 benchmark results

---

### Week 8-10: Paper Writing 📝 CRITICAL

**Tasks:**
- [ ] **Title & Abstract** (Week 8)
  - Draft 3 title options
  - Write 250-word abstract
  - Identify key contributions (3-4 bullets)
  
- [ ] **Introduction** (Week 8)
  - Motivation: Choice overload in AI systems
  - Problem statement: LLM decision instability
  - Gap in literature: No two-tier approach
  - Contributions: CCI, ILDC, two-tier controller
  
- [ ] **Related Work** (Week 8)
  - Expand existing `REFERENCES.md`
  - Position against 2024-2025 work:
    - SITAlign (Chehade et al., 2025)
    - CLAI (Zhang et al., 2025)
    - Behavioral econ in LLMs (Jia et al., 2024)
    - Consumer choice experiments (Cherep et al., 2025)
  - Add recent RAG optimization work
  - Add LLM uncertainty quantification papers
  
- [ ] **Method** (Week 9)
  - Two-tier architecture diagram
  - CCI formulation with equations
  - ILDC computation algorithm
  - Controller policy pseudocode
  
- [ ] **Experiments** (Week 9)
  - Dataset description
  - Experimental setup
  - Baselines and evaluation metrics
  
- [ ] **Results** (Week 9-10)
  - Main results tables
  - Benchmark comparisons
  - Human evaluation findings
  - Ablation studies
  - Qualitative analysis
  
- [ ] **Discussion & Limitations** (Week 10)
  - When does two-tier win?
  - Computational cost trade-offs
  - Limitations: chicken-egg problem, latency
  - Ethical considerations
  
- [ ] **Conclusion & Future Work** (Week 10)
  - Summary of contributions
  - Broader impact
  - Future directions

**Deliverables:**
- ✅ Complete paper draft (8 pages EMNLP format)
- ✅ All figures and tables publication-ready
- ✅ Supplementary materials (code, datasets, survey)

**Owner:** You + Advisor/Collaborator feedback  
**Dependencies:** All previous weeks

---

### Week 11-12: Submission & Rebuttal Prep 🚀

**Tasks:**
- [ ] **Internal Review**
  - Advisor/collaborator review
  - Lab presentation
  - Incorporate feedback
  
- [ ] **Submission Preparation**
  - Check formatting (EMNLP template)
  - Anonymize paper (remove author names, funding)
  - Prepare code release (clean GitHub repo)
  - Write reproducibility checklist
  
- [ ] **Target Venues**
  - **Primary:** EMNLP 2026 (deadline ~May 2026)
  - **Secondary:** ACL 2026 Findings (deadline ~February 2026)
  - **Backup:** NeurIPS Workshop on Decision-Making (deadline ~September 2026)
  
- [ ] **Rebuttal Materials**
  - Anticipate reviewer concerns
  - Prepare additional experiments if needed
  - Draft rebuttal template

**Deliverables:**
- ✅ Submitted paper
- ✅ Public GitHub repo with reproducible code
- ✅ Rebuttal preparation document

**Owner:** You  
**Deadline:** Week 12 submission

---

## Phase 2: Publication & Dissemination (Weeks 13-24)

**Goal:** Get paper accepted, present at conference, and build community impact.

### Week 13-16: Review Period & Revision

**Tasks:**
- [ ] Respond to reviewers (if R&R)
- [ ] Run additional experiments as requested
- [ ] Revise paper based on feedback
- [ ] Resubmit or submit to backup venue

**Deliverables:**
- ✅ Revised paper
- ✅ Detailed response to reviewers

---

### Week 17-20: Conference Preparation

**Tasks:**
- [ ] Create poster (if accepted)
- [ ] Prepare 15-minute talk slides
- [ ] Record video presentation
- [ ] Register for conference

**Deliverables:**
- ✅ Conference materials (poster, slides, video)

---

### Week 21-24: Dissemination & Outreach

**Tasks:**
- [ ] **Blog Post**
  - Write accessible explanation (like your "simple terms" summary)
  - Post on Medium, personal website
  - Share on Twitter/X, LinkedIn
  
- [ ] **GitHub Release**
  - Clean up code and documentation
  - Add installation guide and examples
  - Create demo Colab notebook
  - Announce on Reddit (r/MachineLearning)
  
- [ ] **Academic Outreach**
  - Email researchers in related areas
  - Present at university seminars
  - Apply to give talks at other institutions

**Deliverables:**
- ✅ 500+ blog post views
- ✅ 50+ GitHub stars
- ✅ 3+ seminar invitations

---

## Phase 3: Extension & Impact (Weeks 25-52) 🔮 FUTURE

**Goal:** Extend research impact through domain applications and partnerships.

### High-Stakes Domain Application

**Options:**
1. **Medical Decision Support**
   - Partner with hospital or medical AI company
   - Choose treatment options, drug selection
   - Validate with clinicians
   
2. **Legal Research Assistant**
   - Select relevant case law from 100+ candidates
   - Partner with law firm or LegalTech startup
   - Validate with lawyers
   
3. **Financial Portfolio Selection**
   - Investment option filtering
   - Partner with fintech company
   - Validate with financial advisors

**Tasks:**
- [ ] Identify domain partner
- [ ] Collect domain-specific dataset
- [ ] Adapt framework to domain constraints
- [ ] Run pilot study with domain experts
- [ ] Write domain-specific paper (e.g., for medical informatics venue)

**Timeline:** 6-9 months  
**Deliverables:** Second paper, potential deployment

---

### Theoretical Analysis

**Tasks:**
- [ ] Formalize problem as PAC-learning framework
- [ ] Prove sample complexity bounds
- [ ] Characterize conditions when two-tier provably beats one-tier
- [ ] Submit to theory venue (COLT, ALT)

**Timeline:** 6-12 months  
**Deliverables:** Theory paper, journal submission

---

### Industrial Partnership

**Options:**
1. **RAG Platforms (LlamaIndex, LangChain)**
   - Integrate as optional module
   - Contribute to open-source
   
2. **AI Startups (Perplexity, You.com, Glean)**
   - Pilot in production RAG systems
   - Measure real-world impact
   
3. **Enterprise AI (IBM Watson, Google Cloud AI)**
   - Licensing or collaboration agreement

**Tasks:**
- [ ] Prepare partnership pitch deck
- [ ] Reach out to engineering/research teams
- [ ] Negotiate terms (open-source contribution vs licensing)
- [ ] Deploy in production pilot

**Timeline:** 3-6 months  
**Deliverables:** Industry case study, potential funding

---

## Risk Management

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Real LLM results don't match synthetic | High | Medium | Run early pilot (Week 1-2) to validate |
| Human evaluation shows no preference | High | Low | Pre-test survey with 5 participants |
| Benchmark performance not competitive | Medium | Medium | Focus on specialized domains (RAG, tool-calling) |
| Paper rejected from EMNLP | Medium | Medium | Have backup venues ready (ACL, workshops) |
| Budget constraints (API costs, participants) | Medium | Low | Seek small grants, use cheaper models (GPT-3.5) |
| Time constraints (PhD deadlines) | High | Medium | Prioritize Phase 1 only, defer Phase 3 |

---

## Resource Requirements

### Budget
- **API Costs:** $200-500 (GPT-4 experiments)
- **Human Evaluation:** $500-1500 (participants)
- **Conference Registration:** $500-800
- **Travel (if accepted):** $1000-2000
- **Total:** $2200-4800

### Time Commitment
- **Weeks 1-12:** 20-30 hours/week (full-time focus)
- **Weeks 13-24:** 10-15 hours/week (revision + dissemination)
- **Weeks 25-52:** 5-10 hours/week (extensions, optional)

### Collaborators (Optional)
- **Advisor:** Paper reviews, feedback
- **Co-author:** Help with human evaluation or domain application
- **Engineer:** Assist with API integration (Week 1-2)

---

## Success Criteria

### Minimum Viable Publication (Must-Have)
- ✅ Real LLM experiments on 2+ models
- ✅ Benchmark validation on 1+ standard dataset
- ✅ Human evaluation showing preference (p < 0.05)
- ✅ Ablation studies demonstrating two-tier superiority

### Strong Publication (Nice-to-Have)
- ✅ Theoretical analysis of when two-tier wins
- ✅ Deployment in real application
- ✅ 3+ benchmark datasets
- ✅ 100+ human evaluation participants

### High-Impact Publication (Stretch)
- ✅ Industry partnership validation
- ✅ Open-source tool with active users
- ✅ Oral presentation at conference (top 5%)
- ✅ Follow-up journal paper

---

## Key Milestones & Checkpoints

| Milestone | Target Date | Checkpoint |
|-----------|-------------|------------|
| ✅ Prototype Complete | Jan 24, 2026 | **DONE** |
| 🔴 Real LLM Integration | Feb 7, 2026 | Week 2 review |
| 🟡 Benchmark Results | Feb 21, 2026 | Week 4 review |
| 🟢 Human Evaluation Complete | Mar 7, 2026 | Week 6 review |
| 🔵 Paper Draft Complete | Apr 4, 2026 | Week 10 review |
| 🚀 EMNLP Submission | May 1, 2026 | Week 12 deadline |
| 📢 Conference Presentation | Nov 2026 | If accepted |

---

## Next Immediate Actions (This Week)

### Priority 1: Real LLM Integration 🔴
1. **Monday:** Set up OpenAI API key, test basic call
2. **Tuesday:** Implement `llm_adapter.py` for GPT-4
3. **Wednesday:** Run pilot with 20 samples, debug issues
4. **Thursday:** Add Claude integration
5. **Friday:** Compare synthetic vs real results, document differences

### Priority 2: Experiment Planning 🟡
1. Select benchmark datasets (Berkeley FCL, MS MARCO)
2. Download and preprocess data
3. Write experiment scripts
4. Estimate API costs

### Priority 3: Paper Outline 📝
1. Draft 3 title options
2. Write 250-word abstract
3. Create section outline
4. Identify key figures needed

---

## Contact & Collaboration

**Project Lead:** Soroush Bagheri  
**GitHub:** [choice-complexity-llm](https://github.com/soroushbagheri/choice-complexity-llm)  
**Advisor:** [Add advisor name/email]  
**Slack/Discord:** [Add communication channel]

**Looking for collaborators on:**
- Human evaluation study design
- Domain application (medical, legal, finance)
- Theoretical analysis
- Industry partnerships

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 24, 2026 | Initial roadmap created |

---

**Let's build something impactful! 🚀**
