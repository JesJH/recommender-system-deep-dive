# Business Value & A/B Testing

Building a great model is necessary but not sufficient. You need to:
1. Design a valid experiment to measure its impact
2. Translate metrics into language stakeholders understand
3. Know when a lift is real vs. noise

---

## A/B Test Design

### The Setup
- **Control**: existing recommendation system (or random / popularity baseline)
- **Treatment**: your new model
- **Unit of randomization**: user (not session — to avoid interference)
- **Exposure**: users are assigned on first visit and held for the test duration

### Key Questions Before Running
1. **What is the primary metric?** (e.g., CTR, revenue per session, conversion rate)
2. **What are the guardrail metrics?** (latency p99, error rate, refund rate — must not regress)
3. **What is the minimum detectable effect (MDE)?** How small a lift is worth detecting?
4. **How long to run?** Long enough to reach statistical power; long enough to avoid novelty effect.

### Sample Size Calculation

For a two-proportion z-test:

```
n = 2 × (z_α/2 + z_β)² × p(1-p) / (MDE)²
```

Where:
- `p` = baseline conversion rate
- `MDE` = minimum effect you want to detect (e.g., 0.5% lift)
- `z_α/2` = 1.96 for 95% confidence
- `z_β` = 0.84 for 80% power

### Common Pitfalls
- **Peeking**: checking results before the test completes inflates false positive rate
- **Multiple testing**: if you test 10 metrics, expect 1 false positive at α=0.10
- **Novelty effect**: users click more on anything new for the first few days — run long enough to wash it out
- **Interference**: users in the control group may interact with treatment users (social, inventory)
- **Simpson's paradox**: aggregate lift can hide losses in important subgroups

---

## Translating Model Metrics to Business Value

| Model Metric | Business Proxy | How to Connect |
|---|---|---|
| NDCG@10 ↑ | Better ranking → more relevant items shown | Correlate with CTR in A/B test |
| Coverage ↑ | More catalog sells → GMV diversification | Track long-tail item sales |
| Diversity ↑ | Users discover more → higher LTV | Measure repeat purchase rate |
| CTR ↑ | More clicks → more purchase intent | Direct, but watch for click-bait |
| Conversion ↑ | More purchases → revenue | Clearest business signal |

### Revenue Lift Calculation
```
Daily active users = 1,000,000
Baseline conversion = 3.0%
Measured lift = +0.3% absolute → 3.3% new conversion
Revenue per conversion = $50

Daily lift = 1,000,000 × 0.003 × $50 = $150,000/day
Annual = ~$55M
```
Even a 0.3% lift is significant at scale — this is why recommendation systems get engineering investment.

---

## What's in the Notebook

1. Simulate an A/B test dataset (synthetic control/treatment split)
2. Compute sample size for a given MDE and baseline rate
3. Analyze results: t-test / z-test, p-value, confidence interval
4. Check for novelty effect (first week vs. steady state)
5. Segment analysis: does the lift hold for new vs. returning users?
6. Build a revenue impact calculator

---

## Communicating Results

**To data/ML team**: NDCG, precision@K, coverage, statistical test results  
**To product managers**: CTR lift, conversion lift, # users impacted, revenue estimate  
**To executives**: Revenue impact, user retention improvement, competitive positioning  

Frame it as: "We ran a 2-week A/B test with 500K users per arm. The new model improved conversion by 0.4% (p=0.003, 95% CI: 0.2%–0.6%), which extrapolates to $X million in annual revenue."
