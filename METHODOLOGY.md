# METHODOLOGY — the mathematics & finance

Every formula the platform uses, with its derivation, assumptions, and financial
interpretation. Written for **model validation** — a quant or risk-model reviewer
should be able to check correctness against this doc and the test suite. Each
section names the implementing module and the test that pins it.

Notation: `Φ` = standard normal CDF, `Φ⁻¹` = its inverse (quantile), `Φ₂(·,·;ρ)`
= bivariate normal CDF with correlation ρ. `D_i` = default indicator of borrower
i (1 if default). PD_i = P(D_i = 1).

---

## 1. Expected loss (the Basel decomposition)

For borrower i:
```
EL_i = PD_i · EAD_i · LGD_i
```
- **PD** probability of default (from the PD model, `model_pd`).
- **EAD** exposure at default (outstanding/credit-line; today an income proxy or
  supplied column).
- **LGD** loss given default ∈ [0,1] (today a flat 0.45 or a supplied vector).

Portfolio EL = Σ_i EL_i (always additive). *Module:* `risk_adjusted_metrics.py`.
*Financial meaning:* the average loss you expect and should price/provision for.

---

## 2. Default correlation via copulas

Individual PDs are *marginals*. To get **joint** behaviour we need a copula — a
function that couples marginals with a dependence structure.

### 2.1 Gaussian copula / latent-variable view
Each borrower has a latent "asset value" `A_i ~ N(0,1)`; default occurs when it
falls below a threshold:
```
D_i = 1  ⇔  A_i ≤ Φ⁻¹(PD_i)
```
This reproduces the marginal exactly: P(A_i ≤ Φ⁻¹(PD_i)) = PD_i. Correlating the
`A_i` correlates the defaults. For a pair with asset correlation ρ:
```
P(D_i ∩ D_j) = Φ₂( Φ⁻¹(PD_i), Φ⁻¹(PD_j) ; ρ )
```
*Module:* `copula_model.py` (`gaussian`), `factor_copula.py`. *Validation:* the
bivariate-normal CDF is checked against `scipy.multivariate_normal` to ~1e-12
(test 34).

### 2.2 Clayton copula (lower-tail dependence)
```
C(u,v;θ) = (u⁻θ + v⁻θ − 1)^(−1/θ),   θ > 0
```
Lower-tail dependence λ_L = 2^(−1/θ) > 0: defaults **cluster in stress** (when
one is in the bad tail, the other likely is too). This is why Clayton is the
default for credit. *Module:* `copula_model.py`. Gumbel (upper tail) and Frank
(no tail) are also provided.

### 2.3 Student-t copula
Like Gaussian but with a shared χ² mixing variable (ν degrees of freedom),
giving **symmetric tail dependence** — heavier joint tails than Gaussian. The
bivariate-t CDF is computed as a χ²-scale-mixture of the Gaussian CDF:
```
T₂(h,k;ρ,ν) = ∫₀^∞ Φ₂( h√(s/ν), k√(s/ν) ; ρ ) · f_{χ²_ν}(s) ds
```
evaluated by Gauss-Laguerre quadrature reusing the fast Gaussian `Φ₂`. *Module:*
`factor_copula.py` (`student_t=True`). *Validation:* matches `scipy.multivariate_t`
to ~3e-4 and the t-factor simulation to ~6e-4 across seeds (test 34).

**Assumption / limitation:** the Gaussian factor copula has **no** tail
dependence — in deep stress it understates clustering; use the t variant when
that matters.

---

## 3. The Vasicek factor model (how we scale to 10M)

A dense n×n correlation is impossible at 10M. Instead, dependence comes from a
small number of **systematic factors** (single-factor here = the classic Vasicek
/ Basel IRB model):
```
A_i = √ρ_i · Y_{f(i)}  +  √(1 − ρ_i) · ε_i
```
- `Y_{f(i)}` standard-normal **systematic** factor of borrower i's group f(i)
  (geography, household, industry). Shared ⇒ correlation.
- `ε_i` standard-normal **idiosyncratic** shock, independent across borrowers.
- `ρ_i` ∈ [0,1) **asset correlation / factor loading** (Basel retail ≈ 0.03–0.16).

**Implied pairwise correlation** (no matrix needed):
```
corr(A_i, A_j) = √(ρ_i · ρ_j)   if f(i)=f(j),   else 0
```
Storage is **O(n)** loadings. *Module:* `factor_copula.py`. *Financial meaning:*
larger/fewer factors ⇒ stronger systematic clustering ⇒ fatter portfolio loss
tail. *Validation:* analytical block matches simulation; 2M-borrower scale test.

---

## 4. Multi-factor copula (geo ⟂ transfer, equally weighted)

Correlation has **two** independent sources the user weights equally: geography
and the money-transfer community. Generalize Vasicek to K factors:
```
A_i = Σ_k β_{i,k} · Y_{k, f_k(i)}  +  √(1 − Σ_k β_{i,k}²) · ε_i
```
with Y_{k,·} independent across dimensions k. **Implied correlation:**
```
corr(A_i, A_j) = Σ_k β_{i,k}·β_{j,k} · 1[f_k(i)=f_k(j) ≥ 0]
```
So sharing only geo ⇒ β_geo²; sharing both ⇒ β_geo²+β_transfer²; sharing none
⇒ 0. **Equal betas ⇒ equally important** dimensions.

**Constraint:** Σ_k β_{i,k}² < 1 (positive idiosyncratic variance) — enforced in
`fit`. Storage O(n·K). *Module:* `multi_factor_copula.py`. *Validation:* implied
correlation exact to 1e-9; block == simulation to ~7e-4; ordering both>geo>none
(test 39).

---

## 5. Loss covariance (the core risk object)

Default indicators are Bernoulli, so:
```
Cov(D_i, D_j) = P(D_i ∩ D_j) − PD_i·PD_j         (i ≠ j)
Var(D_i)      = PD_i·(1 − PD_i)                    (diagonal)
```
Scaling by the loss given default of each borrower (`w_i = EAD_i·LGD_i`):
```
LossCov[i,j] = w_i · Cov(D_i, D_j) · w_j
```
**Portfolio/segment loss variance** is the block-sum:
```
Var(Loss_S) = Σ_{i∈S} Σ_{j∈S} LossCov[i,j]
```
*Module:* `risk_adjusted_metrics.py`. *Validation:* equals brute-force simulation
to ≈0 relative error; block-on-demand == dense to 1e-9 (test 32). **This is
INV-6: never average per-borrower ratios — sum the covariance block.**

Joint probabilities respect the **Fréchet bounds**
`max(PD_i+PD_j−1, 0) ≤ P(D_i∩D_j) ≤ min(PD_i, PD_j)` (verified in audit).

---

## 6. The seven risk-adjusted metrics

Let EP = E[Profit] = Revenue − EL, Cap = capital (= k·EAD), σ_L0 =
√(Σ diag LossCov) (independence std), σ_L1 = √(block-sum LossCov) (copula std),
σ_L2 = simulated downside semideviation, rf = risk-free rate, h = hurdle rate.

| Metric | Formula | Reads as |
|---|---|---|
| `coefficient_of_variation` | σ_L0 / EL | riskiness, correlation-blind |
| `coefficient_of_variation_copula` | σ_L1 / EL | riskiness, **copula-aware** (inflates for contagious segments) |
| `raroc` | EP / Cap | return on regulatory capital (correlation-blind) |
| `sharpe_indep` | (EP − rf·Revenue) / σ_L0 | profit vs risk-free opportunity cost |
| `sortino_indep` | (EP − h·Cap) / σ_L0 | profit vs hurdle, downside = independence |
| `sortino_copula` | (EP − h·Cap) / σ_L1 | **primary**: profit vs hurdle, downside = copula |
| `sortino_simulated` | (EP − h·Cap) / σ_L2 | as above, downside from full simulated tail (3+-way clustering) |

*Module:* `risk_adjusted_metrics.py`. **Why both σ_L0 and σ_L1:** the gap between
the independence and copula denominators *is* the correlation penalty — a borrower
or cluster cheap on RAROC but expensive on `sortino_copula` is carrying hidden
contagion risk. That divergence is the early-warning signal (§9).

**Sign caveat (documented):** ratios with a profit numerator flip sign when
profit < 0; for pure riskiness ranking use the CoV metrics (always ≥ 0).

---

## 7. Diversification ratio

```
DR_S = ( Σ_{i∈S} σ_i ) / σ_portfolio,
   σ_i = √LossCov[i,i],   σ_portfolio = √(Σ_S Σ_S LossCov)
```
By the triangle inequality DR ≥ 1; = 1 only under perfect correlation (no
diversification). **Note the correct form:** numerator is the **sum of individual
stds**, not √(sum of variances). *Module:* `risk_adjusted_metrics.py`
(`diversification_ratio`). *Validation:* test 28/36, audit check 2.

---

## 8. Anchor / dependent contagion (якорный человек)

A *transfer community* (Louvain) can have a single person on whom the others
financially depend. **Anchor score** combines money-source dominance (share of
the community's inbound that this node feeds), being an **articulation point**
(removing it disconnects the cluster), and a star-shaped ego-net.

**Cluster fragility** = anchor_inbound_share × (1 − redundancy), redundancy =
how many alternative money sources dependents have. High ⇒ a cascade is likely.

**Anchor-contagion uplift** (the headline number) — the cluster's expected loss
*conditional on the anchor defaulting*. Under the Gaussian factor model:
```
PD_j | (anchor defaults) = P(D_j ∩ D_anchor) / PD_anchor
EL_cluster | anchor-default = Σ_j w_j · (PD_j | anchor defaults),  anchor at PD=1
uplift_ratio = EL_cluster|anchor-default  /  EL_cluster (unconditional)
```
A diversified cluster shows ≈1×; a fragile star shows ≫1× (demo: 2.6×).
*Modules:* `transfer_clusters.py` (detection), `cluster_metrics.py` (uplift).
*Validation:* star-vs-mesh detection (test 38); uplift > unconditional (test 40).

---

## 9. Portfolio risk & early warning

- **VaR_α / ES_α:** the α-quantile and tail-mean of the simulated loss
  distribution. ES ≥ VaR ≥ ≈EL. *Module:* `risk_metrics.py`.
- **HHI concentration:** Σ_i (EAD_i / Σ EAD)² — exposure concentration.
- **RAROC-vs-Sortino divergence:** rank borrowers by RAROC (correlation-blind)
  and by `sortino_copula` (correlation-aware); large rank gaps flag names whose
  risk is hidden in correlation. *Module:* `metric_comparison.py`.

---

## 10. Rating migration (Markov, existing)

Ratings evolve as a continuous-time Markov chain with generator G:
```
P(Δt) = exp(G · Δt)
```
giving transition probabilities over horizon Δt (and a path toward the absorbing
Default state). *Module:* `rating_engine.py`. (Roadmap §4 extends this to a
full delinquency Markov chain → IFRS9 lifetime-PD term structure.)

---

## 11. Merton structural PD (second signal)

A firm/household defaults when asset value V falls below debt D at horizon T:
```
PD = Φ(−d₂),   d₂ = [ln(V/D) + (r − σ²/2)T] / (σ√T)
```
With a KMV-style retail proxy V ≈ income×12 / 0.08. Used as an **independent
cross-check** on the statistical PD; large divergence is an early-warning flag.
*Module:* `structural_pd.py`.

---

## 12. Assumptions register (for validation)

| # | Assumption | Where it bites | Mitigation |
|---|---|---|---|
| A1 | PD is calibrated (score = probability) | every EL/VaR/metric | calibrate PD (Roadmap §8) |
| A2 | LGD flat 0.45 / EAD income-proxy | loss magnitudes | model LGD/EAD (Roadmap §5) |
| A3 | Gaussian factor ⇒ no tail dependence | deep-stress clustering | use t-factor copula |
| A4 | Factor structure captures all correlation | residual correlation | add factors; validate with backtest |
| A5 | Loadings ρ/β are set, not estimated | correlation level | estimate via GLS/Gauss-Markov (Roadmap §7) |
| A6 | Anchors inferred from flows, not labels | false anchors | thresholds tunable; review top cases |

---

## 13. Validation index (formula → test)

| Formula | Test |
|---|---|
| Bivariate normal/t CDF | test 34 (vs scipy) |
| Loss covariance = simulation | audit check 1; test 32 |
| Block-on-demand == dense | test 32 |
| Fréchet bounds | audit check 3 |
| Diversification ≥ 1 | test 28/36; audit check 2 |
| Multi-factor implied corr | test 39; audit check 5 |
| Anchor detection (star vs mesh) | test 38 |
| Anchor-contagion uplift | test 40 |
| Metric formulas / additivity | tests 24, 25, 29 |

Run all: `python test_copula_framework.py` → `All 41 tests passed.`
