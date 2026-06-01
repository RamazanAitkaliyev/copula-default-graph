# Copula-Based Default Contagion on Transaction Networks

## Complete Tutorial: Concepts, Theory, and Applications

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [Key Concepts](#2-key-concepts)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [System Architecture](#4-system-architecture)
5. [Step-by-Step Usage Guide](#5-step-by-step-usage-guide)
6. [Real-World Applications](#6-real-world-applications)
7. [Interpreting Results](#7-interpreting-results)

---

## 1. The Problem We're Solving

### Traditional Credit Risk (The Limitation)

Traditional probability of default (PD) models treat each borrower independently:

```
P(Default_Alice) = f(income_Alice, debt_Alice, credit_score_Alice, ...)
P(Default_Bob)   = f(income_Bob, debt_Bob, credit_score_Bob, ...)
```

**Problem**: This assumes Alice defaulting has NO effect on Bob's default probability.

### Reality: Defaults Are Correlated

In the real world, defaults cluster together due to:

1. **Financial Connections**: If Alice owes Bob money, Alice's default hurts Bob
2. **Geographic Clustering**: Economic downturns affect entire regions
3. **Industry Exposure**: Tech layoffs affect many tech workers simultaneously
4. **Social Networks**: People in the same community have similar behaviors

### The 2008 Financial Crisis Example

During 2008, mortgage defaults didn't happen independently:
- One homeowner defaults → neighborhood prices drop
- Neighbors now underwater → more defaults
- Bank losses → credit tightens → more defaults
- **Cascade effect**: Individual risks multiplied into systemic crisis

### What This System Does

This framework captures **correlated defaults** by:
1. Building a **transaction network** (who pays whom)
2. Using **copulas** to model joint default probabilities
3. Computing **contagion risk** (how defaults spread)
4. Measuring **portfolio tail risk** (worst-case scenarios)

---

## 2. Key Concepts

### 2.1 Marginal vs Joint Probability

**Marginal PD**: Individual default probability ignoring others
```
P(Alice defaults) = 5%
P(Bob defaults) = 3%
```

**Joint PD**: Probability both default together
```
If independent: P(both default) = 5% × 3% = 0.15%
If correlated:  P(both default) = 0.8% (much higher!)
```

The difference is **tail dependence** - defaults cluster in stress.

### 2.2 Copulas (The Core Innovation)

A **copula** is a function that links marginal distributions to create a joint distribution.

**Why copulas?**
- Separate the "individual risk" from the "dependence structure"
- Model tail dependence (extreme events happening together)
- Flexible: can use any marginal distributions

**Sklar's Theorem**: Any joint distribution can be written as:
```
F(x, y) = C(F_X(x), F_Y(y))
```
Where C is the copula and F_X, F_Y are marginal CDFs.

### 2.3 Types of Copulas (This System Supports 5)

| Copula | Tail Dependence | Best For |
|--------|----------------|----------|
| **Gaussian** | None | Normal times, no crisis clustering |
| **Student-t** | Symmetric (both tails) | Stress in either direction |
| **Clayton** | Lower tail only | **Defaults cluster in crisis** (most common for credit) |
| **Gumbel** | Upper tail only | Survival clustering |
| **Frank** | None | Moderate symmetric dependence |

**For credit risk, Clayton is typically best** because defaults cluster during downturns (lower tail).

### 2.4 Contagion Metrics

**Contagion Vulnerability**: How much does MY default probability increase if my NEIGHBORS default?
```
Vulnerability_i = Σ_j [P(D_i | D_j) - P(D_i)] × P(D_j)
```
High vulnerability = You're exposed to risky neighbors.

**Systemic Importance**: How much do OTHERS' default probabilities increase if I default?
```
Importance_i = Σ_j [P(D_j | D_i) - P(D_j)]
```
High importance = You're "too connected to fail."

### 2.5 Portfolio Risk Metrics

**Expected Loss (EL)**: Average loss across all scenarios
```
EL = Σ_i EAD_i × LGD_i × PD_i
```

**Value at Risk (VaR)**: Loss that won't be exceeded with X% confidence
```
VaR_95 = "We're 95% confident losses won't exceed this"
```

**Expected Shortfall (ES)**: Average loss in the worst X% of scenarios
```
ES_95 = "If we're in the worst 5%, this is the average loss"
```

ES is more informative than VaR because it tells you HOW BAD the tail is.

---

## 3. Mathematical Foundation

### 3.1 Clayton Copula (Primary Model)

The Clayton copula captures **lower tail dependence**:

```
C(u, v; θ) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
```

Where:
- `u, v` are marginal PDs (uniform on [0,1])
- `θ > 0` is the dependence parameter (higher = more dependence)

**Lower tail dependence coefficient**:
```
λ_L = 2^(-1/θ)
```

Example: θ = 2 → λ_L = 0.71 (71% chance of joint default in extreme stress)

### 3.2 From Correlation to Copula Parameter

We estimate θ from the average correlation in the network:

1. Compute Kendall's tau from linear correlation:
   ```
   τ = (2/π) × arcsin(ρ)
   ```

2. Convert to Clayton θ:
   ```
   θ = 2τ / (1 - τ)
   ```

### 3.3 Joint Default Probability

For persons i and j with marginal PDs `p_i` and `p_j`:

```
P(D_i ∩ D_j) = C(p_i, p_j; θ_ij)
```

Where `θ_ij` is adjusted for their specific correlation:
```
θ_ij = θ_global × (1 + ρ_ij) / 2
```

### 3.4 Conditional Default (Contagion)

```
P(D_i | D_j) = P(D_i ∩ D_j) / P(D_j)
```

This is always ≥ P(D_i) when there's positive dependence.

### 3.5 Monte Carlo Simulation

To compute portfolio risk:

1. **Generate correlated uniforms** from the copula
2. **Convert to defaults**: If U_i < PD_i, person i defaults
3. **Compute portfolio loss**: Loss = Σ (default_i × exposure_i × LGD)
4. **Repeat 10,000 times** to build loss distribution
5. **Extract VaR and ES** from the distribution

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                                │
│  • Person features (income, debt, credit score, city, etc.)     │
│  • Transaction records (who paid whom, how much)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA GENERATOR                                │
│  src/data_generator.py                                          │
│  • Creates synthetic network (1000 persons, 3 cities)           │
│  • Embeds high-risk groups and bridge individuals               │
│  • Computes marginal PDs from features                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPH FEATURES                                │
│  src/graph_features.py                                          │
│  • Builds adjacency matrices (who is connected)                 │
│  • Computes centrality (PageRank, betweenness)                  │
│  • Derives correlation matrix from network structure            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    COPULA MODEL                                  │
│  src/copula_model.py                                            │
│  • Fits copula to marginal PDs + correlation matrix             │
│  • Computes joint/conditional default probabilities             │
│  • Monte Carlo simulation of correlated defaults                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RISK METRICS                                  │
│  src/risk_metrics.py                                            │
│  • Individual risk: vulnerability, systemic importance          │
│  • Group risk: cluster analysis                                 │
│  • Portfolio risk: VaR, ES, stress testing                      │
│  • Fraud detection: suspicious patterns                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUTS                                   │
│  • Risk rankings (who is most dangerous?)                       │
│  • Loss distributions (what could we lose?)                     │
│  • Network visualizations (where is risk concentrated?)         │
│  • Stress test results (how bad could it get?)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Step-by-Step Usage Guide

### Step 1: Basic Setup

```python
import numpy as np
from src import (
    generate_network,
    TransactionGraph,
    CopulaDefaultModel,
    RiskAnalyzer,
)

# Set random seed for reproducibility
np.random.seed(42)
```

### Step 2: Generate or Load Data

**Option A: Use synthetic data (for testing)**
```python
# Generate synthetic network
persons, transactions = generate_network(seed=42)

print(f"Persons: {len(persons)}")
print(f"Transactions: {len(transactions)}")
print(f"Cities: {persons['city_name'].unique()}")
```

**Option B: Use your own data**
```python
import pandas as pd

# Your persons DataFrame needs these columns:
# - person_id: unique identifier
# - city_id: which city/region
# - base_pd: probability of default (0 to 1)
# - high_risk_group_id: group membership (-1 if none)
# - is_bridge: whether they connect regions

persons = pd.read_csv('your_persons.csv')

# Your transactions DataFrame needs:
# - sender_id: who sent money
# - receiver_id: who received money
# - amount: transaction amount

transactions = pd.read_csv('your_transactions.csv')
```

### Step 3: Build the Transaction Graph

```python
# Build graph from transactions
graph = TransactionGraph(transactions, persons)

# Check network statistics
stats = graph.get_network_stats()
print(f"Nodes: {stats.n_nodes}")
print(f"Edges: {stats.n_edges}")
print(f"Density: {stats.density:.4f}")
print(f"Average degree: {stats.avg_degree:.1f}")
```

### Step 4: Derive Correlation Matrix

The correlation matrix captures how connected people are:

```python
# Derive correlation from network structure
corr_matrix = graph.get_correlation_matrix(
    base_corr=0.05,        # Minimum correlation for unconnected
    max_corr=0.60,         # Maximum correlation for highly connected
    same_city_boost=0.10,  # Extra correlation for same city
    same_group_boost=0.20, # Extra correlation for same risk group
)

# Check average correlation
n = corr_matrix.shape[0]
off_diag = corr_matrix[~np.eye(n, dtype=bool)]
print(f"Average pairwise correlation: {off_diag.mean():.4f}")
```

### Step 5: Fit the Copula Model

```python
# Choose copula type based on your needs:
# - 'clayton': Lower tail dependence (defaults cluster in crisis) - RECOMMENDED
# - 'gumbel': Upper tail dependence (survival clustering)
# - 'student_t': Symmetric tail dependence
# - 'gaussian': No tail dependence
# - 'frank': No tail dependence, symmetric

copula = CopulaDefaultModel('clayton')
copula.fit(persons['base_pd'].values, corr_matrix)

# Check fitted parameters
print(f"Copula type: {copula.copula_type}")
print(f"Theta parameter: {copula.params.theta:.4f}")
print(f"Tail dependence: {copula.tail_dependence():.4f}")
```

### Step 6: Compute Risk Metrics

```python
# Define exposures (how much you'd lose if each person defaults)
# Could be loan amounts, credit limits, etc.
exposures = persons['income'].values / persons['income'].mean()

# Create risk analyzer
analyzer = RiskAnalyzer(
    copula_model=copula,
    graph=graph,
    persons=persons,
    exposures=exposures,
    lgd=0.45,  # Loss given default (45% is typical)
)
```

#### 6a: Individual Risk Analysis

```python
# Compute individual risk metrics
individual_risks = analyzer.compute_individual_risks()

# See what metrics we get
print(individual_risks.columns.tolist())
# ['person_id', 'city_name', 'risk_archetype', 'marginal_pd',
#  'contagion_vulnerability', 'systemic_importance', 'network_exposure',
#  'composite_risk_score', 'risk_tier', 'in_high_risk_group', 'is_bridge']

# Top 10 riskiest individuals
print("\nTop 10 Riskiest Individuals:")
print(individual_risks.head(10)[
    ['person_id', 'city_name', 'marginal_pd', 'composite_risk_score', 'risk_tier']
])
```

#### 6b: Portfolio Risk Analysis

```python
# Compute portfolio-level risk
portfolio = analyzer.compute_portfolio_risks(n_simulations=10000)

print(f"\nPortfolio Risk Metrics:")
print(f"  Expected Loss: {portfolio.expected_loss:.2f}")
print(f"  VaR 95%: {portfolio.var_95:.2f}")
print(f"  VaR 99%: {portfolio.var_99:.2f}")
print(f"  Expected Shortfall 95%: {portfolio.es_95:.2f}")
print(f"  Expected Shortfall 99%: {portfolio.es_99:.2f}")
print(f"  Default Correlation: {portfolio.default_correlation:.4f}")
print(f"  Tail Risk Ratio (ES/VaR): {portfolio.tail_risk_ratio:.2f}")
```

#### 6c: Stress Testing

```python
# Run stress test: what if PDs double and correlations increase?
stress_results = analyzer.stress_test(
    pd_multiplier=2.0,       # Double all PDs
    correlation_boost=0.20,  # Add 0.2 to all correlations
)

print("\nStress Test Results:")
print(f"  Base Expected Loss: {stress_results['base']['expected_loss']:.2f}")
print(f"  Stressed Expected Loss: {stress_results['stressed']['expected_loss']:.2f}")
print(f"  Change: {stress_results['change']['expected_loss']:+.1%}")
```

### Step 7: Visualize Results

```python
import matplotlib.pyplot as plt

# Plot network colored by PD
fig = graph.plot_network(
    color_by='base_pd',      # Color nodes by default probability
    size_by='degree',        # Size nodes by number of connections
    layout='city',           # Separate by city
    title='Transaction Network (color = PD)',
    figsize=(14, 10),
)
plt.savefig('network_visualization.png', dpi=150)
plt.show()

# Plot loss distribution
losses = analyzer.get_loss_distribution(n_simulations=10000)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(losses, bins=50, density=True, alpha=0.7, color='steelblue')
ax.axvline(portfolio.var_95, color='orange', linestyle='--', label=f'VaR 95% = {portfolio.var_95:.1f}')
ax.axvline(portfolio.es_95, color='red', linestyle='--', label=f'ES 95% = {portfolio.es_95:.1f}')
ax.set_xlabel('Portfolio Loss')
ax.set_ylabel('Density')
ax.set_title('Loss Distribution with VaR and ES')
ax.legend()
plt.savefig('loss_distribution.png', dpi=150)
plt.show()
```

### Step 8: Compare Copulas

```python
from src import compare_copulas

# Compare all copula types
comparison = compare_copulas(
    persons['base_pd'].values,
    corr_matrix,
    n_simulations=5000
)

print("\nCopula Comparison:")
for copula_type, metrics in comparison.items():
    print(f"\n{copula_type.upper()}:")
    print(f"  Theta: {metrics['theta']:.4f}")
    print(f"  Lower Tail Dependence: {metrics['tail_dependence']:.4f}")
    print(f"  Upper Tail Dependence: {metrics['tail_dependence_upper']:.4f}")
    print(f"  Simulated Default Rate: {metrics['sim_default_rate']:.4f}")
```

---

## 6. Real-World Applications

### Application 1: Bank Credit Portfolio

**Scenario**: A bank has 10,000 small business loans

**Use case**:
1. Build transaction graph from payment flows between businesses
2. Fit Clayton copula to capture crisis clustering
3. Compute portfolio VaR and ES for capital requirements
4. Identify systemically important borrowers for closer monitoring

```python
# Example: Identify businesses that could trigger cascades
top_systemic = individual_risks.nlargest(20, 'systemic_importance')
print("Businesses requiring enhanced monitoring:")
print(top_systemic[['person_id', 'marginal_pd', 'systemic_importance']])
```

### Application 2: Supply Chain Risk

**Scenario**: A manufacturer wants to assess supplier default risk

**Use case**:
1. Map supplier payment relationships as transaction graph
2. Identify critical suppliers (high systemic importance)
3. Compute probability of multiple supplier defaults
4. Stress test: what if a key region has economic shock?

```python
# Example: Stress test a specific region
region_ids = persons[persons['city_name'] == 'Gamma']['person_id'].values
stress_tester = ContagionStressTester(copula, graph)
cascade = stress_tester.stress_scenario(
    shock_nodes=list(region_ids[:10]),  # Shock 10 suppliers
    pd_multiplier=3.0
)
print(f"Initial shocks: {len(cascade['initial_defaults'])}")
print(f"Total defaults after cascade: {cascade['total_defaults']}")
print(f"Cascade multiplier: {cascade['cascade_multiplier']:.1f}x")
```

### Application 3: Fraud Ring Detection

**Scenario**: Detect suspicious transaction patterns

**Use case**:
1. Look for dense clusters with circular money flows
2. Flag groups with unusually high joint default probabilities
3. Identify individuals with suspicious transaction patterns

```python
from src.risk_metrics import FraudRingDetector

detector = FraudRingDetector(graph, copula, persons)
suspicious = detector.detect_suspicious_clusters(
    min_cluster_size=3,
    joint_pd_threshold=0.15,
    density_threshold=0.5
)

print(f"Found {len(suspicious)} suspicious clusters:")
for cluster in suspicious[:5]:
    print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} members, "
          f"suspicion score = {cluster['suspicion_score']:.2f}")
```

### Application 4: Insurance Portfolio

**Scenario**: An insurer has correlated risks (e.g., flood insurance in a region)

**Use case**:
1. Model policyholder locations as network (geographic proximity)
2. Use copula to capture correlated claims
3. Compute tail risk for reinsurance decisions
4. Price policies accounting for correlation

---

## 7. Interpreting Results

### 7.1 Understanding Risk Tiers

| Tier | Percentile | Action |
|------|------------|--------|
| **Low** | 0-60% | Standard monitoring |
| **Medium** | 60-85% | Enhanced monitoring |
| **High** | 85-95% | Active risk management |
| **Critical** | 95-100% | Immediate intervention |

### 7.2 Key Metrics to Watch

**For Individuals**:
- **High marginal PD + High systemic importance** = "Too connected to fail"
  - Action: Reduce exposure or require more collateral

- **Low marginal PD + High vulnerability** = "Hidden risk"
  - Action: Monitor neighbors' health

**For Portfolio**:
- **ES/VaR ratio > 1.5** = Heavy tail risk
  - Action: Consider tail hedging or reinsurance

- **High default correlation** = Diversification not working
  - Action: Reduce concentration in correlated segments

### 7.3 Stress Test Interpretation

| EL Change | Severity | Interpretation |
|-----------|----------|----------------|
| < 50% | Low | Portfolio resilient to stress |
| 50-100% | Moderate | Some vulnerability, manageable |
| 100-200% | High | Significant stress exposure |
| > 200% | Severe | Consider risk reduction |

### 7.4 Copula Selection Guide

| Situation | Recommended Copula |
|-----------|-------------------|
| Credit risk (defaults cluster in crisis) | **Clayton** |
| Two-sided risk (could go either way) | **Student-t** |
| No extreme dependence expected | **Gaussian** or **Frank** |
| Survival analysis (survivors cluster) | **Gumbel** |

---

## Quick Reference: Complete Pipeline

```python
import numpy as np
from src import (
    generate_network, TransactionGraph,
    CopulaDefaultModel, RiskAnalyzer
)

# 1. Data
np.random.seed(42)
persons, transactions = generate_network()

# 2. Graph
graph = TransactionGraph(transactions, persons)
corr_matrix = graph.get_correlation_matrix()

# 3. Copula
copula = CopulaDefaultModel('clayton')
copula.fit(persons['base_pd'].values, corr_matrix)

# 4. Risk Analysis
analyzer = RiskAnalyzer(copula, graph, persons, lgd=0.45)
individual_risks = analyzer.compute_individual_risks()
portfolio_risks = analyzer.compute_portfolio_risks(n_simulations=10000)
stress_results = analyzer.stress_test(pd_multiplier=2.0)

# 5. Results
print(f"Expected Loss: {portfolio_risks.expected_loss:.2f}")
print(f"VaR 95%: {portfolio_risks.var_95:.2f}")
print(f"ES 95%: {portfolio_risks.es_95:.2f}")
print(f"Stress EL Change: {stress_results['change']['expected_loss']:+.1%}")
```

---

## Further Reading

1. **Li, D. X. (2000)**: "On Default Correlation: A Copula Function Approach" - The foundational paper on copulas in credit risk

2. **Schönbucher, P. J. (2003)**: "Credit Derivatives Pricing Models" - Comprehensive treatment of credit modeling

3. **McNeil, A. J., Frey, R., & Embrechts, P. (2015)**: "Quantitative Risk Management" - The bible of risk management

4. **Newman, M. (2010)**: "Networks: An Introduction" - Network analysis foundations
