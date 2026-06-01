# Copula-Based Default Contagion on Transaction Networks

**AI agents / Claude Code:** read [`CLAUDE.md`](CLAUDE.md) first — it has the file map, data schema, invariants, and debug commands.

A framework for modeling probability of default (PD) with graph-based contagion effects using copulas.

## The Problem

Traditional PD models treat individuals independently:
```
P(Default_i) = f(income_i, age_i, credit_score_i, ...)
```

**Reality**: Defaults are correlated through:
- Financial connections (who sends money to whom)
- Geographic clustering (local economic shocks)
- Social networks (similar behaviors)

**This Project**: Model joint defaults using copulas on transaction graphs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COPULA DEFAULT GRAPH                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Individual  │     │    Graph     │     │   Copula     │    │
│  │  PD Model    │────▶│   Features   │────▶│  Contagion   │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ P(Default_i) │     │  Centrality  │     │ Joint P(D_i, │    │
│  │ = f(X_i)     │     │  Clustering  │     │    D_j | ρ)  │    │
│  └──────────────┘     │  PageRank    │     └──────────────┘    │
│                       └──────────────┘                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  RISK METRICS                            │   │
│  │  • Portfolio Expected Loss                               │   │
│  │  • Default Contagion Probability                         │   │
│  │  • Systemic Risk Score                                   │   │
│  │  • Fraud Ring Detection                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Individual PD Model

Logistic regression or gradient boosting on individual features:

| Feature | Description |
|---------|-------------|
| income | Monthly income |
| age | Age in years |
| employment_years | Years at current job |
| debt_to_income | Total debt / income ratio |
| num_credit_lines | Number of credit accounts |
| missed_payments | Historical missed payments |
| location_risk | Regional economic risk score |

### 2. Graph-Based Features

Transaction network provides additional signals:

| Feature | Interpretation |
|---------|---------------|
| in_degree | How many people send money to this person |
| out_degree | How many people this person sends to |
| pagerank | Importance in the network |
| clustering_coeff | How connected are my connections |
| avg_neighbor_pd | Average PD of connected people |
| high_risk_neighbors | Count of high-PD connections |
| transaction_volatility | Variance in transaction amounts |

### 3. Copula for Joint Defaults

**Why Copulas?**
- Marginal PDs may be 5% each
- But P(both default | connected) could be 15% due to tail dependence

**Copula Types for Default:**

| Copula | Tail Dependence | Use Case |
|--------|-----------------|----------|
| Gaussian | None | Normal times |
| Student-t | Symmetric | Stress scenarios |
| Clayton | Lower tail | Default clustering |
| Gumbel | Upper tail | Survival clustering |

**Clayton Copula** (best for defaults):
```
C(u, v; θ) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
```
- θ > 0: positive dependence
- Higher θ = stronger default clustering

### 4. Contagion Model

```
P(Default_i | Neighbor_j defaults) > P(Default_i)
```

Modeled via:
1. Direct effect: Transaction exposure
2. Indirect effect: Reputation/信用 contagion
3. Systemic effect: Location/sector shocks

## Project Structure

```
copula_default_graph/
├── CLAUDE.md                    # AI-agent guide (file map, invariants, debug)
├── README.md
├── requirements.txt
├── pyproject.toml               # pip install -e . support
├── debug.py                     # Quick diagnostic helpers
├── main.py                      # 13-step end-to-end pipeline
├── test_copula_framework.py     # 23 unit tests
├── copula_default_analysis.ipynb # Interactive walkthrough
├── output/                      # Generated charts + CSVs (run main.py)
└── src/
    ├── __init__.py              # Re-exports all public API
    ├── config.py                # NetworkConfig, CopulaConfig, …
    ├── data_generator.py        # generate_network() synthetic data
    ├── graph_features.py        # TransactionGraph, correlation matrix
    ├── copula_model.py          # CopulaDefaultModel (5 types)
    ├── risk_metrics.py          # RiskAnalyzer, FraudRingDetector
    ├── pd_model.py              # IndividualPDModel (logistic / GB)
    ├── client_value_metrics.py  # Sharpe, RAROC, client segments
    ├── rating_engine.py         # PD → AAA…Default + migration matrix
    ├── structural_pd.py         # Merton structural PD (KMV proxy)
    ├── flexible_probs.py        # Regime-aware copula calibration
    └── customer_profile.py      # Per-borrower risk report + watchlist
```

## Quick commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full 13-step pipeline (writes charts + CSVs to output/)
python main.py

# Run all 23 tests
python test_copula_framework.py

# Debug helpers
python debug.py smoke              # fast sanity check (n=100, no plots)
python debug.py copula             # print fitted copula state
python debug.py stress             # run stress test, print delta table
python debug.py ratings            # show rating distribution
python debug.py profile 42         # full risk report for borrower 42
python debug.py test data_generation  # run one named test
```

## Quick Start (API)

```python
from src.data_generator import generate_network, get_summary_stats
from src.graph_features import TransactionGraph
from src.copula_model import CopulaDefaultModel
from src.risk_metrics import RiskAnalyzer

# 1) Generate synthetic network data
persons, transactions = generate_network(seed=42)
stats = get_summary_stats(persons, transactions)

# 2) Build graph and derive correlation matrix
graph = TransactionGraph(transactions, persons)
corr_matrix = graph.get_correlation_matrix()

# 3) Fit copula dependency model on marginal PD + correlation matrix
copula = CopulaDefaultModel(copula_type='clayton')
copula.fit(persons['base_pd'].values, corr_matrix)

# 4) Compute risk metrics at individual/group/portfolio levels
analyzer = RiskAnalyzer(copula, graph, persons, lgd=0.45)
individual_risks = analyzer.compute_individual_risks()
portfolio_risks = analyzer.compute_portfolio_risks()

print(stats['n_persons'], stats['n_transactions'])
print(portfolio_risks.expected_loss, portfolio_risks.var_95, portfolio_risks.es_95)
```

## Mathematical Framework

### Joint Default Probability

For connected persons i and j:

```
P(D_i ∩ D_j) = C(P(D_i), P(D_j); θ_ij)
```

Where:
- C is the copula function
- θ_ij depends on connection strength

### Portfolio Expected Loss

```
E[Loss] = Σ_i EAD_i × LGD_i × PD_i + Contagion_Adjustment
```

### Contagion Adjustment

```
Contagion_i = Σ_j w_ij × [P(D_i|D_j) - P(D_i)] × P(D_j)
```

## Use Cases

1. **Credit Portfolio Management**
   - Identify concentration risk from connected borrowers
   - Stress test contagion scenarios

2. **Fraud Ring Detection**
   - Unusual transaction patterns + high joint default probability
   - Circular money flows with rising PDs

3. **Systemic Risk Assessment**
   - Which defaults would cascade through the network?
   - Critical nodes (too connected to fail)

4. **Location-Based Risk**
   - Regional economic shocks affecting clusters
   - Geographic diversification metrics

## References

- Li, D. X. (2000). "On Default Correlation: A Copula Function Approach"
- Schönbucher, P. J. (2003). "Credit Derivatives Pricing Models"
- Newman, M. (2010). "Networks: An Introduction"
