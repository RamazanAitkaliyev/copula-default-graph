"""
Risk-Adjusted Metric Family  (src/risk_adjusted_metrics.py)
============================================================

PURPOSE
-------
A pluggable registry of Sharpe/Sortino/CoV-inspired metrics computed from
one shared set of **absolute, additive primitives**, so every metric can be
evaluated at any aggregation level (borrower, segment, geo, group, portfolio)
and the results aggregate correctly under copula correlation.

AGENT ENTRY POINT
-----------------
Preferred: use RiskAgentAPI.segment_metrics() or RiskAgentAPI.flag_divergences().
Direct use:
    calc = RiskRatioCalculator(copula, persons, exposures=exposures, lgd=0.45)
    calc.by_segment('city_name')     # metrics by any column
    calc.all_metrics()               # whole portfolio
    calc.per_borrower()              # one row per borrower (expensive: O(n²))

PRECONDITIONS
-------------
  - copula must be fitted (copula.is_fitted == True).
  - persons must have 'person_id' column with unique integers.
  - Prefer persons from ClientValueCalculator (has estimated_revenue and
    exposure_at_default). Using raw income-based EAD produces unrealistic
    revenue (2% × normalised EAD ≈ 0.003–0.097) and 66% negative-profit borrowers.

RETURNS
-------
  - by_segment()  → pd.DataFrame  (one row per segment)
  - per_borrower() → pd.DataFrame (one row per borrower)
  - all_metrics() → dict {metric_name: float}  (np.nan = undefined/div-by-zero)
  - metric()      → float
  - diversification_ratio() → float ≥ 1.0

INVARIANTS
----------
# AGENT: INV-6 — Segment variance MUST use block_sum, never per-borrower average.
#   Var(Loss_S) = sum(LossCov[i,j] for i,j in S×S)
#   This is implemented in _inputs_for() and by_segment().
#   Never call df.groupby(...)[metric].mean() for aggregation.

# AGENT: INV-7 — RiskRatioCalculator raises ValueError if copula not fitted.
#   Always check copula.is_fitted before constructing.

# AGENT: For n ≤ LOSS_COV_DENSE_MAX_NODES the loss-covariance matrix is built
#   ONCE in __init__ from the full joint_default_probability() (n×n) call.
#   For larger n NO dense matrix is built — each segment's block is computed on
#   demand via joint_default_probability_block(idx). Either way, read blocks via
#   _loss_cov_block(idx) (or the public by_segment/metric methods), never the
#   full self.loss_cov (which raises MemoryError above the threshold).

METRIC SEMANTICS
----------------
  coefficient_of_variation        σ_L0 / E[Loss]           Always ≥ 0. Safe for ranking.
  coefficient_of_variation_copula σ_L1 / E[Loss]           Copula-aware. Inflates for clusters.
  raroc                           E[Profit] / Capital       Correlation-blind.
  sharpe_indep                    (E[Profit]−rf·Rev)/σ_L0  Benchmarks vs risk-free revenue.
  sortino_indep                   (E[Profit]−h·Cap)/σ_L0   Benchmarks vs hurdle·capital.
  sortino_copula                  (E[Profit]−h·Cap)/σ_L1   PRIMARY METRIC. Copula-aware denominator.
  sortino_simulated               (E[Profit]−h·Cap)/σ_L2   Full MC tail. Requires with_sim=True.

  Sharpe and Sortino use DIFFERENT numerators by design:
    Sharpe: rf×Revenue  — opportunity cost of deploying revenue at risk-free rate
    Sortino: h×Capital  — required return on regulatory capital held
  These answer different questions and must not be conflated.

  np.nan is returned (not 0) for all undefined cases (E[Loss]=0, σ=0).
  Check with np.isfinite() — never treat nan as 0.

DIVERSIFICATION RATIO
---------------------
  DR = Σσ_i / σ_portfolio  ≥ 1  (triangle inequality)
  where Σσ_i = Σ sqrt(LossCov[i,i])  (sum of individual stds)
  and   σ_portfolio = sqrt(block_sum(LossCov))

# AGENT: The correct formula is SUM of individual sqrt(diag), not sqrt(SUM of diag).
#   These are different: Σ√x ≠ √(Σx). Using √(Σx) produces DR < 1 for correlated
#   groups — a mathematical impossibility that signals a bug.

SINGLE-BORROWER NOTE
--------------------
For n=1: L0 == L1 (no off-diagonal terms). The diversification ratio = 1.0.
The copula adds no information at n=1. Use segment/portfolio level for
contagion insights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Above this borrower count, the full dense (n,n) loss-covariance matrix is NOT
# materialised (it would be ~petabytes at 10M). Segment blocks are computed on
# demand instead. Segments themselves must still fit in memory as m×m blocks.
LOSS_COV_DENSE_MAX_NODES = 20_000


# ─────────────────────────────────────────────────────────────────────────────
# MetricInputs — aggregated primitives for one unit / segment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricInputs:
    """
    Pre-aggregated primitives for one unit (borrower, segment, or portfolio).
    All quantities are absolute and additive: a segment's values equal the
    sum of its members' values (for linear pieces) or the block-sum of the
    loss-covariance matrix (for the variance).
    """
    expected_profit: float       # Σ E[revenue_i − loss_i]
    expected_loss: float         # Σ EAD_i · LGD_i · PD_i
    expected_revenue: float      # Σ E[revenue_i]
    capital: float               # Σ capital_i  (e.g. 8 % × EAD_i)
    loss_var_L1: float           # block-sum of LossCov (pairwise copula) — copula-aware
    loss_std_indep: float        # √(Σ diag(LossCov))  — assumes independence (L0)
    hurdle_rate: float           # required return on capital
    risk_free_rate: float
    downside_semidev: Optional[float] = None  # from simulation (L2); None until computed


# ─────────────────────────────────────────────────────────────────────────────
# Metric registry
# ─────────────────────────────────────────────────────────────────────────────

MetricFn = Callable[[MetricInputs], float]
_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator that registers a metric function by name."""
    def decorator(fn: MetricFn) -> MetricFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def available_metrics() -> List[str]:
    """Return sorted list of all registered metric names."""
    return sorted(_REGISTRY)


def compute_metric(name: str, inputs: MetricInputs) -> float:
    """
    Compute a single named metric from a MetricInputs bundle.

    Returns np.nan on divide-by-zero (not a fudge constant).
    Raises KeyError for unknown metric names.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Available: {available_metrics()}")
    return _REGISTRY[name](inputs)


# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────────────────────────────────────

@register_metric("coefficient_of_variation")
def _cov(inp: MetricInputs) -> float:
    """
    Loss CoV (L0, no correlation): σ_loss_indep / E[loss].
    Always ≥ 0 and well-defined even when profit < 0 — safe for riskiness ranking.
    """
    if inp.expected_loss <= 0:
        return np.nan
    return inp.loss_std_indep / inp.expected_loss


@register_metric("coefficient_of_variation_copula")
def _cov_copula(inp: MetricInputs) -> float:
    """
    Loss CoV (L1, copula-aware): √(loss_var_L1) / E[loss].
    The off-diagonal LossCov terms inflate the numerator for contagious segments.
    """
    if inp.expected_loss <= 0 or inp.loss_var_L1 < 0:
        return np.nan
    return np.sqrt(inp.loss_var_L1) / inp.expected_loss


@register_metric("raroc")
def _raroc(inp: MetricInputs) -> float:
    """
    RAROC = E[Profit] / Capital.
    Correlation-blind (capital = k·EAD, not copula-adjusted).
    NOTE: can be negative; negative RAROC means value-destruction.
    """
    if inp.capital <= 0:
        return np.nan
    return inp.expected_profit / inp.capital


@register_metric("sharpe_indep")
def _sharpe_indep(inp: MetricInputs) -> float:
    """
    Sharpe (L0, independent): (E[Profit] − rf·Revenue) / σ_loss_indep.
    Benchmarks profit against the risk-free opportunity cost of the revenue base.
    Compare with sortino_indep (same denominator, hurdle·Capital numerator) and
    sortino_copula (copula-aware denominator, hurdle·Capital numerator).
    NOTE: sign flips when numerator < 0 — prefer CoV for pure riskiness ranking.
    """
    if inp.loss_std_indep <= 0:
        return np.nan
    excess = inp.expected_profit - inp.risk_free_rate * inp.expected_revenue
    return excess / inp.loss_std_indep


@register_metric("sortino_indep")
def _sortino_indep(inp: MetricInputs) -> float:
    """
    Sortino (L0, independent): (E[Profit] − hurdle·Capital) / σ_loss_indep.
    Differs from sharpe_indep in the numerator: Sortino benchmarks against the
    required return on capital (hurdle·Capital), while Sharpe benchmarks against
    the risk-free return on revenue (rf·Revenue). The denominator is the same
    independence-assumption std for both L0 metrics.
    NOTE: sign flips when numerator < 0.
    """
    if inp.loss_std_indep <= 0:
        return np.nan
    numerator = inp.expected_profit - inp.hurdle_rate * inp.capital
    return numerator / inp.loss_std_indep


@register_metric("sortino_copula")
def _sortino_copula(inp: MetricInputs) -> float:
    """
    Sortino (L1, copula-aware): (E[Profit] − hurdle·Capital) / √(loss_var_L1).
    The denominator inflates for concentrated/contagious segments via off-diagonal
    LossCov terms. THIS is the metric that sees correlation where RAROC does not —
    their divergence is the early-warning deliverable.
    NOTE: sign flips when numerator < 0.
    """
    if inp.loss_var_L1 < 0:
        return np.nan
    denom = np.sqrt(inp.loss_var_L1)
    if denom <= 0:
        return np.nan
    numerator = inp.expected_profit - inp.hurdle_rate * inp.capital
    return numerator / denom


@register_metric("sortino_simulated")
def _sortino_simulated(inp: MetricInputs) -> float:
    """
    Sortino (L2, simulated): uses the downside semideviation from Monte-Carlo
    draws via copula.simulate_defaults() — captures full joint tail, including
    3+-way default clustering. Requires downside_semidev to be pre-computed.
    Returns np.nan if simulation results are not available.
    """
    if inp.downside_semidev is None or inp.downside_semidev <= 0:
        return np.nan
    numerator = inp.expected_profit - inp.hurdle_rate * inp.capital
    return numerator / inp.downside_semidev


# ─────────────────────────────────────────────────────────────────────────────
# RiskRatioCalculator — the main public class
# ─────────────────────────────────────────────────────────────────────────────

class RiskRatioCalculator:
    """
    Compute all registered risk-adjusted metrics at any aggregation level.

    All hot-path computations are vectorized; no iterrows in production code.
    The loss-covariance matrix (the core object) is built once from the full
    copula.joint_default_probability() matrix call.

    Parameters
    ----------
    copula : CopulaDefaultModel
        A fitted copula model.
    persons : pd.DataFrame
        Must have 'person_id'. May optionally have 'revenue' and 'capital' columns
        to override the proxy calculations.
    exposures : np.ndarray, optional
        EAD per person (length n). Defaults to income-based proxy.
    lgd : float or np.ndarray
        Loss given default. Scalar or length-n vector.
    revenue : np.ndarray, optional
        Pre-computed revenue per person. If None, uses fee proxy or persons['revenue'].
    capital : np.ndarray, optional
        Pre-computed regulatory capital per person. If None, uses capital_ratio * EAD.
    hurdle_rate : float
        Required return on capital for Sortino numerator.
    risk_free_rate : float
        Risk-free rate for Sharpe numerator.
    capital_ratio : float
        Fraction of EAD held as capital when capital is not provided (default 0.08).
    """

    def __init__(
        self,
        copula,
        persons: pd.DataFrame,
        *,
        exposures: Optional[np.ndarray] = None,
        lgd: float = 0.45,
        revenue: Optional[np.ndarray] = None,
        capital: Optional[np.ndarray] = None,
        hurdle_rate: float = 0.10,
        risk_free_rate: float = 0.02,
        capital_ratio: float = 0.08,
    ) -> None:
        if not copula.is_fitted:
            raise ValueError("copula must be fitted before building RiskRatioCalculator")

        self._copula = copula
        self._persons = persons.reset_index(drop=True)
        self._n = copula.n
        self.hurdle_rate = hurdle_rate
        self.risk_free_rate = risk_free_rate
        self._capital_ratio = capital_ratio

        # ── resolve EAD ──────────────────────────────────────────────────────
        if exposures is not None:
            self.ead = np.asarray(exposures, dtype=float)
        elif "exposure_at_default" in persons.columns:
            self.ead = persons["exposure_at_default"].values.astype(float)
        elif "income" in persons.columns:
            # Proxy: 3 months income (same order of magnitude as a credit line)
            self.ead = persons["income"].values.astype(float) * 3.0
        else:
            self.ead = np.ones(self._n)

        # ── LGD ──────────────────────────────────────────────────────────────
        self.lgd = (np.full(self._n, lgd) if np.isscalar(lgd)
                    else np.asarray(lgd, dtype=float))

        # ── resolve revenue ───────────────────────────────────────────────────
        if revenue is not None:
            self.revenue = np.asarray(revenue, dtype=float)
        elif "revenue" in persons.columns:
            self.revenue = persons["revenue"].values.astype(float)
        elif "estimated_revenue" in persons.columns:
            self.revenue = persons["estimated_revenue"].values.astype(float)
        else:
            # Fee proxy: 2 % of EAD per year (documented fallback)
            self.revenue = self.ead * 0.02

        # ── resolve capital ───────────────────────────────────────────────────
        if capital is not None:
            self.capital = np.asarray(capital, dtype=float)
        elif "capital" in persons.columns:
            self.capital = persons["capital"].values.astype(float)
        else:
            self.capital = self.ead * capital_ratio

        # ── PD and basic EL ──────────────────────────────────────────────────
        self.pd = copula.marginal_pds.copy()
        self.el = self.ead * self.lgd * self.pd          # E[Loss_i]
        self.eprofit = self.revenue - self.el            # E[Profit_i]
        self._el_vec = self.ead * self.lgd               # loss weight per borrower

        # ── loss-covariance: dense for small n, block-on-demand for large n ──
        # Cov(D_i, D_j) = P(D_i ∩ D_j) − PD_i · PD_j
        # LossCov[i,j]  = (EAD_i·LGD_i) · Cov(D_i,D_j) · (EAD_j·LGD_j)
        #
        # AGENT/SCALE: For n above LOSS_COV_DENSE_MAX_NODES, the full (n,n) matrix
        # would be ~petabytes at 10M borrowers. Instead we keep only per-borrower
        # vectors and compute each segment's block on demand from the copula's
        # blockwise joint-default probabilities. Every consumer reads only blocks
        # (loss_cov[ix_(idx,idx)]), so nothing materialises the full matrix.
        self._dense_loss_cov: Optional[np.ndarray] = None
        if self._n <= LOSS_COV_DENSE_MAX_NODES:
            # Two supported copula interfaces:
            #  * legacy CopulaDefaultModel exposes a no-arg full-matrix
            #    `joint_default_probability()` returning the (n,n) matrix;
            #  * (multi-)factor copulas expose `joint_default_probability_block(idx)`
            #    and DO NOT have a no-arg full-matrix form (it would defeat the
            #    O(n) design). For small n we materialise the dense matrix from a
            #    single full-index block call — identical result, one interface.
            if hasattr(copula, "joint_default_probability_block") and \
                    self._copula_block_is_primary(copula):
                J = copula.joint_default_probability_block(np.arange(self._n))
            else:
                J = copula.joint_default_probability()   # (n,n) full matrix
            pd_outer = np.outer(self.pd, self.pd)
            cov_def = J - pd_outer
            np.fill_diagonal(cov_def, self.pd * (1.0 - self.pd))
            self._dense_loss_cov = (
                self._el_vec[:, None] * cov_def * self._el_vec[None, :]
            )
        else:
            logger.info(
                "RiskRatioCalculator: n=%d > %d — using block-on-demand "
                "loss-covariance (no dense n×n matrix materialised).",
                self._n, LOSS_COV_DENSE_MAX_NODES,
            )

    @staticmethod
    def _copula_block_is_primary(copula) -> bool:
        """
        True when the copula's primary joint-default interface is the BLOCK form
        (factor copulas), i.e. it has no usable no-arg full-matrix
        `joint_default_probability()`. We detect this by checking whether that
        method requires positional pair arguments (i, j) — factor copulas do.
        """
        import inspect
        fn = getattr(copula, "joint_default_probability", None)
        if fn is None:
            return True  # only the block form exists
        try:
            sig = inspect.signature(fn)
            required = [
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            return len(required) >= 1  # needs (i, j) → block-primary
        except (ValueError, TypeError):
            return False

    # ── loss-covariance block accessor (scale-safe) ───────────────────────────

    @property
    def loss_cov(self) -> np.ndarray:
        """
        Full dense loss-covariance matrix (small n only).

        Raises MemoryError above LOSS_COV_DENSE_MAX_NODES. Use _loss_cov_block()
        (or the public by_segment / metric methods) for large portfolios.
        """
        if self._dense_loss_cov is not None:
            return self._dense_loss_cov
        raise MemoryError(
            f"Full loss_cov matrix not materialised for n={self._n} "
            f"(> LOSS_COV_DENSE_MAX_NODES={LOSS_COV_DENSE_MAX_NODES}). "
            f"Use by_segment()/metric() which compute blocks on demand, "
            f"or _loss_cov_block(idx) for a specific segment."
        )

    def _loss_cov_block(self, idx: np.ndarray) -> np.ndarray:
        """
        Return the loss-covariance block for borrower indices `idx` (m×m).

        Fast path: slice the prebuilt dense matrix (small n).
        Scale path: compute the block from the copula's blockwise joint-default
        probabilities — only m×m memory, never n×n.

        The result is identical either way:
            block[a,b] = el_i · (P(D_i ∩ D_j) − PD_i·PD_j) · el_j
        with exact Bernoulli variance PD_i(1−PD_i) on the diagonal.
        """
        idx = np.asarray(idx, dtype=int)
        if self._dense_loss_cov is not None:
            return self._dense_loss_cov[np.ix_(idx, idx)]

        # Block-on-demand: ask the copula only for this sub-block of joint probs.
        pd_sub = self.pd[idx]
        el_sub = self._el_vec[idx]
        J_block = self._copula.joint_default_probability_block(idx)  # (m,m)
        cov = J_block - np.outer(pd_sub, pd_sub)
        np.fill_diagonal(cov, pd_sub * (1.0 - pd_sub))
        return el_sub[:, None] * cov * el_sub[None, :]

    # ── private helpers ───────────────────────────────────────────────────────

    def _resolve_members(self, members: Optional[np.ndarray]) -> np.ndarray:
        """Return index array; None → all borrowers."""
        if members is None:
            return np.arange(self._n)
        return np.asarray(members, dtype=int)

    def _inputs_for(
        self,
        members: Optional[np.ndarray] = None,
        *,
        downside_semidev: Optional[float] = None,
    ) -> MetricInputs:
        """
        Aggregate primitives for a subset of borrowers.
        Aggregation is ALWAYS from segment-level block-sums of the loss-covariance
        matrix — never an average of per-borrower ratios (incorrect under correlation).
        """
        idx = self._resolve_members(members)

        ep = float(self.eprofit[idx].sum())
        el = float(self.el[idx].sum())
        rev = float(self.revenue[idx].sum())
        cap = float(self.capital[idx].sum())

        block = self._loss_cov_block(idx)
        loss_var_L1 = float(block.sum())
        loss_var_indep = float(np.diag(block).sum())     # diagonal only → independence

        return MetricInputs(
            expected_profit=ep,
            expected_loss=el,
            expected_revenue=rev,
            capital=cap,
            loss_var_L1=max(loss_var_L1, 0.0),           # numerical safety (not a fudge)
            loss_std_indep=float(np.sqrt(max(loss_var_indep, 0.0))),
            hurdle_rate=self.hurdle_rate,
            risk_free_rate=self.risk_free_rate,
            downside_semidev=downside_semidev,
        )

    def _compute_downside_semidev(
        self,
        members: np.ndarray,
        n_sim: int = 10_000,
        *,
        _cached_defaults: Optional[np.ndarray] = None,
    ) -> float:
        """
        Downside semideviation of portfolio loss (Sortino L2 denominator).
        Uses a pre-simulated defaults matrix if provided (so all segments can
        share one simulation draw — consistent noise, lower cost).
        """
        if _cached_defaults is None:
            defaults = self._copula.simulate_defaults(n_sim)   # (n_sim, n)
        else:
            defaults = _cached_defaults

        # Slice to segment members
        el_vec = (self.ead * self.lgd)[members]                # (m,)
        seg_defaults = defaults[:, members]                    # (n_sim, m)
        losses = (seg_defaults * el_vec[None, :]).sum(axis=1)  # (n_sim,)

        mean_loss = losses.mean()
        downside = losses - mean_loss
        semivar = (downside[downside > 0] ** 2).mean() if (downside > 0).any() else 0.0
        return float(np.sqrt(semivar))

    # ── public API ────────────────────────────────────────────────────────────

    def metric(
        self,
        name: str,
        members: Optional[np.ndarray] = None,
        *,
        with_sim: bool = False,
        n_sim: int = 10_000,
    ) -> float:
        """Compute a single named metric for a subset (or all) borrowers."""
        dsdev = None
        if with_sim or name == "sortino_simulated":
            idx = self._resolve_members(members)
            dsdev = self._compute_downside_semidev(idx, n_sim=n_sim)
        return compute_metric(name, self._inputs_for(members, downside_semidev=dsdev))

    def all_metrics(
        self,
        members: Optional[np.ndarray] = None,
        *,
        with_sim: bool = False,
        n_sim: int = 10_000,
        _cached_defaults: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute every registered metric for a subset (or all) borrowers.
        Returns a dict {metric_name: value}.  np.nan signals undefined (div-by-0).
        Also includes 'numerator_negative' (bool) as a flag.
        """
        idx = self._resolve_members(members)
        dsdev = None
        if with_sim:
            dsdev = self._compute_downside_semidev(
                idx, n_sim=n_sim, _cached_defaults=_cached_defaults
            )
        inputs = self._inputs_for(members, downside_semidev=dsdev)

        results: Dict[str, float] = {}
        for name in available_metrics():
            try:
                results[name] = compute_metric(name, inputs)
            except Exception:
                results[name] = np.nan

        # Flag: Sortino/RAROC/Sharpe invert sign when numerator < 0 — use CoV for ranking instead
        results["numerator_negative"] = bool(
            inputs.expected_profit - inputs.hurdle_rate * inputs.capital < 0
        )
        return results

    def per_borrower(
        self,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute metrics for each borrower individually (single-member segments).

        FULLY VECTORIZED — no per-borrower loop, no block materialisation, so it
        runs in O(n) and scales to millions of borrowers.

        Why vectorisable: for a single borrower the loss-covariance "block" is
        1×1, so L0 == L1 and the loss std is exactly
            σ_i = (EAD_i·LGD_i)·√(PD_i·(1−PD_i)).
        The copula adds no diversification information at n=1 (off-diagonal terms
        require ≥2 borrowers). These per-borrower values are building blocks; use
        by_segment() for contagion-aware insight.

        NOTE: 'sortino_simulated' is omitted here (it needs a Monte-Carlo draw,
        which is meaningless for a single borrower).
        """
        # By default include every registered metric. 'sortino_simulated' is
        # meaningless per single borrower (needs a Monte-Carlo draw), so it is
        # emitted as an all-NaN column to preserve the column contract.
        metric_names = metrics or available_metrics()

        n = self._n
        # Per-borrower primitives (all length-n arrays).
        ep = self.eprofit                         # E[Profit_i]
        el = self.el                              # E[Loss_i]
        rev = self.revenue
        cap = self.capital
        sigma = self._el_vec * np.sqrt(np.clip(self.pd * (1.0 - self.pd), 0.0, None))
        # For one borrower loss_var_L1 == loss_var_indep == sigma².
        loss_var = sigma ** 2

        out: Dict[str, np.ndarray] = {
            "person_id": self._persons["person_id"].to_numpy(),
        }

        # Vectorized metric formulas (mirror the registry, but array-wise).
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = np.where(el > 0, sigma / el, np.nan)
            raroc = np.where(cap > 0, ep / cap, np.nan)
            sharpe = np.where(sigma > 0, (ep - self.risk_free_rate * rev) / sigma, np.nan)
            sortino = np.where(sigma > 0, (ep - self.hurdle_rate * cap) / sigma, np.nan)

        vectorized = {
            "coefficient_of_variation": cov,
            "coefficient_of_variation_copula": cov,   # L0==L1 at n=1
            "raroc": raroc,
            "sharpe_indep": sharpe,
            "sortino_indep": sortino,
            "sortino_copula": sortino,                # L0==L1 at n=1
        }
        for name in metric_names:
            if name == "sortino_simulated":
                # Undefined for a single borrower; emit NaN column for contract.
                out[name] = np.full(n, np.nan)
            elif name in vectorized:
                out[name] = vectorized[name]
            elif name in _REGISTRY:
                # Unknown custom metric: fall back to per-row compute (rare path).
                vals = np.empty(n, dtype=float)
                for i in range(n):
                    inp = MetricInputs(
                        expected_profit=float(ep[i]), expected_loss=float(el[i]),
                        expected_revenue=float(rev[i]), capital=float(cap[i]),
                        loss_var_L1=float(loss_var[i]), loss_std_indep=float(sigma[i]),
                        hurdle_rate=self.hurdle_rate, risk_free_rate=self.risk_free_rate,
                    )
                    try:
                        vals[i] = compute_metric(name, inp)
                    except Exception:
                        vals[i] = np.nan
                out[name] = vals

        out["numerator_negative"] = (ep - self.hurdle_rate * cap) < 0
        return pd.DataFrame(out)

    def by_segment(
        self,
        segment_col: str,
        *,
        metrics: Optional[List[str]] = None,
        with_sim: bool = False,
        n_sim: int = 10_000,
        drop_unlabelled: bool = True,
    ) -> pd.DataFrame:
        """
        Compute metrics aggregated by any column in persons.

        Parameters
        ----------
        segment_col : str
            Column in persons to group by (e.g. 'city_name', 'risk_archetype',
            'high_risk_group_id').
        metrics : list, optional
            Subset of metric names. Defaults to all registered metrics.
        with_sim : bool
            If True, also compute sortino_simulated; uses one shared simulation
            draw for all segments (consistent noise).
        drop_unlabelled : bool
            For integer group columns (e.g. 'high_risk_group_id'), drop rows
            where value == -1 (meaning "not in a group").

        Returns
        -------
        pd.DataFrame with one row per segment, columns:
            segment, n, exposure, exposure_share, expected_profit, expected_loss,
            loss_std_indep, loss_std_copula, diversification_ratio,
            <metric columns>, numerator_negative
        """
        if segment_col not in self._persons.columns:
            raise ValueError(f"'{segment_col}' not found in persons columns")

        metric_names = metrics or available_metrics()

        # One shared simulation draw so all segments have consistent random noise
        cached_defaults = None
        if with_sim:
            cached_defaults = self._copula.simulate_defaults(n_sim)

        labels = self._persons[segment_col].values
        unique_labels = np.unique(labels)

        rows = []
        total_exposure = float(self.ead.sum())

        for lbl in unique_labels:
            # Drop "no group" sentinel for integer group columns
            if drop_unlabelled and isinstance(lbl, (int, np.integer)) and lbl == -1:
                continue

            idx = np.where(labels == lbl)[0]
            n_seg = len(idx)
            seg_exposure = float(self.ead[idx].sum())

            # Extract the block once; reuse for all derived quantities.
            block = self._loss_cov_block(idx)
            block_sum = float(block.sum())
            block_diag = np.diag(block)

            loss_var_L1_seg = max(block_sum, 0.0)
            loss_var_indep_seg = float(block_diag.sum())
            loss_std_copula = float(np.sqrt(loss_var_L1_seg))

            # Build MetricInputs from the pre-computed block (avoids double extraction)
            ep = float(self.eprofit[idx].sum())
            el = float(self.el[idx].sum())
            rev = float(self.revenue[idx].sum())
            cap = float(self.capital[idx].sum())
            inp = MetricInputs(
                expected_profit=ep,
                expected_loss=el,
                expected_revenue=rev,
                capital=cap,
                loss_var_L1=loss_var_L1_seg,
                loss_std_indep=float(np.sqrt(max(loss_var_indep_seg, 0.0))),
                hurdle_rate=self.hurdle_rate,
                risk_free_rate=self.risk_free_rate,
            )

            # Diversification ratio: Σσ_i / σ_portfolio  (≥1 by triangle inequality)
            # Σσ_i = sum of individual stds; σ_portfolio = sqrt of full block-variance.
            # NOTE: must sum sqrt(diag[i]), NOT sqrt(sum(diag)) — those are different.
            sum_individual_stds = float(np.sqrt(block_diag).sum())
            div_ratio = (sum_individual_stds / loss_std_copula
                         if loss_std_copula > 1e-15 else 1.0)

            row: Dict = {
                "segment": lbl,
                "n": n_seg,
                "exposure": round(seg_exposure, 2),
                "exposure_share": round(seg_exposure / total_exposure, 4) if total_exposure > 0 else 0.0,
                "expected_profit": round(inp.expected_profit, 4),
                "expected_loss": round(inp.expected_loss, 4),
                "loss_std_indep": round(inp.loss_std_indep, 6),
                "loss_std_copula": round(loss_std_copula, 6),
                "diversification_ratio": round(div_ratio, 4),
            }

            # Simulated downside semidev for this segment (L2, optional)
            dsdev = None
            if with_sim and cached_defaults is not None:
                dsdev = self._compute_downside_semidev(
                    idx, n_sim=n_sim, _cached_defaults=cached_defaults
                )
            # Reuse inp; only rebuild if dsdev was computed
            inp_full = inp if dsdev is None else MetricInputs(
                expected_profit=inp.expected_profit,
                expected_loss=inp.expected_loss,
                expected_revenue=inp.expected_revenue,
                capital=inp.capital,
                loss_var_L1=inp.loss_var_L1,
                loss_std_indep=inp.loss_std_indep,
                hurdle_rate=inp.hurdle_rate,
                risk_free_rate=inp.risk_free_rate,
                downside_semidev=dsdev,
            )

            for name in metric_names:
                if name in _REGISTRY:
                    try:
                        row[name] = compute_metric(name, inp_full)
                    except Exception:
                        row[name] = np.nan

            row["numerator_negative"] = bool(
                inp.expected_profit - inp.hurdle_rate * inp.capital < 0
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def diversification_ratio(self, members: Optional[np.ndarray] = None) -> float:
        """
        Portfolio diversification ratio: Σσ_i / σ_portfolio (≥1 by triangle inequality).
        Equals 1 when all borrowers are perfectly correlated (no diversification benefit).
        Greater values indicate more diversification.

        Σσ_i is the SUM of individual standard deviations (not sqrt of the sum of variances).
        σ_portfolio is sqrt of the full block-variance (includes off-diagonal covariances).
        """
        idx = self._resolve_members(members)
        block = self._loss_cov_block(idx)
        sigma_portfolio = float(np.sqrt(max(block.sum(), 0.0)))
        # Sum of individual stds: sqrt each diagonal entry, then sum
        sum_individual_stds = float(np.sqrt(np.diag(block)).sum())
        if sigma_portfolio < 1e-15:
            return 1.0
        return sum_individual_stds / sigma_portfolio
