"""
Empirical Copula Calibration  (src/copula_calibration.py)
=========================================================

Estimate copula dependence parameters from OBSERVED default / delinquency /
rating data, so the risk team has a defensible answer to "where did the copula
parameters come from?" — instead of relying only on configured correlation /
loadings.

This is the implementation of `implementation_plans/07_copula_parameter_calibration_plan.md`
(first version: empirical dependence measures + per-family fitters + a family
comparison, plus a diagnostic-only agent entry point). It is self-contained
(numpy / pandas / scipy + the project-local `dependence.schweizer_wolff`).

Pipeline
--------
1. ``build_default_panel`` — validate / assemble a borrower-period default panel.
2. ``empirical_dependence_measures`` — Kendall τ, Spearman ρ, Schweizer-Wolff,
   default correlation, observed vs independent joint-default rate, lower-tail
   co-default frequency — overall and per segment.
3. ``calibrate_copula`` — map the empirical dependence to copula parameters for
   Gaussian / Student-t / Clayton, pick a recommended family, and report
   goodness-of-fit (model vs observed joint-default, Fréchet-bound check).

Key formulas
------------
- Clayton:    θ = 2τ / (1 − τ)           (Kendall τ → Clayton θ)
- Gumbel:     θ = 1 / (1 − τ)
- Gaussian:   ρ_latent = sin(π τ / 2)    (Kendall τ → latent correlation)
- Default correlation:
      ρ_D = (p_AB − p_A p_B) / sqrt(p_A(1−p_A) p_B(1−p_B))
  where p_AB is the observed joint-default rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .dependence import schweizer_wolff


# ──────────────────────────────────────────────────────────────────────────────
# Parameter conversions (Kendall τ → copula parameters)
# ──────────────────────────────────────────────────────────────────────────────

def clayton_theta_from_tau(tau: float) -> float:
    """Clayton θ from Kendall's τ: θ = 2τ/(1−τ); clipped to a positive range."""
    tau = float(np.clip(tau, 1e-4, 0.95))
    return max(1e-3, 2.0 * tau / (1.0 - tau))


def gumbel_theta_from_tau(tau: float) -> float:
    """Gumbel θ from Kendall's τ: θ = 1/(1−τ) ≥ 1."""
    tau = float(np.clip(tau, 1e-4, 0.95))
    return max(1.0, 1.0 / (1.0 - tau))


def gaussian_rho_from_tau(tau: float) -> float:
    """Latent Gaussian correlation from Kendall's τ: ρ = sin(πτ/2)."""
    tau = float(np.clip(tau, -0.95, 0.95))
    return float(np.sin(np.pi * tau / 2.0))


def default_correlation(p_a: float, p_b: float, p_ab: float) -> float:
    """
    Pairwise default correlation from marginal and joint default rates.

    ρ_D = (p_AB − p_A·p_B) / sqrt(p_A(1−p_A)·p_B(1−p_B)).
    Returns 0 if either marginal is degenerate (0 or 1).
    """
    denom = np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    if denom <= 1e-12:
        return 0.0
    return float((p_ab - p_a * p_b) / denom)


# ──────────────────────────────────────────────────────────────────────────────
# Panel assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_default_panel(
    events: pd.DataFrame,
    borrower_col: str = "person_id",
    time_col: str = "period",
    default_col: str = "default",
    pd_col: str = "model_pd",
    segment_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate and assemble a borrower-period default panel.

    Parameters
    ----------
    events : DataFrame
        Long table with one row per (borrower, period) carrying a 0/1 default
        indicator and (optionally) a PD estimate and a segment id.
    borrower_col, time_col, default_col, pd_col, segment_col : str
        Column names (``segment_col`` optional).

    Returns
    -------
    DataFrame with canonical columns ``person_id, period, default[, model_pd,
    segment]`` — validated (binary default, sorted).
    """
    required = [borrower_col, time_col, default_col]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"events missing required columns: {missing}")

    out = pd.DataFrame({
        "person_id": events[borrower_col].to_numpy(),
        "period": events[time_col].to_numpy(),
        "default": events[default_col].to_numpy(),
    })
    uniq = set(np.unique(out["default"].to_numpy()))
    if not uniq.issubset({0, 1}):
        raise ValueError(f"default column must be binary 0/1, found values {uniq}")
    if pd_col in events.columns:
        out["model_pd"] = np.clip(events[pd_col].to_numpy(), 0.0, 1.0)
    if segment_col is not None and segment_col in events.columns:
        out["segment"] = events[segment_col].to_numpy()
    return out.sort_values(["period", "person_id"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Empirical dependence measures
# ──────────────────────────────────────────────────────────────────────────────

def _default_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Pivot the panel to a (borrowers × periods) 0/1 default matrix.

    Returns ``(matrix, borrower_ids)``. Missing (borrower, period) cells are 0.
    """
    pivot = panel.pivot_table(
        index="person_id", columns="period", values="default",
        aggfunc="max", fill_value=0,
    )
    return pivot.to_numpy().astype(float), pivot.index.to_numpy()


def _sample_pairs(n: int, max_pairs: int,
                  rng: np.random.Generator) -> List[tuple]:
    """
    Up to ``max_pairs`` distinct borrower index pairs ``(i, j)`` with ``i < j``.

    For small ``n`` (where ``n(n-1)/2 <= max_pairs``) every pair is returned. For
    large ``n`` the pairs are drawn DIRECTLY (rejection sampling on ``i != j``)
    so the full ``O(n²)`` combination list is never materialised — important on
    the scalable path where ``n`` can be tens of thousands.
    """
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return list(combinations(range(n), 2))

    seen: set = set()
    # Oversample a little to absorb collisions, then trim to max_pairs.
    while len(seen) < max_pairs:
        a = rng.integers(0, n, size=max_pairs)
        b = rng.integers(0, n, size=max_pairs)
        for i, j in zip(a.tolist(), b.tolist()):
            if i != j:
                seen.add((i, j) if i < j else (j, i))
                if len(seen) >= max_pairs:
                    break
    return list(seen)


def _pairwise_dependence(default_mat: np.ndarray, max_pairs: int = 5000,
                         rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Average pairwise default-dependence stats over borrower pairs.

    For scale, if there are more than ``max_pairs`` borrower pairs a random
    sample of pairs is used (drawn directly, without enumerating all pairs). Each
    borrower is the time series of its default indicator across periods.

    Returns the aggregate stats plus the stacked per-pair default series
    (``x_series``/``y_series``) so the caller can compute rank measures
    (Kendall τ / Spearman ρ / Schweizer-Wolff) on the SAME borrower pairs — i.e.
    on genuine co-default observations rather than an artefact of row ordering.
    """
    n = default_mat.shape[0]
    if n < 2 or default_mat.shape[1] < 2:
        return {"default_corr": 0.0, "observed_joint_default": 0.0,
                "independent_joint_default": 0.0, "tail_codefault_rate": 0.0,
                "n_pairs": 0, "x_series": np.zeros(0), "y_series": np.zeros(0)}

    rng = rng or np.random.default_rng(0)
    pairs = _sample_pairs(n, max_pairs, rng)

    corrs, joint_obs, joint_ind, tail = [], [], [], []
    xs, ys = [], []
    for i, j in pairs:
        di, dj = default_mat[i], default_mat[j]
        p_i, p_j = di.mean(), dj.mean()
        p_ij = float(np.mean((di == 1) & (dj == 1)))
        corrs.append(default_correlation(p_i, p_j, p_ij))
        joint_obs.append(p_ij)
        joint_ind.append(p_i * p_j)
        # Lower-tail co-default: P(both default | at least one defaults).
        either = float(np.mean((di == 1) | (dj == 1)))
        tail.append(p_ij / either if either > 0 else 0.0)
        # Period-aligned default series for this borrower pair (the axis that
        # actually carries co-default dependence).
        xs.append(di)
        ys.append(dj)

    return {
        "default_corr": float(np.mean(corrs)),
        "observed_joint_default": float(np.mean(joint_obs)),
        "independent_joint_default": float(np.mean(joint_ind)),
        "tail_codefault_rate": float(np.mean(tail)),
        "n_pairs": len(pairs),
        "x_series": np.concatenate(xs) if xs else np.zeros(0),
        "y_series": np.concatenate(ys) if ys else np.zeros(0),
    }


def empirical_dependence_measures(
    panel: pd.DataFrame,
    segment_col: Optional[str] = None,
    max_pairs: int = 5000,
) -> pd.DataFrame:
    """
    Compute empirical default-dependence measures, overall and per segment.

    Returns one row per segment (plus an ``__ALL__`` row) with columns:
    ``segment, n_borrowers, n_periods, n_defaults, observed_joint_default,
    independent_joint_default, default_corr, kendall_tau, spearman_rho,
    schweizer_wolff, tail_codefault_rate, warnings``.
    """
    from scipy.stats import kendalltau, spearmanr

    def _one(sub: pd.DataFrame, label) -> Dict:
        mat, _ = _default_matrix(sub)
        n_borrowers, n_periods = mat.shape
        warns = []
        if n_borrowers < 2:
            warns.append("fewer than 2 borrowers")
        if n_periods < 2:
            warns.append("fewer than 2 periods")
        n_defaults = int(sub["default"].sum())
        if n_defaults < 5:
            warns.append("fewer than 5 defaults — estimates unstable")

        dep = _pairwise_dependence(mat, max_pairs=max_pairs)

        # Rank measures (Kendall τ / Spearman ρ / Schweizer-Wolff) on the
        # period-aligned default series of SAMPLED BORROWER PAIRS — i.e. on
        # genuine co-default observations (borrower A's default in period t vs
        # borrower B's default in period t), NOT on a row-ordering artefact.
        x = dep["x_series"]
        y = dep["y_series"]
        if x.size >= 2 and x.std() > 0 and y.std() > 0:
            tau = kendalltau(x, y).statistic
            rho = spearmanr(x, y).statistic
            sw = schweizer_wolff(np.column_stack((x, y)))
            tau = 0.0 if tau is None or np.isnan(tau) else float(tau)
            rho = 0.0 if rho is None or np.isnan(rho) else float(rho)
        else:
            tau = rho = sw = 0.0

        return {
            "segment": label,
            "n_borrowers": n_borrowers,
            "n_periods": n_periods,
            "n_defaults": n_defaults,
            "observed_joint_default": round(dep["observed_joint_default"], 6),
            "independent_joint_default": round(dep["independent_joint_default"], 6),
            "default_corr": round(dep["default_corr"], 4),
            "kendall_tau": round(tau, 4),
            "spearman_rho": round(rho, 4),
            "schweizer_wolff": round(float(sw), 4),
            "tail_codefault_rate": round(dep["tail_codefault_rate"], 4),
            "warnings": "; ".join(warns) if warns else "",
        }

    rows = [_one(panel, "__ALL__")]
    seg = segment_col if (segment_col and segment_col in panel.columns) else (
        "segment" if "segment" in panel.columns else None
    )
    if seg is not None:
        for val, sub in panel.groupby(seg):
            rows.append(_one(sub, val))
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration + goodness-of-fit
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Outcome of empirical copula calibration."""
    recommended_family: str
    recommended_params: Dict[str, float]
    family_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    empirical: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def calibrate_copula(
    panel: pd.DataFrame,
    family: str = "auto",
    segment_col: Optional[str] = None,
    max_pairs: int = 5000,
) -> CalibrationResult:
    """
    Calibrate copula parameters from an observed default panel.

    Maps the overall empirical dependence to Gaussian / Student-t / Clayton
    parameters, scores each family by how well its model joint-default rate
    matches the observed one, and recommends a family.

    Parameters
    ----------
    panel : DataFrame
        Output of ``build_default_panel``.
    family : {"auto", "gaussian", "student_t", "clayton"}
        Force a family, or "auto" to recommend by goodness-of-fit.
    segment_col : str, optional
        If given, the empirical measures are also broken out per segment (the
        recommendation still uses the overall measures).
    max_pairs : int
        Pair-sampling cap for the dependence measures (scale).

    Returns
    -------
    CalibrationResult
    """
    measures = empirical_dependence_measures(panel, segment_col=segment_col, max_pairs=max_pairs)
    overall = measures[measures["segment"] == "__ALL__"].iloc[0].to_dict()

    tau = float(overall["kendall_tau"])
    obs_joint = float(overall["observed_joint_default"])
    warnings: List[str] = []
    if overall["warnings"]:
        warnings.append(str(overall["warnings"]))

    # Marginal default rate (for the model joint-default reconstruction).
    p_marg = float(panel["default"].mean())

    families: Dict[str, Dict[str, float]] = {
        "gaussian": {"rho": gaussian_rho_from_tau(tau)},
        "student_t": {"rho": gaussian_rho_from_tau(tau), "nu": 6.0},
        "clayton": {"theta": clayton_theta_from_tau(tau)},
    }

    # Model joint-default rate per family, via a quick Monte-Carlo of the
    # bivariate copula at the marginal default rate.
    rng = np.random.default_rng(0)

    def _model_joint(fam: str, params: Dict[str, float], n: int = 20000) -> float:
        from scipy.stats import norm
        thr = norm.ppf(p_marg) if 0 < p_marg < 1 else (-np.inf if p_marg == 0 else np.inf)
        if fam in ("gaussian", "student_t"):
            rho = params["rho"]
            cov = np.array([[1.0, rho], [rho, 1.0]])
            z = rng.multivariate_normal([0, 0], cov, size=n)
            if fam == "student_t":
                nu = params["nu"]
                g = rng.chisquare(nu, size=n) / nu
                z = z / np.sqrt(g)[:, None]
                from scipy.stats import t as student
                thr = student.ppf(p_marg, df=nu) if 0 < p_marg < 1 else thr
                d = z <= thr
            else:
                d = z <= thr
            return float(np.mean(d[:, 0] & d[:, 1]))
        else:  # clayton (lower-tail) via conditional sampling
            theta = params["theta"]
            u = rng.uniform(size=n)
            w = rng.uniform(size=n)
            # Clayton conditional inverse: v = (u^{-θ} (w^{-θ/(1+θ)} − 1) + 1)^{-1/θ}
            v = (u ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
            d0 = u <= p_marg
            d1 = v <= p_marg
            return float(np.mean(d0 & d1))

    rows = []
    for fam, params in families.items():
        model_joint = _model_joint(fam, params)
        # Fréchet bounds: max(p_A+p_B−1, 0) ≤ p_AB ≤ min(p_A, p_B).
        frechet_lo = max(2 * p_marg - 1, 0.0)
        frechet_hi = p_marg
        within = frechet_lo - 1e-6 <= model_joint <= frechet_hi + 1e-6
        rows.append({
            "family": fam,
            **{k: round(v, 4) for k, v in params.items()},
            "model_joint_default": round(model_joint, 6),
            "observed_joint_default": round(obs_joint, 6),
            "abs_error": round(abs(model_joint - obs_joint), 6),
            "frechet_ok": bool(within),
        })
    family_table = pd.DataFrame(rows).sort_values("abs_error").reset_index(drop=True)

    if family == "auto":
        best = family_table.iloc[0]
        recommended = str(best["family"])
    else:
        if family not in families:
            raise ValueError(f"family must be one of {list(families) + ['auto']}")
        recommended = family
    recommended_params = families[recommended]

    if tau <= 0:
        warnings.append("non-positive Kendall τ — data shows little/no positive dependence")

    return CalibrationResult(
        recommended_family=recommended,
        recommended_params=recommended_params,
        family_table=family_table,
        empirical=overall,
        warnings=warnings,
    )
