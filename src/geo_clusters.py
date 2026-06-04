"""
geo_clusters.py — Geolocation clustering for the credit-risk framework.

ROLE: Data Scientist (dependence structure). See ROLES.md.

Turns per-person coordinates (geo_longitude, geo_latitude) into a
`geo_cluster_id` that can be used as:

  * a SYSTEMATIC FACTOR in the (multi-)factor copula — people in the same
    geographic cluster share a regional risk driver (local economy, employer,
    natural disaster, housing market), so their defaults are correlated; and
  * a SEGMENT for risk-metric roll-ups (EAD / EL / σ(Loss) / CoV / RAROC /
    Sortino per geo cluster, via RiskRatioCalculator.by_segment).

Design
------
- Density-based **DBSCAN** with the **haversine** metric on (lat, lon) in
  radians → clusters of ARBITRARY SIZE and shape, with no fixed `k`. This is
  exactly the "differently sized geo clusters" requirement.
- `eps` is specified in **kilometres** (converted to radians internally), so the
  knob is human-meaningful ("cluster people within ~5 km").
- DBSCAN noise points (label -1) are given their OWN unique negative cluster ids
  so that, as a copula factor, they read as "no shared geo factor" (independent)
  rather than all being lumped into one giant pseudo-cluster.
- **Fallback**: if coordinates are absent, fall back to `city_id` (or a single
  cluster) so the rest of the pipeline still runs.
- Multi-resolution: call `fit` with different `eps_km` to get coarse/fine
  clusters that nest; or use `level="city"` for the coarse administrative level.

Everything is SAVED-friendly: `assign()` returns the augmented persons frame and
`summary()` returns a per-cluster table you can write to CSV.

No new dependencies — uses scikit-learn (already required).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - sklearn is a hard dependency elsewhere
    _HAS_SKLEARN = False


# Mean Earth radius (km) — used to convert eps in km ↔ radians for haversine.
EARTH_RADIUS_KM = 6371.0088


@dataclass
class GeoClusterConfig:
    """Configuration for geolocation clustering."""
    eps_km: float = 5.0          # neighbourhood radius in kilometres
    min_samples: int = 5         # DBSCAN core-point threshold
    level: str = "dbscan"        # "dbscan" | "city"  (coarse administrative)
    lon_col: str = "geo_longitude"
    lat_col: str = "geo_latitude"
    city_col: str = "city_id"
    out_col: str = "geo_cluster_id"


class GeoClusterer:
    """
    Assign each person a geographic cluster id.

    Usage
    -----
        gc = GeoClusterer(GeoClusterConfig(eps_km=5.0, min_samples=5))
        persons = gc.fit(persons).assign(persons)   # adds 'geo_cluster_id'
        summary = gc.summary()                       # per-cluster table → CSV

    Notes
    -----
    `geo_cluster_id` semantics for the copula factor:
        >= 0   → a genuine shared geo cluster (correlated)
        <  0   → unique negative id per isolated/noise person (independent)
    """

    def __init__(self, config: Optional[GeoClusterConfig] = None) -> None:
        self.config = config or GeoClusterConfig()
        self.labels_: Optional[np.ndarray] = None
        self.method_: Optional[str] = None       # which path actually ran
        self._person_ids: Optional[np.ndarray] = None
        self._coords: Optional[np.ndarray] = None  # (n,2) lat,lon degrees or None

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, persons: pd.DataFrame) -> "GeoClusterer":
        """Compute geo cluster labels from persons' coordinates (or city
        fallback). Stores `labels_`; call `assign`/`summary` afterwards."""
        cfg = self.config
        n = len(persons)
        self._person_ids = (
            persons["person_id"].to_numpy() if "person_id" in persons.columns
            else np.arange(n)
        )

        has_coords = (
            cfg.lon_col in persons.columns and cfg.lat_col in persons.columns
        )

        if cfg.level == "city" or not has_coords:
            self.labels_ = self._fit_city_fallback(persons)
            self.method_ = "city" if (cfg.level == "city" or not has_coords) else self.method_
            self._coords = None
            return self

        lon = pd.to_numeric(persons[cfg.lon_col], errors="coerce").to_numpy()
        lat = pd.to_numeric(persons[cfg.lat_col], errors="coerce").to_numpy()
        self._coords = np.column_stack([lat, lon])

        valid = np.isfinite(lon) & np.isfinite(lat)
        if valid.sum() < cfg.min_samples or not _HAS_SKLEARN:
            # Not enough usable coordinates → fall back to city grouping.
            self.labels_ = self._fit_city_fallback(persons)
            self.method_ = "city_fallback"
            return self

        labels = np.full(n, -1, dtype=np.int64)
        coords_rad = np.radians(np.column_stack([lat[valid], lon[valid]]))
        eps_rad = cfg.eps_km / EARTH_RADIUS_KM
        db = DBSCAN(
            eps=eps_rad, min_samples=cfg.min_samples, metric="haversine"
        ).fit(coords_rad)
        labels[valid] = db.labels_  # -1 = noise within the valid subset

        self.labels_ = self._explode_noise(labels)
        self.method_ = "dbscan"
        return self

    # ── assignment & summary ────────────────────────────────────────────────
    def assign(self, persons: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `persons` with the geo cluster id column added."""
        if self.labels_ is None:
            raise RuntimeError("GeoClusterer.fit() must be called before assign().")
        out = persons.copy()
        out[self.config.out_col] = self.labels_
        return out

    def summary(self) -> pd.DataFrame:
        """
        Per-cluster table: cluster_id, n_members, centroid_lat/lon, span_km.
        Only genuine clusters (id >= 0) get a centroid; isolated points are
        aggregated into a single 'noise' summary row for readability.
        """
        if self.labels_ is None:
            raise RuntimeError("GeoClusterer.fit() must be called before summary().")

        labels = self.labels_
        rows = []
        genuine = np.unique(labels[labels >= 0])
        for cid in genuine:
            mask = labels == cid
            row = {"geo_cluster_id": int(cid), "n_members": int(mask.sum())}
            if self._coords is not None:
                pts = self._coords[mask]
                pts = pts[np.isfinite(pts).all(axis=1)]
                if len(pts):
                    row["centroid_lat"] = float(pts[:, 0].mean())
                    row["centroid_lon"] = float(pts[:, 1].mean())
                    row["span_km"] = float(self._span_km(pts))
            rows.append(row)

        n_noise = int((labels < 0).sum())
        if n_noise:
            rows.append({
                "geo_cluster_id": -1,
                "n_members": n_noise,
                "centroid_lat": np.nan,
                "centroid_lon": np.nan,
                "span_km": np.nan,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("n_members", ascending=False).reset_index(drop=True)
        return df

    # ── internals ────────────────────────────────────────────────────────────
    def _fit_city_fallback(self, persons: pd.DataFrame) -> np.ndarray:
        cfg = self.config
        if cfg.city_col in persons.columns:
            codes, _ = pd.factorize(persons[cfg.city_col])
            return codes.astype(np.int64)
        # No geography at all → single cluster.
        return np.zeros(len(persons), dtype=np.int64)

    @staticmethod
    def _explode_noise(labels: np.ndarray) -> np.ndarray:
        """
        Replace DBSCAN noise (-1) with unique NEGATIVE ids so each isolated
        person reads as 'no shared geo factor' (independent) in the copula,
        instead of all noise collapsing into one spurious cluster.
        """
        out = labels.astype(np.int64).copy()
        noise_pos = np.flatnonzero(out == -1)
        # -1, -2, -3, ... unique per noise point
        out[noise_pos] = -1 - np.arange(len(noise_pos), dtype=np.int64)
        return out

    @staticmethod
    def _span_km(latlon_deg: np.ndarray) -> float:
        """Max pairwise great-circle distance (km) within a small cluster.
        For large clusters, approximate via bounding-box diagonal to stay O(n)."""
        if len(latlon_deg) <= 1:
            return 0.0
        if len(latlon_deg) > 2000:
            lat = latlon_deg[:, 0]
            lon = latlon_deg[:, 1]
            return GeoClusterer._haversine_km(
                lat.min(), lon.min(), lat.max(), lon.max()
            )
        lat = np.radians(latlon_deg[:, 0])
        lon = np.radians(latlon_deg[:, 1])
        # pairwise haversine, take max; small clusters only
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
        d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return float(d.max())

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


__all__ = ["GeoClusterer", "GeoClusterConfig", "EARTH_RADIUS_KM"]
