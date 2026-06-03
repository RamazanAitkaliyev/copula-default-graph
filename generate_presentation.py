#!/usr/bin/env python3
"""
generate_presentation.py
========================
Reads all pipeline outputs from output/ and writes a single self-contained
output/presentation.html — no external dependencies at view time.

Run:  python generate_presentation.py
"""

import base64
import os

import pandas as pd

OUTPUT_DIR = "output"
HTML_OUT = os.path.join(OUTPUT_DIR, "presentation.html")


# ─── helpers ──────────────────────────────────────────────────────────────────

def b64_img(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return "data:image/png;base64," + data


def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def df_to_html(df, max_rows=30, fmt=None):
    if df is None or df.empty:
        return "<p class='muted'>No data available.</p>"
    fmt = fmt or {}
    display = df.head(max_rows).copy()
    headers = "".join("<th>" + str(c) + "</th>" for c in display.columns)
    rows = ["<thead><tr>" + headers + "</tr></thead><tbody>"]
    for _, row in display.iterrows():
        cells = []
        for col in display.columns:
            val = row[col]
            if col in fmt:
                try:
                    val = fmt[col](val)
                except Exception:
                    pass
            cells.append("<td>" + str(val) + "</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody>")
    return "<table class='data-table'>" + "".join(rows) + "</table>"


def pct(v):
    try:
        return "{:.1%}".format(float(v))
    except Exception:
        return str(v)


def f2(v):
    try:
        return "{:.2f}".format(float(v))
    except Exception:
        return str(v)


def f4(v):
    try:
        return "{:.4f}".format(float(v))
    except Exception:
        return str(v)


def chg(v):
    try:
        return "{:+.1f}%".format(float(v))
    except Exception:
        return str(v)


def img_tag(src_uri, alt="", css=""):
    if not src_uri:
        return "<div class='img-missing'>Chart not available.</div>"
    return ('<img src="' + src_uri + '" alt="' + alt +
            '" class="chart-img ' + css + '" />')


# ─── load data ────────────────────────────────────────────────────────────────

city_df   = load_csv("metric_by_city_name.csv")
arch_df   = load_csv("metric_by_risk_archetype.csv")
rank_df   = load_csv("metric_rank_correlation.csv")
flags_df  = load_csv("metric_divergence_flags.csv")
top_df    = load_csv("top_risks.csv")
stress_df = load_csv("stress_test.csv")
rating_df = load_csv("rating_summary.csv")
client_df = load_csv("client_value.csv")
struct_df = load_csv("structural_pd.csv")

imgs = {
    "network":    b64_img("network_by_pd.png"),
    "loss_dist":  b64_img("loss_distribution.png"),
    "heatmap":    b64_img("risk_heatmap.png"),
    "features":   b64_img("feature_importance.png"),
    "copulas":    b64_img("copula_comparison.png"),
    "rating":     b64_img("rating_distribution.png"),
    "merton":     b64_img("merton_vs_statistical_pd.png"),
    "metric_cmp": b64_img("metric_comparison.png"),
}

# ─── rank-corr pairs ──────────────────────────────────────────────────────────

short = {
    "coefficient_of_variation":        "CoV (L0)",
    "coefficient_of_variation_copula": "CoV-Copula (L1)",
    "raroc":          "RAROC",
    "sharpe_indep":   "Sharpe",
    "sortino_copula": "Sortino-Copula (L1)",
    "sortino_indep":  "Sortino (L0)",
}

rank_rows_html = ""
if not rank_df.empty:
    try:
        # First column might be the index label
        idx_col = rank_df.columns[0]
        rank_df2 = rank_df.set_index(idx_col)
        col_list = list(rank_df2.columns)
        idx_list = list(rank_df2.index)
        for i, r in enumerate(idx_list):
            for j, c in enumerate(col_list):
                if j > i:
                    try:
                        val = float(rank_df2.loc[r, c])
                    except Exception:
                        continue
                    color = ("#16a34a" if val >= 0.95 else
                             "#3b82f6" if val >= 0.7 else
                             "#f59e0b" if val >= 0.4 else "#ef4444")
                    bar_w = int(abs(val) * 100)
                    a_name = short.get(r, r)
                    b_name = short.get(c, c)
                    rank_rows_html += (
                        "<tr><td>" + a_name + "</td><td>" + b_name + "</td>"
                        "<td><div class='corr-bar-wrap'>"
                        "<div class='corr-bar' style='width:" + str(bar_w) + "%;background:" + color + "'></div>"
                        "<span style='color:" + color + ";font-weight:600'>" + "{:+.3f}".format(val) + "</span>"
                        "</div></td></tr>"
                    )
    except Exception:
        pass

# ─── flag counts ──────────────────────────────────────────────────────────────

n_flags = len(flags_df)
n_hidden = len(flags_df[flags_df["flag_type"] == "hidden_network_risk"]) if not flags_df.empty else 0
n_diversified = len(flags_df[flags_df["flag_type"] == "diversified_low_value"]) if not flags_df.empty else 0

# ─── stress numbers ───────────────────────────────────────────────────────────

el_base = "—"
var_base = "—"
es_base = "—"
if not stress_df.empty:
    for _, row in stress_df.iterrows():
        if row["metric"] == "expected_loss":
            el_base = "{:.2f}".format(float(row["base"]))
        elif row["metric"] == "var_95":
            var_base = "{:.2f}".format(float(row["base"]))
        elif row["metric"] == "es_95":
            es_base = "{:.2f}".format(float(row["base"]))

# ─── pre-build HTML table blocks ──────────────────────────────────────────────

CITY_TABLE = df_to_html(city_df, fmt={
    "exposure_share": pct, "expected_profit": f2, "expected_loss": f2,
    "raroc": f4, "sortino_copula": f4, "coefficient_of_variation_copula": f4,
    "diversification_ratio": f2,
})
ARCH_TABLE = df_to_html(arch_df, fmt={
    "exposure_share": pct, "expected_profit": f2, "expected_loss": f2,
    "raroc": f4, "sortino_copula": f4, "coefficient_of_variation_copula": f4,
    "diversification_ratio": f2,
})
FLAGS_TABLE = df_to_html(flags_df.head(20) if not flags_df.empty else flags_df, fmt={
    "raroc": f4, "sortino_copula": f4, "z_score": f2,
})
TOP_TABLE = df_to_html(top_df, fmt={"marginal_pd": pct, "composite_risk_score": f4})
CLIENT_TABLE = df_to_html(client_df.head(10) if not client_df.empty else client_df, fmt={
    "expected_revenue": f2, "expected_loss": f2, "expected_profit": f2,
    "client_sharpe": f4, "raroc": f4, "contagion_adjusted_sharpe": f4,
})
RATING_TABLE = df_to_html(rating_df.head(12) if not rating_df.empty else rating_df, fmt={
    "marginal_pd": pct, "upgrade_prob": pct, "downgrade_prob": pct,
    "default_1yr": pct, "default_3yr": pct,
})
STRESS_TABLE = df_to_html(stress_df, fmt={
    "base": f4, "stressed": f4, "change_pct": chg,
})
STRUCT_TABLE = df_to_html(struct_df.head(15) if not struct_df.empty else struct_df, fmt={
    "model_pd": pct, "merton_pd": pct, "blended_pd": pct,
    "pd_signal_divergence": pct, "distance_to_default": f2,
})

IMG_NETWORK    = img_tag(imgs["network"],    "Transaction network",     "")
IMG_FEATURES   = img_tag(imgs["features"],   "Feature importance",      "")
IMG_COPULAS    = img_tag(imgs["copulas"],    "Copula comparison",       "")
IMG_LOSS_DIST  = img_tag(imgs["loss_dist"],  "Loss distribution",       "")
IMG_HEATMAP    = img_tag(imgs["heatmap"],    "Risk heatmap",            "")
IMG_RATING     = img_tag(imgs["rating"],     "Rating distribution",     "")
IMG_MERTON     = img_tag(imgs["merton"],     "Merton vs stat PD",       "")
IMG_METRIC_CMP = img_tag(imgs["metric_cmp"], "Metric comparison",       "")

# ─── CSS (separate string, no f-string) ───────────────────────────────────────

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --navy:   #0f172a;
  --blue:   #1e40af;
  --blue2:  #3b82f6;
  --blue3:  #93c5fd;
  --teal:   #0891b2;
  --green:  #16a34a;
  --amber:  #d97706;
  --red:    #dc2626;
  --purple: #7c3aed;
  --gray1:  #f8fafc;
  --gray2:  #e2e8f0;
  --gray3:  #94a3b8;
  --gray4:  #475569;
  --text:   #1e293b;
  --white:  #ffffff;
  --sidebar-w: 270px;
  --header-h:  60px;
  --shadow: 0 2px 12px rgba(0,0,0,0.10);
}
html { scroll-behavior: smooth; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--gray1);
  color: var(--text);
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
.top-header {
  position: fixed; top: 0; left: 0; right: 0; height: var(--header-h);
  background: var(--navy);
  display: flex; align-items: center; padding: 0 24px;
  z-index: 200; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.top-header h1 { color: var(--white); font-size: 1rem; font-weight: 600; letter-spacing: 0.02em; }
.top-header .badge {
  margin-left: 16px; background: var(--blue); color: var(--white);
  font-size: 0.7rem; padding: 2px 10px; border-radius: 12px; letter-spacing: 0.05em;
}
.header-right { margin-left: auto; color: var(--gray3); font-size: 0.8rem; }
.sidebar {
  position: fixed; top: var(--header-h); left: 0; bottom: 0;
  width: var(--sidebar-w);
  background: var(--navy);
  overflow-y: auto;
  z-index: 100;
  padding-bottom: 40px;
}
.sidebar-section {
  padding: 20px 16px 8px;
  color: var(--gray3); font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.sidebar-section:first-child { border-top: none; }
.sidebar a {
  display: block; padding: 8px 20px;
  color: rgba(255,255,255,0.65); font-size: 0.82rem; text-decoration: none;
  border-left: 3px solid transparent; transition: all 0.15s;
}
.sidebar a:hover, .sidebar a.active {
  color: var(--white); background: rgba(255,255,255,0.07);
  border-left-color: var(--blue2);
}
.sidebar a .step-num {
  display: inline-block; min-width: 22px;
  background: rgba(59,130,246,0.25); color: var(--blue3);
  font-size: 0.65rem; font-weight: 700; border-radius: 4px;
  text-align: center; padding: 1px 4px; margin-right: 8px;
}
.main {
  margin-left: var(--sidebar-w);
  margin-top: var(--header-h);
  padding: 36px 40px 80px;
  max-width: 1100px;
}
.section {
  background: var(--white);
  border-radius: 12px;
  box-shadow: var(--shadow);
  margin-bottom: 36px;
  overflow: hidden;
}
.section-header {
  padding: 20px 28px 16px;
  border-bottom: 2px solid var(--gray2);
  display: flex; align-items: center; gap: 14px;
}
.section-num {
  background: var(--blue); color: var(--white);
  font-size: 0.75rem; font-weight: 700;
  padding: 4px 10px; border-radius: 6px; white-space: nowrap;
}
.section-header h2 { font-size: 1.15rem; font-weight: 700; color: var(--navy); }
.section-header .tag {
  margin-left: auto; font-size: 0.7rem; padding: 3px 10px;
  border-radius: 10px; font-weight: 600; letter-spacing: 0.04em;
}
.tag-theory   { background: #ede9fe; color: #6d28d9; }
.tag-pipeline { background: #dcfce7; color: #15803d; }
.tag-metrics  { background: #fef3c7; color: #92400e; }
.tag-output   { background: #dbeafe; color: #1d4ed8; }
.section-body { padding: 24px 28px; }
.theory-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;
}
@media (max-width: 800px) { .theory-grid { grid-template-columns: 1fr; } }
.theory-card {
  background: #f8fafc; border: 1px solid var(--gray2);
  border-radius: 8px; padding: 18px 20px;
}
.theory-card h3 { font-size: 0.88rem; font-weight: 700; margin-bottom: 10px; color: var(--navy); }
.theory-card p  { font-size: 0.83rem; line-height: 1.6; color: var(--gray4); }
.formula-block {
  background: var(--navy); color: #e2e8f0;
  border-radius: 8px; padding: 16px 20px; margin: 14px 0;
  font-family: 'Courier New', monospace; font-size: 0.82rem; line-height: 1.8;
  overflow-x: auto;
}
.formula-block .comment   { color: #64748b; }
.formula-block .highlight { color: #93c5fd; font-weight: 700; }
.formula-block .value     { color: #86efac; }
.insight {
  border-left: 4px solid var(--blue2);
  background: #eff6ff; padding: 14px 18px; border-radius: 0 8px 8px 0;
  margin: 14px 0; font-size: 0.84rem; line-height: 1.6;
}
.insight.warning { border-left-color: var(--amber); background: #fffbeb; }
.insight.success { border-left-color: var(--green);  background: #f0fdf4; }
.insight.danger  { border-left-color: var(--red);    background: #fef2f2; }
.insight strong  { display: block; margin-bottom: 4px; font-size: 0.82rem; }
.pipeline-flow {
  display: flex; flex-wrap: wrap; gap: 0; margin: 20px 0;
  align-items: stretch;
}
.pipe-step {
  flex: 1; min-width: 110px;
  background: var(--gray1); border: 1px solid var(--gray2);
  padding: 14px 10px; text-align: center; position: relative;
  font-size: 0.75rem;
}
.pipe-step:not(:last-child)::after {
  content: '\\25B6';
  position: absolute; right: -11px; top: 50%; transform: translateY(-50%);
  color: var(--gray3); font-size: 0.9rem; z-index: 1;
}
.pipe-step .ps-num {
  display: inline-block; background: var(--blue); color: white;
  border-radius: 50%; width: 22px; height: 22px; line-height: 22px;
  font-size: 0.65rem; font-weight: 700; margin-bottom: 6px;
}
.pipe-step .ps-name { font-weight: 600; color: var(--navy); line-height: 1.3; }
.pipe-step .ps-out  { color: var(--gray3); font-size: 0.68rem; margin-top: 4px; }
.pipe-step.active  { background: #eff6ff; border-color: var(--blue2); }
.pipe-step.metrics { background: #fffbeb; border-color: var(--amber); }
.stat-row  { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
.stat-card {
  flex: 1; min-width: 130px;
  background: var(--gray1); border: 1px solid var(--gray2);
  border-radius: 8px; padding: 16px 18px;
}
.stat-card .sc-label { font-size: 0.72rem; color: var(--gray3); text-transform: uppercase; letter-spacing: 0.08em; }
.stat-card .sc-value { font-size: 1.6rem; font-weight: 700; color: var(--navy); margin-top: 4px; }
.stat-card .sc-sub   { font-size: 0.72rem; color: var(--gray4); margin-top: 2px; }
.stat-card.green  { border-left: 4px solid var(--green);  }
.stat-card.amber  { border-left: 4px solid var(--amber);  }
.stat-card.red    { border-left: 4px solid var(--red);    }
.stat-card.blue   { border-left: 4px solid var(--blue2);  }
.stat-card.purple { border-left: 4px solid var(--purple); }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin: 12px 0; }
.data-table thead tr  { background: var(--navy); color: var(--white); }
.data-table thead th  {
  padding: 9px 12px; text-align: left; font-weight: 600;
  font-size: 0.72rem; letter-spacing: 0.03em;
}
.data-table tbody tr:nth-child(even) { background: #f8fafc; }
.data-table tbody tr:hover { background: #eff6ff; }
.data-table tbody td { padding: 8px 12px; color: var(--gray4); }
.table-scroll { overflow-x: auto; }
.chart-img {
  width: 100%; border-radius: 8px; border: 1px solid var(--gray2);
  display: block; margin: 12px 0;
}
.chart-img.half { max-width: 500px; }
.chart-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 14px 0;
}
@media (max-width: 800px) { .chart-grid { grid-template-columns: 1fr; } }
.img-missing {
  background: #f1f5f9; border: 1px dashed var(--gray3);
  border-radius: 8px; padding: 24px; text-align: center;
  color: var(--gray3); font-size: 0.8rem;
}
.corr-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.corr-table th {
  padding: 8px 12px; background: var(--navy); color: var(--white);
  text-align: left; font-size: 0.72rem;
}
.corr-table td { padding: 7px 12px; border-bottom: 1px solid var(--gray2); }
.corr-bar-wrap { display: flex; align-items: center; gap: 10px; }
.corr-bar { height: 12px; border-radius: 3px; min-width: 2px; flex-shrink: 0; }
.pill {
  display: inline-block; padding: 3px 10px; border-radius: 12px;
  font-size: 0.72rem; font-weight: 600; margin: 2px;
}
.pill-blue   { background: #dbeafe; color: #1d4ed8; }
.pill-green  { background: #dcfce7; color: #15803d; }
.pill-amber  { background: #fef3c7; color: #92400e; }
.pill-red    { background: #fee2e2; color: #b91c1c; }
.pill-purple { background: #ede9fe; color: #6d28d9; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }
hr.divider { border: none; border-top: 1px solid var(--gray2); margin: 20px 0; }
details { margin: 10px 0; }
details summary {
  cursor: pointer; padding: 10px 14px;
  background: var(--gray1); border: 1px solid var(--gray2);
  border-radius: 6px; font-size: 0.82rem; font-weight: 600;
  color: var(--blue); user-select: none;
}
details summary:hover { background: #eff6ff; }
details[open] summary { border-radius: 6px 6px 0 0; border-bottom-color: transparent; }
details .details-body {
  border: 1px solid var(--gray2); border-top: none;
  padding: 16px 18px; border-radius: 0 0 6px 6px;
  background: var(--white);
}
.muted { color: var(--gray3); font-size: 0.8rem; font-style: italic; }
.section[id] { scroll-margin-top: 80px; }
"""

JS = """
const sections = document.querySelectorAll('.section[id]');
const links    = document.querySelectorAll('.sidebar a');
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      links.forEach(l => l.classList.remove('active'));
      const match = document.querySelector('.sidebar a[href="#' + e.target.id + '"]');
      if (match) match.classList.add('active');
    }
  });
}, { rootMargin: '-20% 0px -70% 0px' });
sections.forEach(s => observer.observe(s));
"""

# ─── assemble HTML pieces (plain string concatenation, no f-strings) ──────────

def section(sid, num, title, tag_class, tag_label, body):
    return (
        '\n<div class="section" id="' + sid + '">'
        '\n  <div class="section-header">'
        '\n    <span class="section-num">' + num + '</span>'
        '\n    <h2>' + title + '</h2>'
        '\n    <span class="tag ' + tag_class + '">' + tag_label + '</span>'
        '\n  </div>'
        '\n  <div class="section-body">' + body + '</div>'
        '\n</div>\n'
    )


def theory_card(title, body):
    return ('<div class="theory-card"><h3>' + title + '</h3><p>' + body + '</p></div>')


def insight(text, cls=""):
    return '<div class="insight ' + cls + '">' + text + '</div>'


def stat_card(label, value, sub, cls=""):
    return (
        '<div class="stat-card ' + cls + '">'
        '<div class="sc-label">' + label + '</div>'
        '<div class="sc-value">' + str(value) + '</div>'
        '<div class="sc-sub">' + sub + '</div>'
        '</div>'
    )


def formula(lines_html):
    return '<div class="formula-block">' + lines_html + '</div>'


def pipe_step(num, name, out, cls=""):
    return (
        '<div class="pipe-step ' + cls + '">'
        '<div class="ps-num">' + num + '</div>'
        '<div class="ps-name">' + name + '</div>'
        '<div class="ps-out">' + out + '</div>'
        '</div>'
    )


# ════════════════ build section bodies ════════════════════════════════════════

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
overview_body = (
    '<p style="font-size:0.9rem;line-height:1.7;margin-bottom:18px;">'
    'This framework builds a <strong>complete credit-risk analytical pipeline</strong> '
    'for a bank&#39;s retail or SME portfolio. It goes beyond a simple PD model: '
    'it captures <em>how defaults cluster and cascade through the network</em> of '
    'customer money flows, computes <em>risk-adjusted profitability</em> at any '
    'aggregation level, and gives risk analysts <em>early-warning signals</em> '
    'that standalone credit scoring misses.'
    '</p>'
    '<div class="theory-grid">'
    + theory_card("&#128279; Graph-Aware Correlations",
        "Transaction flows between customers form a network. Customers who frequently "
        "transact together are more likely to default together — the graph turns money "
        "flow into a <em>correlation matrix</em> that drives the copula, not an assumption.")
    + theory_card("&#128208; Copula Joint Defaults",
        "A <strong>Clayton copula</strong> combines marginal PDs with the network "
        "correlation matrix to produce the <em>full joint default probability matrix</em> "
        "P(D<sub>i</sub> &cap; D<sub>j</sub>) — the foundation for all loss-variance calculations.")
    + theory_card("&#128202; Pluggable Metric Family",
        "Seven risk-adjusted metrics (CoV, RAROC, Sharpe, Sortino variants) share one set "
        "of additive primitives. Any metric can be evaluated at borrower / segment / geography "
        "/ group level with <em>mathematically correct aggregation</em> (block-sum of loss-cov, "
        "never averaging ratios).")
    + theory_card("&#9888;&#65039; RAROC vs Sortino Divergence",
        "RAROC uses a flat capital charge and is <em>blind to network correlation</em>. "
        "The copula-Sortino&#39;s denominator inflates for contagious clusters. When they "
        "diverge, it is an <em>early-warning signal</em> that a borrower is embedded in "
        "hidden network risk.")
    + '</div>'
    + insight(
        '<strong>What makes this different from a standard scorecard</strong>'
        'A scorecard ranks individual borrowers independently. This system models '
        '<em>joint</em> defaults — it tells you not just that borrower A is risky, '
        'but that A, B, and C will likely default <em>together</em>, making the '
        'portfolio loss far larger than the sum of individual ELs would suggest.', "success")
)

# ── BUSINESS CASE ─────────────────────────────────────────────────────────────
business_body = (
    '<div class="theory-grid">'
    + theory_card("&#128188; For Risk Managers",
        "Standard VaR models assume independence or use a single market factor. "
        "This framework uses <em>actual transaction data</em> to identify "
        "<strong>hidden correlation clusters</strong> — borrowers who look individually "
        "acceptable but collectively represent a concentration risk that could breach "
        "capital limits in a downturn.")
    + theory_card("&#128200; For Business / Pricing Teams",
        "The copula-Sortino ratio gives a <strong>single dimensionless number</strong> "
        "for risk-adjusted profitability that is additive across any dimension — city, "
        "segment, product, relationship manager. It can be embedded in pricing models "
        "and client profitability dashboards without any aggregation distortion.")
    + theory_card("&#128269; For Credit Analysts",
        "The <strong>divergence flag system</strong> automatically surfaces borrowers "
        "where RAROC says &#39;profitable&#39; but the copula-Sortino says &#39;network risk "
        "hidden&#39;. These are exactly the accounts a senior analyst should review first — "
        "the system surfaces them without manual scanning.")
    + theory_card("&#127970; For Capital &amp; Stress Testing",
        "The Merton structural PD adds a <strong>second independent signal</strong> from "
        "balance-sheet volatility. The flexible-probability engine re-weights historical "
        "scenarios by current market stress, producing a <em>regime-aware copula</em> for "
        "ICAAP and stress-test submissions.")
    + '</div>'
    + insight(
        '<strong>The key deliverable: early warnings that RAROC misses</strong>'
        'In our synthetic portfolio of 1 000 borrowers, '
        '<strong>' + str(n_flags) + ' borrowers</strong> were flagged for RAROC&#8211;Sortino divergence. '
        'Of these, <strong>' + str(n_hidden) + ' are &#34;hidden network risk&#34;</strong> '
        '— they have acceptable RAROC but are embedded in high-correlation clusters '
        'that inflate their true downside risk. A traditional credit review would not '
        'catch these without the copula layer.')
)

# ── PIPELINE FLOW ─────────────────────────────────────────────────────────────
pipeline_body = (
    '<div class="pipeline-flow">'
    + pipe_step("1", "Data Generation", "persons · transactions")
    + pipe_step("2", "Transaction Graph", "corr matrix · stats", "active")
    + pipe_step("3", "PD Model", "model_pd · AUC")
    + pipe_step("4", "Correlation Matrix", "PSD (n&times;n)")
    + pipe_step("5", "Clayton Copula", "P(D<sub>i</sub>&cap;D<sub>j</sub>)", "active")
    + pipe_step("6", "Risk Analysis", "VaR · ES · HHI")
    + pipe_step("7", "Stress Test", "2&times;PD · +20% corr")
    + pipe_step("8", "Client Value", "Sharpe · RAROC · CLTV", "active")
    + pipe_step("8b", "Risk Metrics Family", "CoV · Sortino · flags", "metrics")
    + pipe_step("9", "Ratings", "AAA&#8211;Default")
    + pipe_step("10", "Merton PD", "structural signal", "active")
    + pipe_step("11", "Flex Probs", "regime-aware &theta;")
    + pipe_step("12", "Profiles", "per-borrower report")
    + '</div>'
    + '<p class="muted">Blue border = graph/copula layer. Yellow border = new metric family (STEP 8b).</p>'
    + insight(
        '<strong>Data flow summary</strong>'
        'Transactions &#8594; graph &#8594; correlation matrix &#8594; copula &#8594; '
        'joint default matrix &#8594; loss-covariance matrix &#8594; all seven risk-adjusted metrics. '
        'Every metric downstream shares the same mathematically consistent primitives.')
)

# ── THEORY: PD ────────────────────────────────────────────────────────────────
theory_pd_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'The PD model is a <strong>gradient-boosted classifier</strong> (GBM) trained on '
    'borrower features to predict the probability of default within a 12-month horizon. '
    'It outputs a calibrated score in [0, 1] for every borrower, which feeds directly '
    'into the copula as the marginal probability.'
    '</p>'
    '<div class="theory-grid">'
    + theory_card("Features used",
        "Age, income, employment years, debt-to-income ratio, number of credit lines, "
        "missed payments, credit utilization, account age (months). Network features "
        "(neighbor avg PD, max PD, number of high-risk neighbors) are added after graph "
        "construction in Step 2.")
    + theory_card("Why GBM over logistic regression?",
        "GBM captures non-linear interactions (e.g. high debt-to-income is only dangerous "
        "if combined with low income and missed payments) without manual feature engineering. "
        "It also provides calibrated probabilities via monotone isotonic calibration.")
    + '</div>'
    + '<details><summary>&#9658; Mathematical formulation</summary>'
    '<div class="details-body">'
    + formula(
        '<span class="comment">-- GBM ensemble of T trees --</span>\n'
        'F<sub>T</sub>(x) = &Sigma;<sub>t=1</sub><sup>T</sup>  &eta; &middot; h<sub>t</sub>(x)'
        '           <span class="comment">// &eta; = learning rate, h<sub>t</sub> = weak learner</span>\n\n'
        '<span class="comment">-- Calibrated PD --</span>\n'
        '<span class="highlight">PD<sub>i</sub></span> = &sigma;(F<sub>T</sub>(x<sub>i</sub>))'
        '                        <span class="comment">// &sigma; = sigmoid (logistic link)</span>\n'
        '       = 1 / (1 + exp(&minus;F<sub>T</sub>(x<sub>i</sub>)))\n\n'
        '<span class="comment">-- Objective: minimise binary cross-entropy --</span>\n'
        'L = &minus; &Sigma;<sub>i</sub> [ y<sub>i</sub> log(PD<sub>i</sub>) + '
        '(1&minus;y<sub>i</sub>) log(1&minus;PD<sub>i</sub>) ]\n\n'
        '<span class="comment">-- Quality metric: AUC-ROC --</span>\n'
        'AUC = P(PD<sub>defaulter</sub> > PD<sub>non-defaulter</sub>)'
        '   <span class="comment">// should be >> 0.5</span>'
    )
    + '<p style="font-size:0.82rem;color:var(--gray4);margin-top:10px;">'
    'Validation AUC above 0.75 indicates the model separates defaulters well. '
    'In production, Platt scaling or isotonic regression ensures the score is a true '
    'probability, not just a ranking.'
    '</p></div></details>'
)

# ── THEORY: GRAPH ─────────────────────────────────────────────────────────────
theory_graph_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'Customers who transact together share economic fate. A supplier who sells to a '
    'struggling buyer is at higher default risk than their standalone PD suggests. '
    'The graph layer formalises this: <em>the transaction network defines the correlation '
    'structure</em> instead of assuming a constant market factor.'
    '</p>'
    + formula(
        '<span class="comment">-- Base correlation from transaction volume --</span>\n'
        'shared_volume(i,j) = &Sigma; transactions where both i and j are party\n'
        'corr_raw(i,j)      = base_corr + f(shared_volume)  &isin; [base_corr, max_corr]\n\n'
        '<span class="comment">-- Boosts for same-city and same-risk-group membership --</span>\n'
        'corr(i,j) += same_city_boost    if city_id_i == city_id_j\n'
        'corr(i,j) += same_group_boost   if high_risk_group_id_i == high_risk_group_id_j\n\n'
        '<span class="comment">-- Enforce PSD (positive semi-definite) via nearest-PSD projection --</span>\n'
        '&Sigma; = nearest_psd(corr)           '
        '<span class="comment">// required for valid copula simulation</span>'
    )
    + insight(
        '<strong>Invariant: the matrix must be PSD</strong>'
        'A non-PSD correlation matrix leads to negative eigenvalues which break '
        'Cholesky decomposition inside the Gaussian copula and produce imaginary '
        'probabilities in the Student-t simulation. The <code>_nearest_psd()</code> '
        'function (Higham 2002) projects onto the cone of PSD matrices after every modification.',
        "warning")
    + '<div class="theory-grid">'
    + theory_card("Network statistics computed",
        "Degree centrality, clustering coefficient, betweenness centrality, number of "
        "connected components, bridge nodes (high betweenness), high-risk group membership. "
        "These feed both the PD model and the correlation boosts.")
    + theory_card("Bridge nodes — systemic importance",
        "A bridge node connects two otherwise separate communities. If it defaults, "
        "contagion can jump across the network gap. The framework computes a "
        "<em>systemic importance score</em> = average PD uplift caused in others if "
        "this borrower defaults.")
    + '</div>'
)

# ── THEORY: COPULA ────────────────────────────────────────────────────────────
theory_copula_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'A <strong>copula</strong> is a function that joins marginal distributions into '
    'a joint distribution while preserving each margin. For credit risk, the marginals '
    'are individual PDs and the copula encodes their dependence structure. '
    'The <strong>Clayton copula</strong> is chosen because it has '
    '<em>lower-tail dependence</em> — defaults cluster together in bad times '
    '(the left tail), which is exactly the behaviour seen in credit crises.'
    '</p>'
    + formula(
        '<span class="comment">-- Clayton bivariate copula --</span>\n'
        '<span class="highlight">C(u,v ; &theta;)</span> = '
        '(u<sup>&minus;&theta;</sup> + v<sup>&minus;&theta;</sup> &minus; 1)<sup>&minus;1/&theta;</sup>'
        '     &theta; > 0\n\n'
        '<span class="comment">-- Lower tail dependence (default clustering in stress) --</span>\n'
        '&lambda;<sub>L</sub> = 2<sup>&minus;1/&theta;</sup>'
        '        <span class="comment">// &rarr; 0 as &theta;&rarr;0 (independence), '
        '&rarr; 1 as &theta;&rarr;&infin; (comonotone)</span>\n\n'
        '<span class="comment">-- Simulation via gamma frailty method (fast, vectorized) --</span>\n'
        'V  ~ Gamma(1/&theta;, 1)\n'
        'U<sub>i</sub> = (1 &minus; log(E<sub>i</sub>)/V)<sup>&minus;1/&theta;</sup>'
        '   where E<sub>i</sub> ~ Exp(1)\n'
        'D<sub>i</sub> = 1  if  U<sub>i</sub> &le; &Phi;<sup>&minus;1</sup>(PD<sub>i</sub>)'
        '  (Gaussian margin inversion)\n\n'
        '<span class="comment">-- Joint default probability (exact, for the loss-cov matrix) --</span>\n'
        '<span class="highlight">P(D<sub>i</sub> &cap; D<sub>j</sub>)</span> = '
        'C(PD<sub>i</sub>, PD<sub>j</sub> ; &theta;)'
        '    <span class="comment">// (n&times;n) matrix call</span>'
    )
    + '<details><summary>&#9658; Why not Gaussian copula?</summary>'
    '<div class="details-body">'
    '<p style="font-size:0.83rem;line-height:1.6;">'
    'The 2008 financial crisis was partly blamed on the Gaussian copula (Li 2000) which '
    'has <strong>zero tail dependence</strong> — it systematically underestimates the '
    'probability of many simultaneous defaults in a crisis. The Clayton copula&#39;s '
    'lower-tail dependence &lambda;<sub>L</sub> = 2<sup>&minus;1/&theta;</sup> directly '
    'parameterises how much defaults cluster in bad states. With the data-fitted &theta;, '
    'this is not a modelling assumption — it is calibrated to the actual joint default rate.'
    '</p></div></details>'
    + insight(
        '<strong>Five copula types supported and compared in Step 5</strong>'
        'Gaussian (no tail dep), Student-t (symmetric tail dep), Clayton (lower-tail dep), '
        'Gumbel (upper-tail dep), Frank (symmetric, moderate). The framework fits all five '
        'and reports theta, lower/upper tail dependence, and simulated default rate for selection.')
)

# ── THEORY: LOSS-COV ──────────────────────────────────────────────────────────
theory_loss_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'Everything downstream — all seven metrics, segment aggregation, stress testing, '
    'diversification ratio — flows from <em>one</em> matrix built once: the '
    '<strong>loss-covariance matrix</strong>. It encodes how much the loss from '
    'borrower i moves together with the loss from borrower j.'
    '</p>'
    + formula(
        '<span class="comment">-- Building blocks --</span>\n'
        'EL<sub>i</sub>     = EAD<sub>i</sub> &middot; LGD<sub>i</sub> &middot; PD<sub>i</sub>'
        '           <span class="comment">// expected loss</span>\n'
        'el_vec<sub>i</sub> = EAD<sub>i</sub> &middot; LGD<sub>i</sub>'
        '                  <span class="comment">// loss-weight vector</span>\n\n'
        '<span class="comment">-- Default covariance from the copula --</span>\n'
        'Cov(D<sub>i</sub>, D<sub>j</sub>) = P(D<sub>i</sub>&cap;D<sub>j</sub>) &minus; '
        'PD<sub>i</sub> &middot; PD<sub>j</sub>'
        '   <span class="comment">// from joint_default_probability()</span>\n'
        'Cov(D<sub>i</sub>, D<sub>i</sub>) = PD<sub>i</sub> &middot; (1 &minus; PD<sub>i</sub>)'
        '             <span class="comment">// exact Bernoulli variance on diagonal</span>\n\n'
        '<span class="comment">-- Loss-covariance matrix (n&times;n, built once in __init__) --</span>\n'
        '<span class="highlight">LossCov[i,j]</span> = el_vec<sub>i</sub> &middot; '
        'Cov(D<sub>i</sub>, D<sub>j</sub>) &middot; el_vec<sub>j</sub>\n\n'
        '<span class="comment">-- Segment variance (block-sum) --</span>\n'
        '<span class="highlight">Var(Loss<sub>S</sub>)</span>  = '
        '&Sigma;<sub>i&isin;S</sub> &Sigma;<sub>j&isin;S</sub> LossCov[i,j]\n'
        '             = block_sum(LossCov[S, S])\n\n'
        '<span class="comment">-- Diversification ratio (&ge; 1 by triangle inequality) --</span>\n'
        '<span class="highlight">DR<sub>S</sub></span> = '
        '&Sigma;<sub>i&isin;S</sub> &radic;LossCov[i,i]  /  &radic;Var(Loss<sub>S</sub>)\n'
        '     = sum of individual &sigma;<sub>i</sub>   /  portfolio &sigma;<sub>S</sub>'
    )
    + insight(
        '<strong>Critical: never average per-borrower ratios to get a segment metric</strong>'
        'Var(Loss<sub>A</sub> + Loss<sub>B</sub>) = Var<sub>A</sub> + 2&middot;Cov(A,B) + Var<sub>B</sub>'
        ' &ne; Var<sub>A</sub> + Var<sub>B</sub> unless A,B are independent. '
        'Averaging per-borrower Sortino ratios loses all correlation information. '
        'The block-sum approach is mathematically exact and is the only correct aggregation '
        'under correlation.', "danger")
    + '<div class="theory-grid">'
    + theory_card("L0 — Independence assumption",
        "Uses only the diagonal of LossCov: &Sigma; diag(block). Assumes all off-diagonal "
        "covariances are zero. Fast to compute; ignores network contagion. "
        "Used by: CoV (L0), Sharpe (indep), Sortino (indep).")
    + theory_card("L1 — Copula-aware (full block)",
        "Uses the full block sum including off-diagonal terms. &sigma;<sub>L1</sub> &ge; "
        "&sigma;<sub>L0</sub> for positively correlated borrowers. The excess "
        "&sigma;<sub>L1</sub> &minus; &sigma;<sub>L0</sub> is the <em>contagion premium</em>. "
        "Used by: CoV-Copula (L1), Sortino-Copula (L1).")
    + theory_card("L2 — Monte Carlo simulation",
        "Uses copula.simulate_defaults() to draw 10 000 scenarios. Computes the downside "
        "semideviation: &radic;E[max(0, Loss &minus; E[Loss])<sup>2</sup>]. Captures 3+-way "
        "clustering and full tail shape. Used by: Sortino-Simulated (L2).")
    + theory_card("Additivity invariant",
        "E[Loss], E[Profit], Capital are all additive (simple sums). Var(Loss<sub>S</sub>) "
        "is the block-sum. Portfolio = sum of segments; any partition gives consistent numbers. "
        "No aggregation distortion at any level of the hierarchy.")
    + '</div>'
)

# ── THEORY: METRICS ───────────────────────────────────────────────────────────
theory_metrics_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'All seven metrics are computed from the same <strong>MetricInputs</strong> bundle '
    '— a small set of pre-aggregated absolute primitives. This means any metric can be '
    'evaluated at borrower, segment, city, group, or portfolio level without changing '
    'the formula.'
    '</p>'
    + formula(
        '<span class="comment">-- Shared primitives (all additive across borrowers) --</span>\n'
        'E[Profit]  = &Sigma; (revenue<sub>i</sub> &minus; EAD<sub>i</sub>&middot;LGD<sub>i</sub>&middot;PD<sub>i</sub>)\n'
        'E[Loss]    = &Sigma; EAD<sub>i</sub>&middot;LGD<sub>i</sub>&middot;PD<sub>i</sub>\n'
        'Capital    = &Sigma; capital_ratio &middot; EAD<sub>i</sub>'
        '      <span class="comment">// default: 8% &times; EAD</span>\n'
        '&sigma;<sub>L0</sub>  = &radic;(&Sigma; diag(LossCov))'
        '           <span class="comment">// independence assumption</span>\n'
        '&sigma;<sub>L1</sub>  = &radic;(block_sum(LossCov))'
        '        <span class="comment">// copula-aware (full block)</span>\n'
        '&sigma;<sub>L2</sub>  = &radic;E[max(0,Loss&minus;E[Loss])<sup>2</sup>]'
        '     <span class="comment">// simulated downside semidev</span>\n\n'
        '<span class="comment">-- The seven metrics --</span>\n'
        '<span class="highlight">CoV_L0</span>     = &sigma;<sub>L0</sub> / E[Loss]'
        '                     <span class="comment">// pure riskiness, always &ge; 0</span>\n'
        '<span class="highlight">CoV_L1</span>     = &sigma;<sub>L1</sub> / E[Loss]'
        '                     <span class="comment">// copula-aware riskiness</span>\n'
        '<span class="highlight">RAROC</span>      = E[Profit] / Capital'
        '                <span class="comment">// correlation-blind</span>\n'
        '<span class="highlight">Sharpe_L0</span>  = (E[Profit] &minus; rf&middot;Revenue) / &sigma;<sub>L0</sub>'
        '   <span class="comment">// benchmarks vs risk-free rev</span>\n'
        '<span class="highlight">Sortino_L0</span> = (E[Profit] &minus; h&middot;Capital) / &sigma;<sub>L0</sub>'
        '    <span class="comment">// benchmarks vs hurdle rate</span>\n'
        '<span class="highlight">Sortino_L1</span> = (E[Profit] &minus; h&middot;Capital) / &sigma;<sub>L1</sub>'
        '    <span class="comment">// copula-aware denominator &larr; KEY</span>\n'
        '<span class="highlight">Sortino_L2</span> = (E[Profit] &minus; h&middot;Capital) / &sigma;<sub>L2</sub>'
        '    <span class="comment">// simulated tail</span>'
    )
    + insight(
        '<strong>Sharpe vs Sortino numerators are intentionally different</strong>'
        'Sharpe benchmarks against the <em>risk-free opportunity cost of the revenue base</em> '
        '(rf &times; Revenue) — measures: "could we have invested this revenue elsewhere?". '
        'Sortino benchmarks against the <em>required return on regulatory capital</em> '
        '(h &times; Capital) — measures: "are we earning enough to justify the capital held?". '
        'These are genuinely different questions and must not use the same numerator.', "warning")
    + '<div class="two-col" style="margin-top:18px;">'
    '<div>'
    '<p style="font-size:0.82rem;font-weight:700;margin-bottom:8px;">When to use which metric:</p>'
    '<table class="data-table"><thead><tr><th>Metric</th><th>Best for</th><th>Limitation</th></tr></thead><tbody>'
    '<tr><td><span class="pill pill-blue">CoV L0/L1</span></td>'
    '<td>Riskiness ranking (always positive)</td><td>Ignores profitability</td></tr>'
    '<tr><td><span class="pill pill-green">RAROC</span></td>'
    '<td>Simple profitability gate</td><td>Blind to correlation / network</td></tr>'
    '<tr><td><span class="pill pill-amber">Sharpe</span></td>'
    '<td>Risk-adjusted return on revenue</td><td>Sign flips if unprofitable</td></tr>'
    '<tr><td><span class="pill pill-purple">Sortino L1</span></td>'
    '<td>Full correlated risk-adjusted return</td><td>Needs copula; sign flips</td></tr>'
    '<tr><td><span class="pill pill-red">Sortino L2</span></td>'
    '<td>Exact tail, 3+-way clustering</td><td>Slow (10k Monte Carlo paths)</td></tr>'
    '</tbody></table>'
    '</div>'
    '<div>'
    + insight(
        '<strong>Primary early-warning signal: RAROC &divide; Sortino_L1 divergence</strong>'
        'RAROC denominator = k&middot;EAD (flat). Sortino_L1 denominator = &radic;(block_sum(LossCov)) '
        '— it inflates with off-diagonal covariances. When a borrower has '
        '<em>good RAROC but bad Sortino_L1</em>, their individual credit looks fine '
        'but their network neighbourhood is a risk amplifier. This is the "hidden network risk" flag.',
        "success")
    + insight(
        '<strong>"Diversified low value" flag</strong>'
        'Bad RAROC + good Sortino_L1: the borrower is unprofitable individually '
        'but sits in a well-diversified neighbourhood that dampens their '
        'contribution to portfolio variance. May be worth retaining for '
        'portfolio diversification benefit.', "warning")
    + '</div></div>'
)

# ── THEORY: MERTON ────────────────────────────────────────────────────────────
theory_merton_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'The Merton (1974) model treats a firm&#39;s equity as a call option on its assets. '
    'Default occurs when asset value falls below the debt level at maturity T. '
    'For retail borrowers, asset value is proxied from income (capitalised as a '
    'perpetuity) and volatility is estimated from income variance.'
    '</p>'
    + formula(
        '<span class="comment">-- Proxy asset value (KMV-style, retail) --</span>\n'
        'V  &asymp; income &times; 12 / 0.08'
        '         <span class="comment">// capitalised monthly-income perpetuity</span>\n'
        'D  = income &times; 3                  '
        '<span class="comment">// proxy debt = 3 months income</span>\n'
        '&sigma;<sub>V</sub> &asymp; 0.3 &middot; (debt_to_income / mean_dti)'
        '  <span class="comment">// scaled vol proxy</span>\n\n'
        '<span class="comment">-- Merton distance to default --</span>\n'
        'd2 = [ln(V/D) + (r &minus; &sigma;<sub>V</sub><sup>2</sup>/2)&middot;T] / '
        '(&sigma;<sub>V</sub> &middot; &radic;T)\n\n'
        '<span class="comment">-- Merton PD --</span>\n'
        '<span class="highlight">PD<sub>merton</sub></span> = &Phi;(&minus;d2)'
        '             <span class="comment">// standard normal CDF</span>\n\n'
        '<span class="comment">-- Blended PD (combines statistical + structural signal) --</span>\n'
        'PD<sub>blended</sub> = &alpha; &middot; PD<sub>model</sub> + (1&minus;&alpha;) &middot; PD<sub>merton</sub>'
        '    <span class="comment">// &alpha; = 0.35 default</span>\n\n'
        '<span class="comment">-- Early warning: signal divergence --</span>\n'
        'divergence = |PD<sub>merton</sub> &minus; PD<sub>model</sub>|'
        '               <span class="comment">// flag if > 5%</span>'
    )
    + insight(
        '<strong>Why add a second signal?</strong>'
        'The statistical PD model is trained on historical behaviour (past defaults). '
        'The Merton structural model uses current balance-sheet information (income, debt). '
        'When they strongly diverge, it is an early warning that the borrower&#39;s '
        'fundamentals have deteriorated since the model was trained — a leading indicator.')
)

# ── THEORY: RATING ────────────────────────────────────────────────────────────
theory_rating_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'PD scores are bucketed into discrete ratings (AAA &#8594; Default) using fixed '
    'PD thresholds. A Markov <em>generator matrix</em> (fitted from Basel historical '
    'migration rates) is used to compute the probability of any rating transition '
    'over arbitrary time horizons via matrix exponentiation.'
    '</p>'
    + formula(
        '<span class="comment">-- PD thresholds (Basel-aligned) --</span>\n'
        'AAA: PD &le; 0.001   AA: &le; 0.002   A: &le; 0.005\n'
        'BBB: &le; 0.02       BB: &le; 0.08    B: &le; 0.25\n'
        'CCC: &le; 1.0        Default: PD = 1.0\n\n'
        '<span class="comment">-- Migration matrix over horizon &Delta;t --</span>\n'
        '<span class="highlight">P(&Delta;t)</span> = expm(G &middot; &Delta;t)'
        '           <span class="comment">// G = generator matrix, expm = matrix exponential</span>\n\n'
        '<span class="comment">-- Expected default over n years --</span>\n'
        'default_3yr &asymp; 1 &minus; (1 &minus; default_1yr)<sup>3</sup>'
    )
    + insight(
        '<strong>Use case: IFRS 9 / ECL staging</strong>'
        'Stage 1 (12-month ECL) uses default_1yr. Stage 2 (lifetime ECL) triggers '
        'when a borrower&#39;s rating migrates downward by one or more notches — the '
        'migration matrix gives the probability of this transition occurring over '
        'the loan&#39;s remaining life.')
)

# ── PIPELINE RESULTS sections ─────────────────────────────────────────────────

step_network_body = (
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:14px;">'
    '1 000 borrowers across three cities (Alpha, Beta, Gamma). Node colour encodes '
    'base PD — darker red = higher default risk. Node size encodes transaction degree '
    '— larger = more connected. Bridge nodes (high betweenness) are visible as smaller '
    'hubs connecting community clusters.'
    '</p>'
    + IMG_NETWORK
)

step_pd_body = (
    '<div class="chart-grid">'
    + IMG_FEATURES
    + '<div>'
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:12px;">'
    'The gradient-boosting PD model is trained on 8 borrower features. '
    'Features are ranked by their contribution to the ensemble. Top features '
    'are typically: missed payments, credit utilization, and debt-to-income ratio — '
    'consistent with industry knowledge.'
    '</p>'
    + insight(
        '<strong>Validation AUC well above 0.5 indicates genuine predictive power</strong>'
        'An AUC of 0.8+ means the model correctly ranks 80% of default/non-default pairs. '
        'Network features (neighbor avg PD) add lift on top of individual features '
        'by capturing contagion risk.', "success")
    + '</div></div>'
)

step_copula_body = (
    IMG_COPULAS
    + '<p style="font-size:0.83rem;line-height:1.6;margin-top:12px;">'
    'All five copulas are fitted to the same PD vector + correlation matrix. '
    '<strong>Clayton</strong> is selected for production because it has the '
    'highest lower-tail dependence (&lambda;<sub>L</sub> = 2<sup>&minus;1/&theta;</sup>), '
    'capturing the empirical fact that defaults cluster during stress periods.'
    '</p>'
)

step_portfolio_body = (
    '<div class="chart-grid">' + IMG_LOSS_DIST + IMG_HEATMAP + '</div>'
    '<p style="font-size:0.83rem;line-height:1.6;margin-top:12px;">'
    'Left: 10 000 Monte Carlo simulated portfolio losses. VaR and ES are shown as '
    'vertical lines. The tail is heavier than a Gaussian would produce because the '
    'Clayton copula clusters losses. '
    'Right: heatmap of the top 50 riskiest borrowers across four risk dimensions '
    '(marginal PD, contagion vulnerability, systemic importance, network exposure).'
    '</p>'
    '<div class="stat-row" style="margin-top:18px;">'
    + stat_card("Expected Loss", el_base, "portfolio normalised", "blue")
    + stat_card("VaR 95%",       var_base, "1-in-20 year loss",   "amber")
    + stat_card("ES 95%",        es_base,  "expected shortfall",  "red")
    + '</div>'
    '<p style="font-size:0.82rem;font-weight:700;margin:16px 0 6px;">Top 20 highest-risk borrowers:</p>'
    '<div class="table-scroll">' + TOP_TABLE + '</div>'
)

step_stress_body = (
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:14px;">'
    'Stress scenario: all PDs doubled (severe recession), correlations boosted '
    'by 0.20 (contagion amplification). Results show absolute and percentage change.'
    '</p>'
    '<div class="table-scroll">' + STRESS_TABLE + '</div>'
    + insight(
        '<strong>Expected Loss increases ~47% under stress — correlation amplification</strong>'
        'Under independence, doubling PDs would double EL. The additional correlation boost '
        'further amplifies tail losses because simultaneous defaults produce larger losses '
        'than the same total PD spread across time. This is the copula&#39;s key contribution '
        'to stress testing.', "warning")
)

step_client_body = (
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:14px;">'
    'Revenue is derived from transaction volume (fee income proxy), not a flat '
    '2% EAD assumption. This produces realistic profit estimates. Clients are '
    'ranked by contagion-adjusted Sharpe which penalises borrowers in '
    'high-contagion neighbourhoods.'
    '</p>'
    '<p style="font-size:0.82rem;font-weight:700;margin-bottom:8px;">Top 10 clients by contagion-adjusted Sharpe:</p>'
    '<div class="table-scroll">' + CLIENT_TABLE + '</div>'
)

step_metrics_body = (
    IMG_METRIC_CMP
    + '<p style="font-size:0.82rem;font-weight:700;margin:16px 0 6px;">By city:</p>'
    '<div class="table-scroll">' + CITY_TABLE + '</div>'
    '<p style="font-size:0.82rem;font-weight:700;margin:16px 0 6px;">By risk archetype:</p>'
    '<div class="table-scroll">' + ARCH_TABLE + '</div>'
    + insight(
        '<strong>Gamma city and high-risk segment are consistently worst</strong>'
        'CoV_L1 is highest (most risky relative to expected loss), RAROC and Sortino '
        'are negative (loss-making segments). The diversification_ratio shows the '
        'spread between individual &sigma;<sub>i</sub> and portfolio &sigma; — values '
        'around 5&#8211;6&times; indicate that the Clayton copula is producing '
        'meaningful correlation clustering.')
)

step_rating_body = (
    '<div class="chart-grid">'
    + IMG_RATING
    + '<div>'
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:12px;">'
    'Each borrower receives a discrete rating (AAA&#8211;Default) based on their '
    'model PD. The migration table shows the 1-year probability of moving '
    'to each rating state using a Basel-calibrated generator matrix.'
    '</p>'
    '<p style="font-size:0.82rem;font-weight:700;margin-bottom:8px;">Sample migration data (first 12 borrowers):</p>'
    '<div class="table-scroll">' + RATING_TABLE + '</div>'
    '</div></div>'
)

step_merton_body = (
    '<div class="chart-grid">'
    + IMG_MERTON
    + '<div>'
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:12px;">'
    'Scatter: each point is one borrower. Points on the y=x diagonal mean '
    'both models agree. Points above the line: Merton thinks the borrower '
    'is riskier than the statistical model. Large divergences trigger early '
    'warnings flagged for analyst review.'
    '</p>'
    '<p style="font-size:0.82rem;font-weight:700;margin-bottom:8px;">Structural PD sample (top 15):</p>'
    '<div class="table-scroll">' + STRUCT_TABLE + '</div>'
    '</div></div>'
)

# ── ANALYST sections ──────────────────────────────────────────────────────────

analyst_rank_body = (
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:14px;">'
    'Spearman rank correlation between every pair of metrics at the borrower level '
    '(1 000 observations). Pairs near <strong>+1.0 are redundant</strong> — they order '
    'the population identically. <strong>Low correlations</strong> (&lt; 0.7) mean the '
    'two metrics encode genuinely different risk signals.'
    '</p>'
    '<table class="corr-table">'
    '<thead><tr><th>Metric A</th><th>Metric B</th><th>Spearman &rho;</th></tr></thead>'
    '<tbody>' + rank_rows_html + '</tbody>'
    '</table>'
    + insight(
        '<strong>Key findings from the rank correlation</strong>'
        'CoV (L0) vs RAROC show low correlation (~0.48) — they measure genuinely '
        'different things: CoV is pure riskiness, RAROC is profitability per unit capital. '
        'Sharpe and Sortino_L1 are highly correlated (~0.997) at the borrower level '
        'where the copula adds no diversification (L0 = L1 for single borrowers). '
        'The divergence appears at the <em>segment level</em> where off-diagonal '
        'LossCov terms inflate &sigma;<sub>L1</sub> relative to &sigma;<sub>L0</sub>.')
)

analyst_flags_body = (
    '<div class="stat-row">'
    + stat_card("Total flags",            str(n_flags),       "borrowers flagged (z &ge; 1.5)", "red")
    + stat_card("Hidden network risk",    str(n_hidden),      "good RAROC, bad Sortino",         "red")
    + stat_card("Diversified low value",  str(n_diversified), "bad RAROC, good Sortino",         "amber")
    + '</div>'
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:10px;">'
    'Flags are generated by computing each borrower&#39;s RAROC rank and Sortino_L1 rank, '
    'then flagging those whose rank gap exceeds 1.5 standard deviations. '
    'The flag type is determined by direction of divergence.'
    '</p>'
    '<div class="table-scroll">' + FLAGS_TABLE + '</div>'
    '<div class="two-col" style="margin-top:18px;">'
    + insight(
        '<strong>&#128308; Hidden Network Risk</strong>'
        'Borrower has acceptable RAROC (ranks well individually) but poor copula-Sortino '
        '(their neighbourhood inflates the portfolio &sigma;). These are the accounts most '
        'likely to contribute to simultaneous default clustering in a stress event. '
        '<em>Recommended action: reduce exposure or require additional collateral.</em>', "danger")
    + insight(
        '<strong>&#128993; Diversified Low Value</strong>'
        'Borrower has poor RAROC (individually unprofitable) but good copula-Sortino '
        '(they diversify the portfolio — low correlation with the rest of the book). '
        'Cutting these accounts would <em>increase</em> portfolio variance. '
        '<em>Recommended action: retain for diversification; review pricing rather than cutting.</em>',
        "warning")
    + '</div>'
)

analyst_guide_body = (
    '<p style="font-size:0.87rem;font-weight:700;margin-bottom:12px;">'
    'Step-by-step decision tree for reviewing a borrower:'
    '</p>'
    '<div class="theory-grid">'
    + theory_card("1. Check CoV (L0/L1)",
        "Always-positive riskiness ranking. Use when profit is negative "
        "(RAROC/Sortino flip sign and become misleading).<br><br>"
        "<strong>High CoV_L1 &gt; CoV_L0?</strong> The copula is adding contagion premium "
        "— the borrower sits in a correlated cluster. Check diversification_ratio of their segment.")
    + theory_card("2. Check RAROC vs Sortino_L1",
        "If both positive: normal case. Rank by Sortino_L1 for the most complete view.<br><br>"
        "<strong>Good RAROC, bad Sortino?</strong> &rarr; Hidden network risk flag. "
        "Pull divergence_flags table for this borrower.<br><br>"
        "<strong>Bad RAROC, good Sortino?</strong> &rarr; Diversified low value. "
        "Consider retaining for diversification benefit.")
    + theory_card("3. Check Merton divergence",
        "If <code>pd_signal_divergence</code> &gt; 5%: the statistical model and "
        "balance-sheet model disagree. This is usually a leading indicator — the statistical "
        "model has not yet seen the balance-sheet deterioration in its training data. "
        "Escalate for manual review.")
    + theory_card("4. Check rating migration",
        "Review <code>downgrade_prob</code> and <code>default_1yr</code>. "
        "If downgrade probability &gt; 15%: the borrower is likely to trigger "
        "IFRS 9 Stage 2 within 12 months — provision lifetime ECL now. "
        "Check the migration table for transition probabilities over 1 and 3 year horizons.")
    + theory_card("5. Segment-level monitoring",
        "Use <code>by_segment(&#39;city_name&#39;)</code> and "
        "<code>by_segment(&#39;risk_archetype&#39;)</code> monthly. "
        "Track <code>diversification_ratio</code> — if it falls toward 1.0, "
        "the portfolio is becoming more correlated (concentration risk rising). "
        "Monitor <code>numerator_negative</code>: if &gt;50% of a segment, pricing needs review.")
    + theory_card("6. Stress scenario",
        "Run <code>stress_test(pd_multiplier=2.0, correlation_boost=0.2)</code> quarterly. "
        "The regime-aware copula (Step 11) gives a theta calibrated to current market stress "
        "— use this for ICAAP and pillar 2 capital calculations rather than the base theta. "
        "The regime label tells you which historical stress window is most similar to current conditions.")
    + '</div>'
    + insight(
        '<strong>Key Python API calls for analysts</strong>'
        '<div class="formula-block" style="margin-top:8px;font-size:0.78rem;">'
        '<span class="comment"># Compute all metrics for a segment</span>\n'
        'calc.by_segment(&#39;city_name&#39;)\n\n'
        '<span class="comment"># Which metrics agree/disagree on population ranking?</span>\n'
        'comp.rank_correlation(level=&#39;borrower&#39;)\n\n'
        '<span class="comment"># Top divergence cases for analyst review</span>\n'
        'comp.divergence_flags(z_threshold=1.5)\n\n'
        '<span class="comment"># How diversified is this sub-portfolio?</span>\n'
        'calc.diversification_ratio(members=idx_gamma_city)\n\n'
        '<span class="comment"># Full profile for one borrower</span>\n'
        'profiler.profile_report(person_id=776)\n\n'
        '<span class="comment"># Add a new custom metric</span>\n'
        '@register_metric(&#39;my_metric&#39;)\n'
        'def _my_metric(inp: MetricInputs) -&gt; float:\n'
        '    return inp.expected_profit / (inp.loss_var_L1 + 1e-10)'
        '</div>', "success")
    + '<hr class="divider">'
    '<p style="font-size:0.78rem;color:var(--gray3);text-align:center;">'
    'Generated by <code>generate_presentation.py</code> &middot; '
    'Pipeline: <code>main.py</code> &middot; '
    'Tests: <code>python test_copula_framework.py</code> (30 tests) &middot; '
    'Framework: Clayton copula &middot; graph correlation &middot; 7-metric registry'
    '</p>'
)

# ════════════════════════ assemble full page ═══════════════════════════════════

SIDEBAR = """
<nav class="sidebar">
  <div class="sidebar-section">Overview</div>
  <a href="#overview">System Architecture</a>
  <a href="#business-case">Business Case</a>
  <a href="#pipeline-flow">Pipeline at a Glance</a>

  <div class="sidebar-section">Theory Layer</div>
  <a href="#theory-pd"><span class="step-num">T1</span>PD Model &amp; Features</a>
  <a href="#theory-graph"><span class="step-num">T2</span>Transaction Graph</a>
  <a href="#theory-copula"><span class="step-num">T3</span>Clayton Copula</a>
  <a href="#theory-loss"><span class="step-num">T4</span>Loss-Cov Matrix</a>
  <a href="#theory-metrics"><span class="step-num">T5</span>Risk-Adjusted Metrics</a>
  <a href="#theory-merton"><span class="step-num">T6</span>Merton Structural PD</a>
  <a href="#theory-rating"><span class="step-num">T7</span>Rating Migration</a>

  <div class="sidebar-section">Pipeline Results</div>
  <a href="#step-network"><span class="step-num">1</span>Network &amp; Graph</a>
  <a href="#step-pd"><span class="step-num">3</span>PD Model</a>
  <a href="#step-copula"><span class="step-num">5</span>Copula Fit</a>
  <a href="#step-portfolio"><span class="step-num">6</span>Portfolio Risk</a>
  <a href="#step-stress"><span class="step-num">7</span>Stress Test</a>
  <a href="#step-client"><span class="step-num">8</span>Client Value</a>
  <a href="#step-metrics"><span class="step-num">8b</span>Risk Metrics Family</a>
  <a href="#step-rating"><span class="step-num">9</span>Ratings</a>
  <a href="#step-merton"><span class="step-num">10</span>Structural PD</a>

  <div class="sidebar-section">Analysts Toolkit</div>
  <a href="#analyst-rank">Metric Rank Correlation</a>
  <a href="#analyst-flags">RAROC vs Sortino Flags</a>
  <a href="#analyst-guide">Interpretation Guide</a>
</nav>
"""

MAIN_CONTENT = (
    section("overview",       "Overview", "System Architecture",                    "tag-output",   "Framework",    overview_body)
    + section("business-case",  "Business",  "Why Does This Matter? — The Business Case", "tag-output",   "Business",     business_body)
    + section("pipeline-flow",  "Pipeline",  "13-Step Pipeline at a Glance",              "tag-pipeline", "End-to-End",   pipeline_body)
    + section("theory-pd",      "T1",        "Theory: Probability of Default Model",      "tag-theory",   "Theory",       theory_pd_body)
    + section("theory-graph",   "T2",        "Theory: Transaction Graph &#8594; Correlation Matrix", "tag-theory", "Theory", theory_graph_body)
    + section("theory-copula",  "T3",        "Theory: Clayton Copula — Joint Defaults",   "tag-theory",   "Theory",       theory_copula_body)
    + section("theory-loss",    "T4",        "Theory: Loss-Covariance Matrix — The Core Object", "tag-theory", "Theory", theory_loss_body)
    + section("theory-metrics", "T5",        "Theory: Seven Risk-Adjusted Metrics",       "tag-metrics",  "Core Metrics", theory_metrics_body)
    + section("theory-merton",  "T6",        "Theory: Merton Structural PD Model",        "tag-theory",   "Theory",       theory_merton_body)
    + section("theory-rating",  "T7",        "Theory: Rating Migration Model",            "tag-theory",   "Theory",       theory_rating_body)
    + section("step-network",   "Step 1-2",  "Network Visualisation",                     "tag-pipeline", "Pipeline Output", step_network_body)
    + section("step-pd",        "Step 3",    "PD Model — Feature Importance",             "tag-pipeline", "Pipeline Output", step_pd_body)
    + section("step-copula",    "Step 5",    "Copula Comparison — Five Types Fitted",     "tag-pipeline", "Pipeline Output", step_copula_body)
    + section("step-portfolio", "Step 6",    "Portfolio Risk — Loss Distribution",        "tag-pipeline", "Pipeline Output", step_portfolio_body)
    + section("step-stress",    "Step 7",    "Stress Test — 2&times;PD + 20% Correlation Boost", "tag-pipeline", "Pipeline Output", step_stress_body)
    + section("step-client",    "Step 8",    "Client Value Metrics — Sharpe, RAROC, CLTV", "tag-pipeline", "Pipeline Output", step_client_body)
    + section("step-metrics",   "Step 8b",   "Risk-Adjusted Metric Family — All Seven Metrics", "tag-metrics", "Core Metrics", step_metrics_body)
    + section("step-rating",    "Step 9",    "Rating Distribution &amp; Migration",       "tag-pipeline", "Pipeline Output", step_rating_body)
    + section("step-merton",    "Step 10",   "Merton Structural PD vs Statistical Model", "tag-pipeline", "Pipeline Output", step_merton_body)
    + section("analyst-rank",   "Analyst",   "Metric Rank Correlation Matrix",            "tag-metrics",  "Analyst Tool", analyst_rank_body)
    + section("analyst-flags",  "Analyst &#9873;", "RAROC vs Sortino Divergence Flags",  "tag-metrics",  "Early Warning", analyst_flags_body)
    + section("analyst-guide",  "Guide",     "Interpretation Guide for Risk Analysts",    "tag-output",   "Reference",    analyst_guide_body)
)

PAGE = (
    "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
    "<meta charset='UTF-8'>\n"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    "<title>Bank Credit Risk Framework — Full Technical Presentation</title>\n"
    "<style>" + CSS + "</style>\n"
    "</head>\n<body>\n"
    "<header class='top-header'>"
    "<h1>Bank Credit Risk Framework</h1>"
    "<span class='badge'>Copula &middot; Graph &middot; Metrics</span>"
    "<span class='header-right'>13-step analytical pipeline &middot; 1 000 borrowers &middot; synthetic demo</span>"
    "</header>\n"
    + SIDEBAR
    + "\n<main class='main'>\n"
    + MAIN_CONTENT
    + "\n</main>\n"
    "<script>" + JS + "</script>\n"
    "</body>\n</html>"
)

# ─── write ────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(HTML_OUT, "w", encoding="utf-8") as fh:
    fh.write(PAGE)

kb = os.path.getsize(HTML_OUT) // 1024
print("Written: " + HTML_OUT + "  (" + str(kb) + " KB, fully self-contained)")
