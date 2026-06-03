#!/usr/bin/env python3
"""
generate_presentation_ru.py
============================
Reads all pipeline outputs from output/ and writes a single self-contained
output/presentation_ru.html — Russian-language version.

Run:  python generate_presentation_ru.py
"""

import base64
import os

import pandas as pd

OUTPUT_DIR = "output"
HTML_OUT = os.path.join(OUTPUT_DIR, "presentation_ru.html")


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
        return "<p class='muted'>Данные отсутствуют.</p>"
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
        return "<div class='img-missing'>График недоступен.</div>"
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
    "coefficient_of_variation":        "КоВ (L0)",
    "coefficient_of_variation_copula": "КоВ-Копула (L1)",
    "raroc":          "RAROC",
    "sharpe_indep":   "Шарп",
    "sortino_copula": "Сортино-Копула (L1)",
    "sortino_indep":  "Сортино (L0)",
}

rank_rows_html = ""
if not rank_df.empty:
    try:
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

# ─── HTML table blocks ────────────────────────────────────────────────────────

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
TOP_TABLE    = df_to_html(top_df, fmt={"marginal_pd": pct, "composite_risk_score": f4})
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

IMG_NETWORK    = img_tag(imgs["network"],    "Сеть транзакций",          "")
IMG_FEATURES   = img_tag(imgs["features"],   "Важность признаков",       "")
IMG_COPULAS    = img_tag(imgs["copulas"],    "Сравнение копул",          "")
IMG_LOSS_DIST  = img_tag(imgs["loss_dist"],  "Распределение потерь",     "")
IMG_HEATMAP    = img_tag(imgs["heatmap"],    "Тепловая карта рисков",    "")
IMG_RATING     = img_tag(imgs["rating"],     "Рейтинговое распределение","")
IMG_MERTON     = img_tag(imgs["merton"],     "Мертон vs статистич. PD",  "")
IMG_METRIC_CMP = img_tag(imgs["metric_cmp"], "Сравнение метрик",         "")

# ─── CSS ─────────────────────────────────────────────────────────────────────

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --navy:   #0f172a;
  --blue:   #1e40af;
  --blue2:  #3b82f6;
  --blue3:  #93c5fd;
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
  --sidebar-w: 280px;
  --header-h:  60px;
  --shadow: 0 2px 12px rgba(0,0,0,0.10);
}
html { scroll-behavior: smooth; }
body {
  font-family: 'Segoe UI', 'Arial', system-ui, sans-serif;
  background: var(--gray1);
  color: var(--text);
  display: flex; flex-direction: column; min-height: 100vh;
}
.top-header {
  position: fixed; top: 0; left: 0; right: 0; height: var(--header-h);
  background: var(--navy);
  display: flex; align-items: center; padding: 0 24px;
  z-index: 200; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.top-header h1 { color: var(--white); font-size: 1rem; font-weight: 600; letter-spacing: 0.01em; }
.top-header .badge {
  margin-left: 16px; background: var(--blue); color: var(--white);
  font-size: 0.7rem; padding: 2px 10px; border-radius: 12px;
}
.header-right { margin-left: auto; color: var(--gray3); font-size: 0.8rem; }
.sidebar {
  position: fixed; top: var(--header-h); left: 0; bottom: 0;
  width: var(--sidebar-w); background: var(--navy);
  overflow-y: auto; z-index: 100; padding-bottom: 40px;
}
.sidebar-section {
  padding: 20px 16px 8px;
  color: var(--gray3); font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
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
  margin-left: var(--sidebar-w); margin-top: var(--header-h);
  padding: 36px 40px 80px; max-width: 1100px;
}
.section {
  background: var(--white); border-radius: 12px;
  box-shadow: var(--shadow); margin-bottom: 36px; overflow: hidden;
}
.section-header {
  padding: 20px 28px 16px; border-bottom: 2px solid var(--gray2);
  display: flex; align-items: center; gap: 14px;
}
.section-num {
  background: var(--blue); color: var(--white);
  font-size: 0.75rem; font-weight: 700;
  padding: 4px 10px; border-radius: 6px; white-space: nowrap;
}
.section-header h2 { font-size: 1.1rem; font-weight: 700; color: var(--navy); }
.section-header .tag {
  margin-left: auto; font-size: 0.7rem; padding: 3px 10px;
  border-radius: 10px; font-weight: 600; letter-spacing: 0.03em;
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
.theory-card p  { font-size: 0.83rem; line-height: 1.65; color: var(--gray4); }
.formula-block {
  background: var(--navy); color: #e2e8f0;
  border-radius: 8px; padding: 16px 20px; margin: 14px 0;
  font-family: 'Courier New', monospace; font-size: 0.82rem; line-height: 1.8;
  overflow-x: auto;
}
.formula-block .comment   { color: #64748b; }
.formula-block .highlight { color: #93c5fd; font-weight: 700; }
.insight {
  border-left: 4px solid var(--blue2);
  background: #eff6ff; padding: 14px 18px; border-radius: 0 8px 8px 0;
  margin: 14px 0; font-size: 0.84rem; line-height: 1.65;
}
.insight.warning { border-left-color: var(--amber); background: #fffbeb; }
.insight.success { border-left-color: var(--green);  background: #f0fdf4; }
.insight.danger  { border-left-color: var(--red);    background: #fef2f2; }
.insight strong  { display: block; margin-bottom: 4px; font-size: 0.82rem; }
.pipeline-flow { display: flex; flex-wrap: wrap; gap: 0; margin: 20px 0; align-items: stretch; }
.pipe-step {
  flex: 1; min-width: 100px;
  background: var(--gray1); border: 1px solid var(--gray2);
  padding: 12px 8px; text-align: center; position: relative; font-size: 0.72rem;
}
.pipe-step:not(:last-child)::after {
  content: '\\25B6'; position: absolute; right: -11px; top: 50%;
  transform: translateY(-50%); color: var(--gray3); font-size: 0.9rem; z-index: 1;
}
.pipe-step .ps-num {
  display: inline-block; background: var(--blue); color: white;
  border-radius: 50%; width: 22px; height: 22px; line-height: 22px;
  font-size: 0.65rem; font-weight: 700; margin-bottom: 6px;
}
.pipe-step .ps-name { font-weight: 600; color: var(--navy); line-height: 1.3; }
.pipe-step .ps-out  { color: var(--gray3); font-size: 0.66rem; margin-top: 4px; }
.pipe-step.active  { background: #eff6ff; border-color: var(--blue2); }
.pipe-step.metrics { background: #fffbeb; border-color: var(--amber); }
.stat-row  { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
.stat-card {
  flex: 1; min-width: 130px;
  background: var(--gray1); border: 1px solid var(--gray2);
  border-radius: 8px; padding: 16px 18px;
}
.stat-card .sc-label { font-size: 0.72rem; color: var(--gray3); text-transform: uppercase; letter-spacing: 0.06em; }
.stat-card .sc-value { font-size: 1.6rem; font-weight: 700; color: var(--navy); margin-top: 4px; }
.stat-card .sc-sub   { font-size: 0.72rem; color: var(--gray4); margin-top: 2px; }
.stat-card.green  { border-left: 4px solid var(--green);  }
.stat-card.amber  { border-left: 4px solid var(--amber);  }
.stat-card.red    { border-left: 4px solid var(--red);    }
.stat-card.blue   { border-left: 4px solid var(--blue2);  }
.stat-card.purple { border-left: 4px solid var(--purple); }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin: 12px 0; }
.data-table thead tr  { background: var(--navy); color: var(--white); }
.data-table thead th  { padding: 9px 12px; text-align: left; font-weight: 600; font-size: 0.72rem; }
.data-table tbody tr:nth-child(even) { background: #f8fafc; }
.data-table tbody tr:hover { background: #eff6ff; }
.data-table tbody td { padding: 8px 12px; color: var(--gray4); }
.table-scroll { overflow-x: auto; }
.chart-img {
  width: 100%; border-radius: 8px; border: 1px solid var(--gray2);
  display: block; margin: 12px 0;
}
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 14px 0; }
@media (max-width: 800px) { .chart-grid { grid-template-columns: 1fr; } }
.img-missing {
  background: #f1f5f9; border: 1px dashed var(--gray3);
  border-radius: 8px; padding: 24px; text-align: center;
  color: var(--gray3); font-size: 0.8rem;
}
.corr-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.corr-table th { padding: 8px 12px; background: var(--navy); color: var(--white); text-align: left; font-size: 0.72rem; }
.corr-table td { padding: 7px 12px; border-bottom: 1px solid var(--gray2); }
.corr-bar-wrap { display: flex; align-items: center; gap: 10px; }
.corr-bar { height: 12px; border-radius: 3px; min-width: 2px; flex-shrink: 0; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; margin: 2px; }
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
  padding: 16px 18px; border-radius: 0 0 6px 6px; background: var(--white);
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

# ─── builder helpers ──────────────────────────────────────────────────────────

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


def card(title, body):
    return '<div class="theory-card"><h3>' + title + '</h3><p>' + body + '</p></div>'


def insight(text, cls=""):
    return '<div class="insight ' + cls + '">' + text + '</div>'


def stat(label, value, sub, cls=""):
    return (
        '<div class="stat-card ' + cls + '">'
        '<div class="sc-label">' + label + '</div>'
        '<div class="sc-value">' + str(value) + '</div>'
        '<div class="sc-sub">' + sub + '</div>'
        '</div>'
    )


def formula(html):
    return '<div class="formula-block">' + html + '</div>'


def pipe_step(num, name, out, cls=""):
    return (
        '<div class="pipe-step ' + cls + '">'
        '<div class="ps-num">' + num + '</div>'
        '<div class="ps-name">' + name + '</div>'
        '<div class="ps-out">' + out + '</div>'
        '</div>'
    )


# ════════════════════════ section bodies ══════════════════════════════════════

# ── Обзор ────────────────────────────────────────────────────────────────────
overview_body = (
    '<p style="font-size:0.9rem;line-height:1.7;margin-bottom:18px;">'
    'Данный фреймворк представляет собой <strong>полный конвейер анализа кредитных рисков</strong> '
    'для розничного или МСБ-портфеля банка. Система выходит далеко за рамки простой PD-модели: '
    'она моделирует <em>совместные дефолты через сеть транзакций</em>, вычисляет '
    '<em>риск-скорректированную прибыльность</em> на любом уровне агрегации и предоставляет '
    'аналитикам <em>ранние предупреждения</em>, которые невозможно обнаружить с помощью '
    'стандартного кредитного скоринга.'
    '</p>'
    '<div class="theory-grid">'
    + card("&#128279; Корреляции на основе графа транзакций",
        "Транзакции между клиентами отражают их экономическую взаимосвязь. Клиенты, "
        "активно взаимодействующие между собой, чаще дефолтируют одновременно — граф переводов "
        "формирует <em>матрицу корреляций</em>, которая управляет копулой, а не задаётся вручную.")
    + card("&#128208; Копула и совместные дефолты",
        "<strong>Копула Клейтона</strong> объединяет маргинальные PD с матрицей сетевых корреляций "
        "и строит <em>полную матрицу совместных вероятностей дефолта</em> P(D<sub>i</sub>&cap;D<sub>j</sub>) "
        "— основу всех расчётов дисперсии потерь.")
    + card("&#128202; Семейство риск-скорректированных метрик",
        "Семь метрик (КоВ, RAROC, Шарп, варианты Сортино) вычисляются из единого набора "
        "аддитивных примитивов. Любую метрику можно агрегировать по заёмщику, сегменту, "
        "городу или портфелю с <em>математически корректной агрегацией</em> (блочная сумма "
        "матрицы ковариаций потерь).")
    + card("&#9888;&#65039; Расхождение RAROC и Сортино как ранний сигнал",
        "RAROC не учитывает сетевые корреляции (капитал = k&middot;EAD). Знаменатель "
        "копула-Сортино растёт для заёмщиков в корреляционных кластерах. "
        "Расхождение метрик &mdash; ранний сигнал <em>скрытого сетевого риска</em>.")
    + '</div>'
    + insight(
        '<strong>Чем это отличается от стандартного скоринга</strong>'
        'Скоринговая модель ранжирует заёмщиков независимо. Данная система моделирует '
        '<em>совместные</em> дефолты &mdash; она показывает не просто, что заёмщик A рискован, '
        'но что A, B и C с высокой вероятностью дефолтируют <em>одновременно</em>, '
        'что делает портфельные потери значительно выше суммы индивидуальных EL.', "success")
)

# ── Деловой смысл ─────────────────────────────────────────────────────────────
business_body = (
    '<div class="theory-grid">'
    + card("&#128188; Для риск-менеджеров",
        "Стандартные модели VaR предполагают независимость или используют единый рыночный фактор. "
        "Данная система использует <em>реальные транзакционные данные</em> для выявления "
        "<strong>скрытых корреляционных кластеров</strong> &mdash; заёмщиков, которые "
        "индивидуально выглядят приемлемо, но коллективно представляют риск концентрации, "
        "способный нарушить нормативы капитала в период кризиса.")
    + card("&#128200; Для бизнеса и ценообразования",
        "Копула-Сортино даёт <strong>единое безразмерное число</strong> для риск-скорректированной "
        "доходности, аддитивное по любому измерению &mdash; город, сегмент, продукт, "
        "клиентский менеджер. Его можно встраивать в модели ценообразования и дашборды "
        "прибыльности клиентов без искажений при агрегации.")
    + card("&#128269; Для кредитных аналитиков",
        "Система <strong>автоматически выявляет</strong> заёмщиков, по которым RAROC "
        "говорит «прибыльно», а копула-Сортино &mdash; «скрытый сетевой риск». "
        "Это именно те счета, которые старший аналитик должен проверять в первую очередь "
        "&mdash; система находит их без ручного поиска.")
    + card("&#127970; Для стресс-тестирования и ВПОДК",
        "Мертоновская структурная PD добавляет <strong>второй независимый сигнал</strong> "
        "из балансовой волатильности. Движок гибких вероятностей перевзвешивает "
        "исторические сценарии по текущему макро-стрессу, производя "
        "<em>режим-зависимую копулу</em> для ВПОДК и стресс-тестирования по требованиям регулятора.")
    + '</div>'
    + insight(
        '<strong>Ключевой результат: ранние предупреждения, которые RAROC упускает</strong>'
        'В синтетическом портфеле из 1 000 заёмщиков '
        '<strong>' + str(n_flags) + ' заёмщиков</strong> получили флаги расхождения RAROC&ndash;Сортино. '
        'Из них <strong>' + str(n_hidden) + ' &mdash; «скрытый сетевой риск»</strong>: '
        'приемлемый RAROC при высокой концентрации в корреляционном кластере. '
        'Стандартный кредитный анализ эти случаи не обнаружил бы без копульного слоя.')
)

# ── Схема конвейера ───────────────────────────────────────────────────────────
pipeline_body = (
    '<div class="pipeline-flow">'
    + pipe_step("1",  "Генерация данных",     "заёмщики · транзакции")
    + pipe_step("2",  "Граф транзакций",      "матрица корр.", "active")
    + pipe_step("3",  "Модель PD",            "model_pd · AUC")
    + pipe_step("4",  "Матрица корр.",        "ПОП (n&times;n)")
    + pipe_step("5",  "Копула Клейтона",      "P(D<sub>i</sub>&cap;D<sub>j</sub>)", "active")
    + pipe_step("6",  "Анализ рисков",        "VaR · ES · HHI")
    + pipe_step("7",  "Стресс-тест",          "2&times;PD · +20% корр.")
    + pipe_step("8",  "Ценность клиента",     "Шарп · RAROC · CLTV", "active")
    + pipe_step("8b", "Семейство метрик",     "КоВ · Сортино · флаги", "metrics")
    + pipe_step("9",  "Рейтинги",             "AAA&ndash;Дефолт")
    + pipe_step("10", "Мертон PD",            "структурный сигнал", "active")
    + pipe_step("11", "Гибк. вероятности",    "режим-зависимый &theta;")
    + pipe_step("12", "Профили клиентов",     "отчёт по заёмщику")
    + '</div>'
    + '<p class="muted">Синяя рамка = уровень графа/копулы. Жёлтая рамка = новое семейство метрик (ШАГ 8b).</p>'
    + insight(
        '<strong>Поток данных</strong>'
        'Транзакции &#8594; граф &#8594; матрица корреляций &#8594; копула &#8594; '
        'матрица совместных дефолтов &#8594; матрица ковариаций потерь &#8594; '
        'все семь риск-скорректированных метрик. Каждая метрика использует одни и те же '
        'математически согласованные примитивы.')
)

# ── T1: Модель PD ─────────────────────────────────────────────────────────────
theory_pd_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'Модель PD &mdash; это <strong>классификатор на основе градиентного бустинга</strong> (GBM), '
    'обученный на признаках заёмщика для предсказания вероятности дефолта '
    'в течение 12 месяцев. Модель возвращает калиброванный скор из [0, 1] '
    'для каждого заёмщика, который поступает в копулу как маргинальная вероятность.'
    '</p>'
    '<div class="theory-grid">'
    + card("Используемые признаки",
        "Возраст, доход, стаж работы, долговая нагрузка (DTI), количество кредитных линий, "
        "просроченные платежи, утилизация кредита, возраст счёта. Сетевые признаки "
        "(средний и максимальный PD соседей, число высокорисковых соседей) "
        "добавляются после построения графа на шаге 2.")
    + card("Почему GBM, а не логистическая регрессия?",
        "GBM улавливает нелинейные взаимодействия (например, высокая долговая нагрузка опасна "
        "только при сочетании с низким доходом и просрочками) без ручного конструирования признаков. "
        "Калиброванные вероятности обеспечиваются изотонической регрессией.")
    + '</div>'
    + '<details><summary>&#9658; Математическая постановка</summary>'
    '<div class="details-body">'
    + formula(
        '<span class="comment">-- Ансамбль GBM из T деревьев --</span>\n'
        'F<sub>T</sub>(x) = &Sigma;<sub>t=1</sub><sup>T</sup> &eta; &middot; h<sub>t</sub>(x)'
        '   <span class="comment">// &eta; = скорость обучения, h<sub>t</sub> = слабый классификатор</span>\n\n'
        '<span class="comment">-- Калиброванная PD --</span>\n'
        '<span class="highlight">PD<sub>i</sub></span> = &sigma;(F<sub>T</sub>(x<sub>i</sub>))'
        '   <span class="comment">// &sigma; = сигмоида</span>\n'
        '       = 1 / (1 + exp(&minus;F<sub>T</sub>(x<sub>i</sub>)))\n\n'
        '<span class="comment">-- Функция потерь: бинарная кросс-энтропия --</span>\n'
        'L = &minus; &Sigma;<sub>i</sub> [ y<sub>i</sub> log(PD<sub>i</sub>) + '
        '(1&minus;y<sub>i</sub>) log(1&minus;PD<sub>i</sub>) ]\n\n'
        '<span class="comment">-- Метрика качества: AUC-ROC --</span>\n'
        'AUC = P(PD<sub>дефолт</sub> > PD<sub>нет дефолта</sub>)'
        '   <span class="comment">// должно быть >> 0.5</span>'
    )
    + '<p style="font-size:0.82rem;color:var(--gray4);margin-top:10px;">'
    'Validation AUC выше 0.75 означает, что модель хорошо разделяет дефолтных и '
    'недефолтных заёмщиков. В производстве дополнительная калибровка '
    '(изотоническая регрессия) гарантирует, что скор является истинной вероятностью.'
    '</p></div></details>'
)

# ── T2: Граф ──────────────────────────────────────────────────────────────────
theory_graph_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'Клиенты, совершающие транзакции между собой, разделяют экономическую судьбу. '
    'Поставщик, продающий продукцию проблемному покупателю, имеет повышенный риск дефолта '
    'вне зависимости от своего индивидуального скора. Слой графа формализует это: '
    '<em>сеть транзакций определяет структуру корреляций</em> вместо произвольного '
    'рыночного фактора.'
    '</p>'
    + formula(
        '<span class="comment">-- Базовая корреляция по объёму транзакций --</span>\n'
        'общий_объём(i,j) = &Sigma; транзакций, где участвуют оба i и j\n'
        'corr_raw(i,j)    = base_corr + f(общий_объём) &isin; [base_corr, max_corr]\n\n'
        '<span class="comment">-- Надбавки за географию и группу риска --</span>\n'
        'corr(i,j) += city_boost    если city_id_i == city_id_j\n'
        'corr(i,j) += group_boost   если high_risk_group_id_i == high_risk_group_id_j\n\n'
        '<span class="comment">-- Проекция на ближайшую ПОП-матрицу (метод Хайема 2002) --</span>\n'
        '&Sigma; = nearest_psd(corr)'
        '   <span class="comment">// необходимо для корректной симуляции копулы</span>'
    )
    + insight(
        '<strong>Инвариант: матрица должна быть положительно полуопределённой (ПОП)</strong>'
        'Матрица без свойства ПОП приводит к отрицательным собственным значениям, '
        'которые нарушают разложение Холецкого в гауссовской копуле. '
        'Функция <code>_nearest_psd()</code> (Хайем, 2002) выполняет проекцию на конус ПОП-матриц '
        'после каждого изменения матрицы.', "warning")
    + '<div class="theory-grid">'
    + card("Вычисляемые сетевые характеристики",
        "Степень узла, коэффициент кластеризации, посредничество (betweenness), "
        "число компонент связности, мостовые узлы (высокое посредничество), "
        "принадлежность к группам повышенного риска. Используются в модели PD и матрице корреляций.")
    + card("Мостовые узлы &mdash; системная важность",
        "Мостовой узел соединяет иначе изолированные сообщества. При его дефолте "
        "«заражение» может перескочить через границу кластера. Система вычисляет "
        "<em>индекс системной важности</em> = средний прирост PD у других заёмщиков "
        "при дефолте данного.")
    + '</div>'
)

# ── T3: Копула Клейтона ───────────────────────────────────────────────────────
theory_copula_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    '<strong>Копула</strong> &mdash; это функция, объединяющая маргинальные распределения '
    'в совместное, сохраняя каждое из них. Для кредитного риска маргиналы &mdash; это '
    'индивидуальные PD, а копула кодирует структуру зависимости. '
    '<strong>Копула Клейтона</strong> выбрана потому, что обладает '
    '<em>зависимостью в нижнем хвосте</em> &mdash; дефолты кластеризуются в кризис '
    '(левый хвост), что точно соответствует наблюдениям по кредитным кризисам.'
    '</p>'
    + formula(
        '<span class="comment">-- Двумерная копула Клейтона --</span>\n'
        '<span class="highlight">C(u,v ; &theta;)</span> = '
        '(u<sup>&minus;&theta;</sup> + v<sup>&minus;&theta;</sup> &minus; 1)<sup>&minus;1/&theta;</sup>'
        '   &theta; > 0\n\n'
        '<span class="comment">-- Зависимость в нижнем хвосте (кластеризация дефолтов) --</span>\n'
        '&lambda;<sub>L</sub> = 2<sup>&minus;1/&theta;</sup>'
        '   <span class="comment">// &rarr;0 при &theta;&rarr;0 (независимость), &rarr;1 при &theta;&rarr;&infin;</span>\n\n'
        '<span class="comment">-- Симуляция методом гамма-слабости (быстро, векторизовано) --</span>\n'
        'V  ~ Gamma(1/&theta;, 1)\n'
        'U<sub>i</sub> = (1 &minus; log(E<sub>i</sub>)/V)<sup>&minus;1/&theta;</sup>'
        '   где E<sub>i</sub> ~ Exp(1)\n'
        'D<sub>i</sub> = 1  если  U<sub>i</sub> &le; &Phi;<sup>&minus;1</sup>(PD<sub>i</sub>)\n\n'
        '<span class="comment">-- Совместная вероятность дефолта (матрица n&times;n) --</span>\n'
        '<span class="highlight">P(D<sub>i</sub> &cap; D<sub>j</sub>)</span> = '
        'C(PD<sub>i</sub>, PD<sub>j</sub> ; &theta;)'
    )
    + '<details><summary>&#9658; Почему не гауссовская копула?</summary>'
    '<div class="details-body">'
    '<p style="font-size:0.83rem;line-height:1.6;">'
    'Финансовый кризис 2008 года частично объясняется применением гауссовской копулы '
    '(Ли, 2000), у которой <strong>нулевая хвостовая зависимость</strong> &mdash; '
    'она систематически занижала вероятность массовых одновременных дефолтов в кризис. '
    'Параметр &lambda;<sub>L</sub> = 2<sup>&minus;1/&theta;</sup> копулы Клейтона '
    'напрямую задаёт степень кластеризации дефолтов в стрессовых условиях. '
    'При откалиброванном &theta; это уже не предположение модели &mdash; это '
    'калиброванная характеристика данных.'
    '</p></div></details>'
    + insight(
        '<strong>В системе реализованы и сравниваются пять типов копул (шаг 5)</strong>'
        'Гауссовская (нет хвостовой зависимости), Student-t (симметричная хвостовая), '
        'Клейтон (нижняя хвостовая), Гамбель (верхняя хвостовая), Франк (симметричная, умеренная). '
        'Система калибрует все пять и отображает &theta;, нижнюю/верхнюю хвостовую зависимость '
        'и симулированный уровень дефолтности.')
)

# ── T4: Матрица ковариаций потерь ─────────────────────────────────────────────
theory_loss_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'Всё последующее &mdash; все семь метрик, агрегация по сегментам, стресс-тестирование, '
    'коэффициент диверсификации &mdash; строится на <em>одной</em> матрице, '
    'вычисляемой единожды: <strong>матрице ковариаций потерь</strong>. '
    'Она кодирует степень совместного движения потерь заёмщиков i и j.'
    '</p>'
    + formula(
        '<span class="comment">-- Строительные блоки --</span>\n'
        'EL<sub>i</sub>  = EAD<sub>i</sub> &middot; LGD<sub>i</sub> &middot; PD<sub>i</sub>'
        '   <span class="comment">// ожидаемые потери</span>\n'
        'el<sub>i</sub>  = EAD<sub>i</sub> &middot; LGD<sub>i</sub>'
        '             <span class="comment">// вектор взвешенных потерь</span>\n\n'
        '<span class="comment">-- Ковариация дефолтов из копулы --</span>\n'
        'Cov(D<sub>i</sub>, D<sub>j</sub>) = P(D<sub>i</sub>&cap;D<sub>j</sub>) &minus; PD<sub>i</sub>&middot;PD<sub>j</sub>\n'
        'Cov(D<sub>i</sub>, D<sub>i</sub>) = PD<sub>i</sub>&middot;(1&minus;PD<sub>i</sub>)'
        '   <span class="comment">// дисперсия Бернулли на диагонали</span>\n\n'
        '<span class="comment">-- Матрица ковариаций потерь (n&times;n) --</span>\n'
        '<span class="highlight">LossCov[i,j]</span> = el<sub>i</sub> &middot; Cov(D<sub>i</sub>,D<sub>j</sub>) &middot; el<sub>j</sub>\n\n'
        '<span class="comment">-- Дисперсия потерь сегмента (блочная сумма) --</span>\n'
        '<span class="highlight">Var(Loss<sub>S</sub>)</span> = &Sigma;<sub>i&isin;S</sub>&Sigma;<sub>j&isin;S</sub> LossCov[i,j]\n\n'
        '<span class="comment">-- Коэффициент диверсификации (&ge;1 по неравенству треугольника) --</span>\n'
        '<span class="highlight">KD<sub>S</sub></span> = &Sigma;<sub>i</sub>&radic;LossCov[i,i] / &radic;Var(Loss<sub>S</sub>)'
    )
    + insight(
        '<strong>Критично: нельзя усреднять метрики заёмщиков для получения метрики сегмента</strong>'
        'Var(Loss<sub>A</sub> + Loss<sub>B</sub>) = Var<sub>A</sub> + 2&middot;Cov(A,B) + Var<sub>B</sub> '
        '&ne; Var<sub>A</sub> + Var<sub>B</sub> при наличии корреляции. '
        'Усреднение индивидуальных коэффициентов Сортино уничтожает всю информацию о корреляции. '
        'Блочная сумма математически точна и является единственно корректным методом агрегации.', "danger")
    + '<div class="theory-grid">'
    + card("L0 &mdash; предположение о независимости",
        "Использует только диагональ LossCov: &Sigma; diag(block). "
        "Предполагает нулевые внедиагональные ковариации. "
        "Быстро; игнорирует сетевое «заражение». "
        "Используется: КоВ (L0), Шарп (незав.), Сортино (незав.).")
    + card("L1 &mdash; с учётом копулы (полный блок)",
        "Использует полную блочную сумму, включая внедиагональные элементы. "
        "&sigma;<sub>L1</sub> &ge; &sigma;<sub>L0</sub> для положительно коррелированных заёмщиков. "
        "Избыток &sigma;<sub>L1</sub>&minus;&sigma;<sub>L0</sub> = <em>премия за контагиозность</em>. "
        "Используется: КоВ-Копула (L1), Сортино-Копула (L1).")
    + card("L2 &mdash; Монте-Карло симуляция",
        "Использует copula.simulate_defaults(): 10 000 сценариев. "
        "Вычисляет нижнее полуотклонение: &radic;E[max(0,Loss&minus;E[Loss])<sup>2</sup>]. "
        "Учитывает тройные кластеры дефолтов и форму всего хвоста. "
        "Используется: Сортино-Симуляция (L2).")
    + card("Инвариант аддитивности",
        "E[Loss], E[Profit], Capital &mdash; все аддитивны (простые суммы). "
        "Var(Loss<sub>S</sub>) &mdash; блочная сумма. Портфель = сумма сегментов; "
        "любое разбиение даёт согласованные числа. Нет искажений агрегации ни на одном уровне иерархии.")
    + '</div>'
)

# ── T5: Семь метрик ───────────────────────────────────────────────────────────
theory_metrics_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'Все семь метрик вычисляются из одного набора <strong>примитивов MetricInputs</strong> '
    '&mdash; небольшого набора предварительно агрегированных абсолютных значений. '
    'Это означает, что любую метрику можно вычислить на уровне заёмщика, сегмента, '
    'города, группы или портфеля, не меняя формулу.'
    '</p>'
    + formula(
        '<span class="comment">-- Общие примитивы (аддитивны по заёмщикам) --</span>\n'
        'E[Profit]  = &Sigma; (выручка<sub>i</sub> &minus; EAD<sub>i</sub>&middot;LGD<sub>i</sub>&middot;PD<sub>i</sub>)\n'
        'E[Loss]    = &Sigma; EAD<sub>i</sub>&middot;LGD<sub>i</sub>&middot;PD<sub>i</sub>\n'
        'Capital    = &Sigma; capital_ratio &middot; EAD<sub>i</sub>'
        '   <span class="comment">// 8% &times; EAD по умолчанию</span>\n'
        '&sigma;<sub>L0</sub>  = &radic;(&Sigma; diag(LossCov))'
        '   <span class="comment">// без корреляций</span>\n'
        '&sigma;<sub>L1</sub>  = &radic;(block_sum(LossCov))'
        '   <span class="comment">// с учётом копулы</span>\n'
        '&sigma;<sub>L2</sub>  = &radic;E[max(0,Loss&minus;E[Loss])<sup>2</sup>]'
        '   <span class="comment">// симулированное полуотклонение</span>\n\n'
        '<span class="comment">-- Семь метрик --</span>\n'
        '<span class="highlight">КоВ_L0</span>     = &sigma;<sub>L0</sub> / E[Loss]'
        '   <span class="comment">// чистая рискованность, всегда &ge;0</span>\n'
        '<span class="highlight">КоВ_L1</span>     = &sigma;<sub>L1</sub> / E[Loss]'
        '   <span class="comment">// с учётом копулы, растёт для кластеров</span>\n'
        '<span class="highlight">RAROC</span>      = E[Profit] / Capital'
        '   <span class="comment">// не учитывает корреляции</span>\n'
        '<span class="highlight">Шарп_L0</span>   = (E[Profit] &minus; rf&middot;Выручка) / &sigma;<sub>L0</sub>'
        '   <span class="comment">// ориентир: безрисковый доход</span>\n'
        '<span class="highlight">Сортино_L0</span>= (E[Profit] &minus; h&middot;Capital) / &sigma;<sub>L0</sub>'
        '   <span class="comment">// ориентир: ставка барьера</span>\n'
        '<span class="highlight">Сортино_L1</span>= (E[Profit] &minus; h&middot;Capital) / &sigma;<sub>L1</sub>'
        '   <span class="comment">// ОСНОВНАЯ МЕТРИКА. С учётом копулы.</span>\n'
        '<span class="highlight">Сортино_L2</span>= (E[Profit] &minus; h&middot;Capital) / &sigma;<sub>L2</sub>'
        '   <span class="comment">// полный хвост МК</span>'
    )
    + insight(
        '<strong>Числители Шарпа и Сортино намеренно различаются</strong>'
        'Шарп: ориентир &mdash; <em>безрисковый доход на базе выручки</em> (rf&times;Выручка). '
        'Вопрос: «можно ли было вложить эту выручку без риска?» '
        'Сортино: ориентир &mdash; <em>требуемый доход на регуляторный капитал</em> (h&times;Capital). '
        'Вопрос: «достаточен ли доход для оправдания удерживаемого капитала?» '
        'Это разные вопросы, и они должны использовать разные числители.', "warning")
    + '<div class="two-col" style="margin-top:18px;">'
    '<div>'
    '<p style="font-size:0.82rem;font-weight:700;margin-bottom:8px;">Когда использовать каждую метрику:</p>'
    '<table class="data-table"><thead><tr><th>Метрика</th><th>Назначение</th><th>Ограничение</th></tr></thead><tbody>'
    '<tr><td><span class="pill pill-blue">КоВ L0/L1</span></td>'
    '<td>Ранжирование по рискованности (всегда &ge;0)</td><td>Не учитывает прибыльность</td></tr>'
    '<tr><td><span class="pill pill-green">RAROC</span></td>'
    '<td>Простой фильтр прибыльности</td><td>Не видит сетевые корреляции</td></tr>'
    '<tr><td><span class="pill pill-amber">Шарп</span></td>'
    '<td>Риск-скорр. доход на выручку</td><td>Знак меняется при убытке</td></tr>'
    '<tr><td><span class="pill pill-purple">Сортино L1</span></td>'
    '<td>Полный риск-скорр. доход (копула)</td><td>Нужна копула; знак меняется</td></tr>'
    '<tr><td><span class="pill pill-red">Сортино L2</span></td>'
    '<td>Точный хвост, 3+ дефолта</td><td>Медленно (10к МК-путей)</td></tr>'
    '</tbody></table>'
    '</div>'
    '<div>'
    + insight(
        '<strong>Главный ранний сигнал: расхождение RAROC и Сортино_L1</strong>'
        'Знаменатель RAROC = k&middot;EAD (фиксирован). '
        'Знаменатель Сортино_L1 = &radic;(block_sum(LossCov)) '
        '&mdash; растёт при наличии внедиагональных ковариаций. '
        'Хороший RAROC + плохой Сортино_L1 = '
        'индивидуальный кредит приемлем, но сетевое окружение усиливает риск.', "success")
    + insight(
        '<strong>Флаг «диверсифицированная низкая ценность»</strong>'
        'Плохой RAROC + хороший Сортино_L1: заёмщик индивидуально убыточен, '
        'но снижает портфельную дисперсию (низкая корреляция с остальной книгой). '
        'Закрытие этих счетов <em>увеличит</em> дисперсию портфеля.', "warning")
    + '</div></div>'
)

# ── T6: Мертон ────────────────────────────────────────────────────────────────
theory_merton_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'Модель Мертона (1974) рассматривает собственный капитал компании как колл-опцион '
    'на её активы. Дефолт происходит, когда стоимость активов падает ниже уровня долга '
    'к моменту погашения T. Для розничных заёмщиков стоимость активов аппроксимируется '
    'через доход (капитализированный как перпетуитет), а волатильность &mdash; '
    'через долговую нагрузку.'
    '</p>'
    + formula(
        '<span class="comment">-- Прокси-стоимость активов (KMV-метод, розница) --</span>\n'
        'V  &asymp; доход &times; 12 / 0.08'
        '   <span class="comment">// капитализированный перпетуитет</span>\n'
        'D  = доход &times; 3'
        '                   <span class="comment">// долг &asymp; 3 месячных дохода</span>\n'
        '&sigma;<sub>V</sub> &asymp; 0.3 &middot; (DTI / mean_DTI)'
        '   <span class="comment">// прокси волатильности</span>\n\n'
        '<span class="comment">-- Дистанция до дефолта --</span>\n'
        'd2 = [ln(V/D) + (r &minus; &sigma;<sub>V</sub><sup>2</sup>/2)&middot;T] / (&sigma;<sub>V</sub>&middot;&radic;T)\n\n'
        '<span class="comment">-- Структурная PD Мертона --</span>\n'
        '<span class="highlight">PD<sub>мертон</sub></span> = &Phi;(&minus;d2)'
        '   <span class="comment">// функция нормального распределения</span>\n\n'
        '<span class="comment">-- Смешанная PD (статистическая + структурная) --</span>\n'
        'PD<sub>смешан</sub> = &alpha;&middot;PD<sub>модель</sub> + (1&minus;&alpha;)&middot;PD<sub>мертон</sub>'
        '   <span class="comment">// &alpha;=0.35</span>\n\n'
        '<span class="comment">-- Ранний сигнал: расхождение сигналов --</span>\n'
        'расхождение = |PD<sub>мертон</sub> &minus; PD<sub>модель</sub>|'
        '   <span class="comment">// флаг если > 5%</span>'
    )
    + insight(
        '<strong>Зачем второй сигнал?</strong>'
        'Статистическая PD-модель обучена на исторических дефолтах. '
        'Мертоновская структурная модель использует текущую балансовую информацию (доход, долг). '
        'Сильное расхождение &mdash; опережающий индикатор: балансовые показатели ухудшились, '
        'но исторические данные ещё не отразили это в обучающей выборке.')
)

# ── T7: Рейтинговая миграция ──────────────────────────────────────────────────
theory_rating_body = (
    '<p style="font-size:0.85rem;line-height:1.7;margin-bottom:14px;">'
    'Скоры PD разбиваются на дискретные рейтинговые категории (AAA &#8594; Дефолт) '
    'по фиксированным порогам. Марковская <em>матрица-генератор</em> (откалиброванная '
    'по историческим данным Базельского комитета) используется для вычисления вероятности '
    'любого рейтингового перехода через матричную экспоненту.'
    '</p>'
    + formula(
        '<span class="comment">-- Пороги PD (выровнены с методологией Базеля) --</span>\n'
        'AAA: PD &le; 0.001   AA: &le; 0.002   A: &le; 0.005\n'
        'BBB: &le; 0.02       BB: &le; 0.08    B: &le; 0.25\n'
        'CCC: &le; 1.0        Дефолт: PD = 1.0\n\n'
        '<span class="comment">-- Матрица переходов за горизонт &Delta;t --</span>\n'
        '<span class="highlight">P(&Delta;t)</span> = expm(G &middot; &Delta;t)'
        '   <span class="comment">// G = генератор, expm = матричная экспонента</span>\n\n'
        '<span class="comment">-- Ожидаемый дефолт за 3 года --</span>\n'
        'default_3yr &asymp; 1 &minus; (1 &minus; default_1yr)<sup>3</sup>'
    )
    + insight(
        '<strong>Применение: распределение по стадиям МСФО 9</strong>'
        'Стадия 1 (12-месячный ECL): используется default_1yr. '
        'Стадия 2 (пожизненный ECL): триггер &mdash; рейтинговая миграция вниз на одну и более ступеней. '
        'Матрица переходов даёт вероятность такой миграции за оставшийся срок кредита.')
)

# ── Результаты конвейера ──────────────────────────────────────────────────────

step_network_body = (
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:14px;">'
    '1 000 заёмщиков в трёх городах (Alpha, Beta, Gamma). Цвет узла кодирует '
    'базовую PD &mdash; тёмно-красный = высокий риск дефолта. Размер узла кодирует '
    'степень в графе транзакций. Мостовые узлы (высокое посредничество) видны '
    'как соединительные хабы между кластерами.'
    '</p>'
    + IMG_NETWORK
)

step_pd_body = (
    '<div class="chart-grid">'
    + IMG_FEATURES
    + '<div>'
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:12px;">'
    'Модель PD на основе градиентного бустинга обучена на 8 признаках заёмщика. '
    'Признаки ранжированы по их вкладу в ансамбль. Ключевые признаки, как правило: '
    'просроченные платежи, утилизация кредита, долговая нагрузка &mdash; '
    'в соответствии с отраслевыми знаниями.'
    '</p>'
    + insight(
        '<strong>AUC значительно выше 0.5 свидетельствует о реальной предсказательной силе</strong>'
        'AUC 0.8+ означает, что модель правильно ранжирует 80% пар дефолт/нет-дефолта. '
        'Сетевые признаки (средний PD соседей) дают дополнительный прирост '
        'за счёт захвата риска контагиозности.', "success")
    + '</div></div>'
)

step_copula_body = (
    IMG_COPULAS
    + '<p style="font-size:0.83rem;line-height:1.6;margin-top:12px;">'
    'Все пять копул откалиброваны на одном и том же векторе PD + матрице корреляций. '
    '<strong>Клейтон</strong> выбран для продуктивного использования как имеющий '
    'наибольшую нижнюю хвостовую зависимость (&lambda;<sub>L</sub> = 2<sup>&minus;1/&theta;</sup>), '
    'отражая эмпирический факт кластеризации дефолтов в стрессовых условиях.'
    '</p>'
)

step_portfolio_body = (
    '<div class="chart-grid">' + IMG_LOSS_DIST + IMG_HEATMAP + '</div>'
    '<p style="font-size:0.83rem;line-height:1.6;margin-top:12px;">'
    'Слева: 10 000 симулированных портфельных потерь по Монте-Карло. VaR и ES показаны '
    'вертикальными линиями. Хвост тяжелее, чем дал бы нормальный закон, поскольку '
    'копула Клейтона кластеризует потери. '
    'Справа: тепловая карта топ-50 самых рискованных заёмщиков по четырём измерениям '
    '(маргинальная PD, уязвимость к заражению, системная важность, сетевая экспозиция).'
    '</p>'
    '<div class="stat-row" style="margin-top:18px;">'
    + stat("Ожидаемые потери", el_base,  "нормализованный портфель", "blue")
    + stat("VaR 95%",          var_base, "потери 1 раз в 20 лет",    "amber")
    + stat("ES 95%",           es_base,  "ожидаемый дефицит",        "red")
    + '</div>'
    '<p style="font-size:0.82rem;font-weight:700;margin:16px 0 6px;">Топ-20 наиболее рискованных заёмщиков:</p>'
    '<div class="table-scroll">' + TOP_TABLE + '</div>'
)

step_stress_body = (
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:14px;">'
    'Стрессовый сценарий: все PD удвоены (суровая рецессия), '
    'корреляции повышены на 0.20 (усиление контагиозности). '
    'Результаты &mdash; абсолютные значения и процентное изменение.'
    '</p>'
    '<div class="table-scroll">' + STRESS_TABLE + '</div>'
    + insight(
        '<strong>Ожидаемые потери растут ~47% в стрессе &mdash; эффект корреляции</strong>'
        'При независимости удвоение PD удвоило бы EL. Дополнительный рост корреляций '
        'усиливает хвостовые потери, поскольку одновременные дефолты генерируют '
        'большие потери, чем те же суммарные PD, распределённые во времени. '
        'В этом и состоит ключевой вклад копулы в стресс-тестирование.', "warning")
)

step_client_body = (
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:14px;">'
    'Выручка рассчитана по объёму транзакций (прокси комиссионного дохода), '
    'а не из предположения 2% от EAD. Клиенты ранжированы по скорректированному '
    'на контагиозность коэффициенту Шарпа, который штрафует заёмщиков '
    'в высококонтагиозном окружении.'
    '</p>'
    '<p style="font-size:0.82rem;font-weight:700;margin-bottom:8px;">Топ-10 клиентов по скорректированному на заражение коэффициенту Шарпа:</p>'
    '<div class="table-scroll">' + CLIENT_TABLE + '</div>'
)

step_metrics_body = (
    IMG_METRIC_CMP
    + '<p style="font-size:0.82rem;font-weight:700;margin:16px 0 6px;">По городу:</p>'
    '<div class="table-scroll">' + CITY_TABLE + '</div>'
    '<p style="font-size:0.82rem;font-weight:700;margin:16px 0 6px;">По архетипу риска:</p>'
    '<div class="table-scroll">' + ARCH_TABLE + '</div>'
    + insight(
        '<strong>Город Gamma и высокорисковый сегмент стабильно худшие</strong>'
        'КоВ_L1 максимален (высокая рискованность относительно EL); RAROC и Сортино '
        'отрицательны (убыточные сегменты). Коэффициент диверсификации около 5&ndash;6&times; '
        'указывает на значимую корреляционную кластеризацию, формируемую копулой Клейтона.')
)

step_rating_body = (
    '<div class="chart-grid">'
    + IMG_RATING
    + '<div>'
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:12px;">'
    'Каждый заёмщик получает дискретный рейтинг (AAA&ndash;Дефолт) на основе model_pd. '
    'Матрица переходов показывает 1-летнюю вероятность перехода в каждое '
    'рейтинговое состояние с использованием генератора, откалиброванного по методологии Базеля.'
    '</p>'
    '<p style="font-size:0.82rem;font-weight:700;margin-bottom:8px;">Данные миграции (первые 12 заёмщиков):</p>'
    '<div class="table-scroll">' + RATING_TABLE + '</div>'
    '</div></div>'
)

step_merton_body = (
    '<div class="chart-grid">'
    + IMG_MERTON
    + '<div>'
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:12px;">'
    'Диаграмма рассеяния: каждая точка &mdash; один заёмщик. Точки на диагонали y=x '
    'означают согласие обеих моделей. Точки выше линии: Мертон оценивает заёмщика '
    'рискованней, чем статистическая модель. Большое расхождение &mdash; триггер '
    'ранних предупреждений для ручной проверки.'
    '</p>'
    '<p style="font-size:0.82rem;font-weight:700;margin-bottom:8px;">Структурная PD (первые 15 заёмщиков):</p>'
    '<div class="table-scroll">' + STRUCT_TABLE + '</div>'
    '</div></div>'
)

# ── Инструменты аналитика ──────────────────────────────────────────────────────

analyst_rank_body = (
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:14px;">'
    'Ранговая корреляция Спирмена между всеми метриками на уровне заёмщиков '
    '(1 000 наблюдений). Пары около <strong>+1.0 избыточны</strong> &mdash; '
    'они ранжируют популяцию одинаково. <strong>Низкие корреляции</strong> '
    '(&lt;0.7) означают, что две метрики несут разный сигнал о риске.'
    '</p>'
    '<table class="corr-table">'
    '<thead><tr><th>Метрика A</th><th>Метрика B</th><th>Ранг. корр. &rho;</th></tr></thead>'
    '<tbody>' + rank_rows_html + '</tbody>'
    '</table>'
    + insight(
        '<strong>Ключевые выводы из матрицы корреляций</strong>'
        'КоВ (L0) vs RAROC: низкая корреляция (~0.48) &mdash; разные сигналы: '
        'КоВ &mdash; рискованность, RAROC &mdash; прибыльность на капитал. '
        'Шарп vs Сортино_L1: высокая корреляция (~0.997) на уровне заёмщиков, '
        'где копула не добавляет диверсификации (L0=L1 для n=1). '
        'Расхождение L0 и L1 видно на <em>уровне сегментов</em>, где '
        'внедиагональные элементы LossCov раздувают &sigma;<sub>L1</sub>.')
)

analyst_flags_body = (
    '<div class="stat-row">'
    + stat("Всего флагов",               str(n_flags),       "заёмщиков отмечено (z &ge; 1.5)", "red")
    + stat("Скрытый сетевой риск",        str(n_hidden),      "хороший RAROC, плохой Сортино",   "red")
    + stat("Диверсиф. низкая ценность",  str(n_diversified), "плохой RAROC, хороший Сортино",   "amber")
    + '</div>'
    '<p style="font-size:0.84rem;line-height:1.6;margin-bottom:10px;">'
    'Флаги формируются путём вычисления ранга RAROC и ранга Сортино_L1 '
    'для каждого заёмщика и отметки тех, чей разрыв в рангах превышает 1.5 '
    'стандартных отклонения. Тип флага определяется направлением расхождения.'
    '</p>'
    '<div class="table-scroll">' + FLAGS_TABLE + '</div>'
    '<div class="two-col" style="margin-top:18px;">'
    + insight(
        '<strong>&#128308; Скрытый сетевой риск</strong>'
        'Заёмщик имеет приемлемый RAROC (хорошо ранжируется индивидуально), '
        'но плохой копула-Сортино (его окружение раздувает портфельную &sigma;). '
        'Это именно те счета, которые наиболее вероятно приведут к одновременным '
        'дефолтам в стрессовом сценарии. '
        '<em>Рекомендуемое действие: снижение лимита или требование дополнительного залога.</em>', "danger")
    + insight(
        '<strong>&#128993; Диверсифицированная низкая ценность</strong>'
        'Заёмщик имеет плохой RAROC (индивидуально убыточен), но хороший копула-Сортино '
        '(диверсифицирует портфель &mdash; низкая корреляция с остальной книгой). '
        'Закрытие этих счетов <em>увеличит</em> дисперсию портфеля. '
        '<em>Рекомендуемое действие: пересмотр ценообразования, а не закрытие лимита.</em>', "warning")
    + '</div>'
)

analyst_guide_body = (
    '<p style="font-size:0.87rem;font-weight:700;margin-bottom:12px;">'
    'Пошаговое дерево решений при проверке заёмщика:'
    '</p>'
    '<div class="theory-grid">'
    + card("1. Проверить КоВ (L0/L1)",
        "Всегда положительное ранжирование рискованности. Использовать при отрицательной прибыли "
        "(RAROC/Сортино меняют знак и становятся вводящими в заблуждение).<br><br>"
        "<strong>КоВ_L1 &gt; КоВ_L0?</strong> Копула добавляет премию за контагиозность &mdash; "
        "заёмщик находится в коррелированном кластере. Проверить diversification_ratio сегмента.")
    + card("2. Сравнить RAROC и Сортино_L1",
        "Оба положительны &mdash; нормальный случай. Ранжировать по Сортино_L1 для полной картины.<br><br>"
        "<strong>Хороший RAROC, плохой Сортино?</strong> &rarr; Флаг скрытого сетевого риска. "
        "Проверить таблицу divergence_flags.<br><br>"
        "<strong>Плохой RAROC, хороший Сортино?</strong> &rarr; Диверсифицированная низкая ценность. "
        "Рассмотреть удержание ради диверсификации.")
    + card("3. Проверить расхождение Мертона",
        "Если <code>pd_signal_divergence</code> &gt; 5%: статистическая и балансовая модели расходятся. "
        "Обычно это опережающий индикатор &mdash; балансовые показатели ухудшились, "
        "но обучающая выборка ещё не отражает это. Эскалировать на ручную проверку.")
    + card("4. Проверить рейтинговую миграцию",
        "Проверить <code>downgrade_prob</code> и <code>default_1yr</code>. "
        "Если вероятность понижения &gt; 15%: заёмщик, вероятно, перейдёт в МСФО 9 Стадию 2 "
        "в течение 12 месяцев &mdash; начислить пожизненный ECL сейчас. "
        "Проверить матрицу переходов на горизонтах 1 и 3 года.")
    + card("5. Мониторинг на уровне сегментов",
        "Ежемесячно использовать <code>by_segment('city_name')</code> и "
        "<code>by_segment('risk_archetype')</code>. "
        "Отслеживать <code>diversification_ratio</code>: приближение к 1.0 означает рост корреляции "
        "(концентрационный риск растёт). Флаг <code>numerator_negative</code>: "
        "если &gt;50% сегмента убыточны, требуется пересмотр ценообразования.")
    + card("6. Стрессовый сценарий",
        "Ежеквартально: <code>stress_test(pd_multiplier=2.0, correlation_boost=0.2)</code>. "
        "Режим-зависимая копула (шаг 11) даёт &theta;, откалиброванный на текущий рыночный стресс "
        "&mdash; использовать для расчётов ВПОДК и Pillar 2. "
        "Метка режима указывает, какое историческое окно наиболее похоже на текущие условия.")
    + '</div>'
    + insight(
        '<strong>Ключевые вызовы Python API для аналитиков</strong>'
        '<div class="formula-block" style="margin-top:8px;font-size:0.78rem;">'
        '<span class="comment"># Метрики по сегменту</span>\n'
        'calc.by_segment(&#39;city_name&#39;)\n\n'
        '<span class="comment"># Какие метрики согласуются / расходятся?</span>\n'
        'comp.rank_correlation(level=&#39;borrower&#39;)\n\n'
        '<span class="comment"># Топ случаев расхождения для проверки</span>\n'
        'comp.divergence_flags(z_threshold=1.5)\n\n'
        '<span class="comment"># Коэффициент диверсификации субпортфеля</span>\n'
        'calc.diversification_ratio(members=idx_gamma_city)\n\n'
        '<span class="comment"># Полный профиль одного заёмщика</span>\n'
        'profiler.profile_report(person_id=776)\n\n'
        '<span class="comment"># Добавить собственную метрику</span>\n'
        '@register_metric(&#39;моя_метрика&#39;)\n'
        'def _my(inp: MetricInputs) -&gt; float:\n'
        '    return inp.expected_profit / (inp.loss_var_L1 + 1e-10)'
        '</div>', "success")
    + '<hr class="divider">'
    '<p style="font-size:0.78rem;color:var(--gray3);text-align:center;">'
    'Сгенерировано: <code>generate_presentation_ru.py</code> &middot; '
    'Конвейер: <code>main.py</code> &middot; '
    'Тесты: <code>python test_copula_framework.py</code> (30 тестов) &middot; '
    'Фреймворк: копула Клейтона &middot; граф-корреляции &middot; 7 метрик'
    '</p>'
)

# ════════════════════════ sidebar & main ══════════════════════════════════════

SIDEBAR = """
<nav class="sidebar">
  <div class="sidebar-section">Обзор</div>
  <a href="#overview">Архитектура системы</a>
  <a href="#business-case">Деловой смысл</a>
  <a href="#pipeline-flow">Конвейер — общая схема</a>

  <div class="sidebar-section">Теоретический уровень</div>
  <a href="#theory-pd"><span class="step-num">T1</span>Модель PD</a>
  <a href="#theory-graph"><span class="step-num">T2</span>Граф транзакций</a>
  <a href="#theory-copula"><span class="step-num">T3</span>Копула Клейтона</a>
  <a href="#theory-loss"><span class="step-num">T4</span>Матрица ковариаций</a>
  <a href="#theory-metrics"><span class="step-num">T5</span>Семь метрик</a>
  <a href="#theory-merton"><span class="step-num">T6</span>Мертон PD</a>
  <a href="#theory-rating"><span class="step-num">T7</span>Рейтинговая миграция</a>

  <div class="sidebar-section">Результаты конвейера</div>
  <a href="#step-network"><span class="step-num">1</span>Сеть транзакций</a>
  <a href="#step-pd"><span class="step-num">3</span>Модель PD</a>
  <a href="#step-copula"><span class="step-num">5</span>Подбор копулы</a>
  <a href="#step-portfolio"><span class="step-num">6</span>Портфельный риск</a>
  <a href="#step-stress"><span class="step-num">7</span>Стресс-тест</a>
  <a href="#step-client"><span class="step-num">8</span>Ценность клиента</a>
  <a href="#step-metrics"><span class="step-num">8б</span>Семейство метрик</a>
  <a href="#step-rating"><span class="step-num">9</span>Рейтинги</a>
  <a href="#step-merton"><span class="step-num">10</span>Структурная PD</a>

  <div class="sidebar-section">Инструменты аналитика</div>
  <a href="#analyst-rank">Корреляция метрик</a>
  <a href="#analyst-flags">Флаги RAROC vs Сортино</a>
  <a href="#analyst-guide">Руководство по интерпретации</a>
</nav>
"""

MAIN_CONTENT = (
    section("overview",       "Обзор",    "Архитектура системы",                        "tag-output",   "Фреймворк",      overview_body)
    + section("business-case",  "Бизнес",   "Деловой смысл и ценность для банка",         "tag-output",   "Бизнес",         business_body)
    + section("pipeline-flow",  "Конвейер", "13-шаговый конвейер — общая схема",          "tag-pipeline", "Сквозной",       pipeline_body)
    + section("theory-pd",      "T1",       "Теория: модель вероятности дефолта",         "tag-theory",   "Теория",         theory_pd_body)
    + section("theory-graph",   "T2",       "Теория: граф транзакций &#8594; корреляции", "tag-theory",   "Теория",         theory_graph_body)
    + section("theory-copula",  "T3",       "Теория: копула Клейтона — совместные дефолты","tag-theory",  "Теория",         theory_copula_body)
    + section("theory-loss",    "T4",       "Теория: матрица ковариаций потерь",          "tag-theory",   "Теория",         theory_loss_body)
    + section("theory-metrics", "T5",       "Теория: семь риск-скорректированных метрик", "tag-metrics",  "Ключевые метрики",theory_metrics_body)
    + section("theory-merton",  "T6",       "Теория: структурная модель PD Мертона",      "tag-theory",   "Теория",         theory_merton_body)
    + section("theory-rating",  "T7",       "Теория: модель рейтинговой миграции",        "tag-theory",   "Теория",         theory_rating_body)
    + section("step-network",   "Шаг 1-2",  "Визуализация транзакционной сети",           "tag-pipeline", "Результат",      step_network_body)
    + section("step-pd",        "Шаг 3",    "Модель PD — важность признаков",             "tag-pipeline", "Результат",      step_pd_body)
    + section("step-copula",    "Шаг 5",    "Сравнение пяти типов копул",                 "tag-pipeline", "Результат",      step_copula_body)
    + section("step-portfolio", "Шаг 6",    "Портфельный риск — распределение потерь",   "tag-pipeline", "Результат",      step_portfolio_body)
    + section("step-stress",    "Шаг 7",    "Стресс-тест: 2&times;PD + 20% корреляция",  "tag-pipeline", "Результат",      step_stress_body)
    + section("step-client",    "Шаг 8",    "Ценность клиента: Шарп, RAROC, CLTV",       "tag-pipeline", "Результат",      step_client_body)
    + section("step-metrics",   "Шаг 8б",   "Семейство риск-скорр. метрик: все семь",    "tag-metrics",  "Ключевые метрики",step_metrics_body)
    + section("step-rating",    "Шаг 9",    "Рейтинги и миграционный прогноз",            "tag-pipeline", "Результат",      step_rating_body)
    + section("step-merton",    "Шаг 10",   "Структурная PD Мертона vs статистическая",   "tag-pipeline", "Результат",      step_merton_body)
    + section("analyst-rank",   "Аналитик", "Матрица ранговых корреляций метрик",         "tag-metrics",  "Инструмент",     analyst_rank_body)
    + section("analyst-flags",  "Флаги &#9873;","Флаги расхождения RAROC vs Сортино",    "tag-metrics",  "Ранний сигнал",  analyst_flags_body)
    + section("analyst-guide",  "Справка",  "Руководство по интерпретации для аналитиков","tag-output",   "Справочник",     analyst_guide_body)
)

PAGE = (
    "<!DOCTYPE html>\n<html lang='ru'>\n<head>\n"
    "<meta charset='UTF-8'>\n"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    "<title>Система кредитных рисков банка — Полная техническая презентация</title>\n"
    "<style>" + CSS + "</style>\n"
    "</head>\n<body>\n"
    "<header class='top-header'>"
    "<h1>Система кредитных рисков банка</h1>"
    "<span class='badge'>Копула &middot; Граф &middot; Метрики</span>"
    "<span class='header-right'>13-шаговый конвейер &middot; 1 000 заёмщиков &middot; синтетические данные</span>"
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
print("Записано: " + HTML_OUT + "  (" + str(kb) + " КБ, полностью автономный файл)")
