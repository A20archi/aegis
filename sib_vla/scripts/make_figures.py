"""Generate Week-1 SIB figures as a single self-contained HTML dashboard.

Usage:
    python scripts/make_figures.py [--out results/figures]

Produces results/figures/sib_week1.html  (open in any browser)
"""

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── colour palette ─────────────────────────────────────────────────────────────
C_VANILLA  = "#5B8DB8"
C_SIB      = "#E07B39"
C_LOWPASS  = "#6AAB6A"
C_ADAPTIVE = "#9B6BB5"
BG         = "#FAFAFA"
GRID       = "#E8E8E8"

FONT = dict(family="Inter, Arial, sans-serif", size=14, color="#2B2B2B")


def load(path):
    return json.load(open(path))


def jerk_pct(sib, base):
    return round((1 - sib / base) * 100, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Robustness: jerk vs action-noise level (Spatial)
# ══════════════════════════════════════════════════════════════════════════════
def fig_robustness(spatial_dir: Path):
    noise_levels = [0.0, 0.05, 0.10, 0.20]

    def load_series(tag, noise):
        f = spatial_dir / (f"eval_{tag}.json" if noise == 0.0
                           else f"eval_{tag}__action_noise{noise}.json")
        if not f.exists():
            return None, None
        d = load(f)
        return d["rms_jerk_mean"], d["success_rate"]

    tags = {
        "Vanilla (open-loop)": ("vanilla_n25",  C_VANILLA,  "dash"),
        "Low-pass filter":     ("lowpass",       C_LOWPASS,  "dot"),
        "SIB (ours)":          ("sib_b1e-4",     C_SIB,      "solid"),
        "SIB-Adaptive (ours)": ("sib_adaptive",  C_ADAPTIVE, "dashdot"),
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["<b>Motion Smoothness</b>  (lower = smoother)",
                        "<b>Task Success Rate</b>  (higher = better)"],
        horizontal_spacing=0.12,
    )

    for label, (tag, color, dash) in tags.items():
        jerks, srs = [], []
        for nl in noise_levels:
            j, s = load_series(tag, nl)
            jerks.append(j); srs.append(s)

        kw = dict(x=noise_levels, mode="lines+markers",
                  line=dict(color=color, width=2.5, dash=dash),
                  marker=dict(size=8), name=label, legendgroup=label)

        fig.add_trace(go.Scatter(**kw, y=jerks,
                                 hovertemplate="noise=%{x:.2f}<br>jerk=%{y:.3f}<extra>" + label + "</extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(**kw, y=[s * 100 if s is not None else None for s in srs],
                                 showlegend=False,
                                 hovertemplate="noise=%{x:.2f}<br>SR=%{y:.1f}%<extra>" + label + "</extra>"),
                      row=1, col=2)

    fig.add_annotation(x=0.20, y=0.115, xref="x", yref="y",
                       text="SIB stays flat ✓", showarrow=True,
                       arrowhead=2, ax=-60, ay=-30,
                       font=dict(color=C_SIB, size=12), row=1, col=1)
    fig.add_annotation(x=0.20, y=0.630, xref="x", yref="y",
                       text="Vanilla blows up ✗", showarrow=True,
                       arrowhead=2, ax=40, ay=-20,
                       font=dict(color=C_VANILLA, size=12), row=1, col=1)

    fig.update_xaxes(title_text="Action noise σ", tickvals=noise_levels,
                     ticktext=["0 (clean)", "0.05", "0.10", "0.20"],
                     gridcolor=GRID)
    fig.update_yaxes(title_text="RMS Jerk", gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 80], gridcolor=GRID, row=1, col=2)

    fig.update_layout(
        title=dict(text="<b>Fig 1 · Robustness to Action Noise</b> — LIBERO-Spatial",
                   font=dict(size=18)),
        plot_bgcolor=BG, paper_bgcolor="white",
        font=FONT, height=440, legend=dict(orientation="h", y=-0.22),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Animated jerk bar: clean → noisy
# ══════════════════════════════════════════════════════════════════════════════
def fig_jerk_animation(spatial_dir: Path):
    noise_levels = [0.0, 0.05, 0.10, 0.20]
    x_labels = ["Clean", "Low noise (σ=0.05)", "Medium noise (σ=0.10)", "High noise (σ=0.20)"]

    def jerk(tag, nl):
        f = spatial_dir / (f"eval_{tag}.json" if nl == 0.0
                           else f"eval_{tag}__action_noise{nl}.json")
        return load(f)["rms_jerk_mean"] if f.exists() else None

    van_j = [jerk("vanilla_n25", nl) for nl in noise_levels]
    sib_j = [jerk("sib_b1e-4",   nl) for nl in noise_levels]
    low_j = [jerk("lowpass",     nl) for nl in noise_levels]

    frames = []
    for i in range(1, len(noise_levels) + 1):
        frames.append(go.Frame(
            data=[
                go.Bar(name="Vanilla",     x=x_labels[:i], y=van_j[:i],
                       marker_color=C_VANILLA, offsetgroup="van"),
                go.Bar(name="Low-pass",    x=x_labels[:i], y=low_j[:i],
                       marker_color=C_LOWPASS, offsetgroup="low"),
                go.Bar(name="SIB (ours)", x=x_labels[:i], y=sib_j[:i],
                       marker_color=C_SIB, offsetgroup="sib"),
            ],
            name=str(i),
        ))

    fig = go.Figure(
        data=[
            go.Bar(name="Vanilla",     x=x_labels[:1], y=van_j[:1],
                   marker_color=C_VANILLA, offsetgroup="van"),
            go.Bar(name="Low-pass",    x=x_labels[:1], y=low_j[:1],
                   marker_color=C_LOWPASS, offsetgroup="low"),
            go.Bar(name="SIB (ours)", x=x_labels[:1], y=sib_j[:1],
                   marker_color=C_SIB, offsetgroup="sib"),
        ],
        frames=frames,
    )

    fig.update_layout(
        title=dict(
            text="<b>Fig 2 · How Jerk Changes as Noise Increases</b><br>"
                 "<sup>Press ▶ to animate — SIB's bar stays almost the same height</sup>",
            font=dict(size=18)),
        barmode="group",
        yaxis=dict(title="RMS Jerk (lower = smoother)", range=[0, 0.75], gridcolor=GRID),
        xaxis=dict(title="Noise Condition"),
        plot_bgcolor=BG, paper_bgcolor="white",
        font=FONT, height=480,
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(frame=dict(duration=800, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[str(i+1)],
                              dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                        label=x_labels[i]) for i in range(len(noise_levels))],
            x=0.1, len=0.8, y=-0.05,
            currentvalue=dict(prefix="Condition: ", font=dict(size=13)),
        )],
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-task SR gain (Spatial + Long)
# ══════════════════════════════════════════════════════════════════════════════
def fig_per_task(spatial_dir: Path, long_dir: Path):
    def load_per_task(f):
        d = load(f)
        return {t["task_id"]: t["success_rate"] for t in d.get("per_task", [])}

    sv = load_per_task(spatial_dir / "eval_vanilla_n25.json")
    ss = load_per_task(spatial_dir / "eval_sib_b1e-4.json")
    lv = load_per_task(long_dir    / "eval_vanilla_long.json")
    ls = load_per_task(long_dir    / "eval_sib_long.json")

    task_ids = list(range(10))
    s_delta  = [ss.get(i, 0) - sv.get(i, 0) for i in task_ids]
    l_delta  = [ls.get(i, 0) - lv.get(i, 0) for i in task_ids]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["<b>LIBERO-Spatial</b>  SIB − Vanilla per task",
                        "<b>LIBERO-Long</b>  SIB − Vanilla per task"],
        horizontal_spacing=0.12,
    )

    for col, deltas in enumerate([s_delta, l_delta], start=1):
        colors = [C_SIB if d >= 0 else C_VANILLA for d in deltas]
        fig.add_trace(go.Bar(
            x=[f"Task {i}" for i in task_ids],
            y=[d * 100 for d in deltas],
            marker_color=colors,
            text=[f"{d*100:+.0f}pp" for d in deltas],
            textposition="outside",
            hovertemplate="Task %{x}<br>SIB gain: %{y:.1f}pp<extra></extra>",
            showlegend=False,
        ), row=1, col=col)
        fig.add_hline(y=0, line_dash="dash", line_color="#999", row=1, col=col)

    fig.update_yaxes(title_text="SR gain (pp)", range=[-25, 25], gridcolor=GRID)

    fig.update_layout(
        title=dict(
            text="<b>Fig 3 · Per-Task Success Rate Change</b><br>"
                 "<sup>Orange bar = SIB wins that task, Blue = Vanilla wins</sup>",
            font=dict(size=18)),
        plot_bgcolor=BG, paper_bgcolor="white",
        font=FONT, height=440,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — β sweep: dual-axis showing SR and Jerk vs tightness of filter
# ══════════════════════════════════════════════════════════════════════════════
def fig_rd_frontier(spatial_dir: Path):
    # Ordered from most aggressive filter (β=1e-2) to least (vanilla = no filter)
    points = [
        ("β=1e-2\n(too tight)", "sib_b1e-2"),
        ("β=3e-4",              "sib_b3e-4"),
        ("β=1e-4\n(sweet spot)","sib_b1e-4"),
        ("Vanilla\n(no filter)","vanilla_n25"),
    ]

    x_labels, srs, jerks = [], [], []
    for label, tag in points:
        f = spatial_dir / f"eval_{tag}.json"
        if not f.exists():
            continue
        d = load(f)
        x_labels.append(label)
        srs.append(d["success_rate"] * 100)
        jerks.append(d["rms_jerk_mean"])

    best_idx = x_labels.index("β=1e-4\n(sweet spot)")
    van_idx  = x_labels.index("Vanilla\n(no filter)")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "<b>Task Success Rate</b>  — we want this HIGH and stable",
            "<b>Motion Jerk</b>  — we want this LOW",
        ],
        vertical_spacing=0.14,
        row_heights=[0.5, 0.5],
    )

    # SR bars — colour the sweet spot orange, others grey, vanilla blue
    sr_colors = ["#C0C8D0"] * len(x_labels)
    sr_colors[best_idx] = C_SIB
    sr_colors[van_idx]  = C_VANILLA

    fig.add_trace(go.Bar(
        x=x_labels, y=srs,
        marker_color=sr_colors,
        text=[f"{v:.1f}%" for v in srs],
        textposition="outside",
        hovertemplate="%{x}<br>SR: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Vanilla SR reference line
    fig.add_hline(y=srs[van_idx], line_dash="dash", line_color=C_VANILLA,
                  line_width=1.5, row=1, col=1)
    fig.add_annotation(
        x=x_labels[0], y=srs[van_idx] + 1.5,
        text="Vanilla baseline", showarrow=False,
        font=dict(color=C_VANILLA, size=11), xanchor="left", row=1, col=1,
    )

    # Jerk bars
    jerk_colors = ["#C0C8D0"] * len(x_labels)
    jerk_colors[best_idx] = C_SIB
    jerk_colors[van_idx]  = C_VANILLA

    fig.add_trace(go.Bar(
        x=x_labels, y=jerks,
        marker_color=jerk_colors,
        text=[f"{v:.3f}" for v in jerks],
        textposition="outside",
        hovertemplate="%{x}<br>Jerk: %{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    # Vanilla jerk reference line
    fig.add_hline(y=jerks[van_idx], line_dash="dash", line_color=C_VANILLA,
                  line_width=1.5, row=2, col=1)

    # Sweet-spot annotation on jerk panel
    red = jerk_pct(jerks[best_idx], jerks[van_idx])
    fig.add_annotation(
        x=x_labels[best_idx], y=jerks[best_idx] + 0.04,
        text=f"<b>−{red}% jerk vs vanilla</b>",
        showarrow=True, arrowhead=2, ax=0, ay=-35,
        font=dict(color=C_SIB, size=12), row=2, col=1,
    )

    fig.update_yaxes(title_text="SR (%)", gridcolor=GRID, range=[0, 80],  row=1, col=1)
    fig.update_yaxes(title_text="Jerk",   gridcolor=GRID, range=[0, 0.55], row=2, col=1)
    fig.update_xaxes(title_text="Filter tightness  →  (left = strong filter, right = no filter)",
                     row=2, col=1)

    fig.update_layout(
        title=dict(
            text="<b>Fig 4 · Choosing the Filter Strength (β)</b><br>"
                 "<sup>Too tight → SR drops. Too loose → jerk stays high. β=1e-4 is the sweet spot.</sup>",
            font=dict(size=18)),
        plot_bgcolor=BG, paper_bgcolor="white",
        font=FONT, height=560,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Cross-suite summary
# ══════════════════════════════════════════════════════════════════════════════
def fig_cross_suite(spatial_dir: Path, long_dir: Path):
    suites   = ["LIBERO-Spatial", "LIBERO-Long"]
    van_sr   = [load(spatial_dir / "eval_vanilla_n25.json")["success_rate"] * 100,
                load(long_dir    / "eval_vanilla_long.json")["success_rate"] * 100]
    sib_sr   = [load(spatial_dir / "eval_sib_b1e-4.json")["success_rate"] * 100,
                load(long_dir    / "eval_sib_long.json")["success_rate"] * 100]
    van_jerk = [load(spatial_dir / "eval_vanilla_n25.json")["rms_jerk_mean"],
                load(long_dir    / "eval_vanilla_long.json")["rms_jerk_mean"]]
    sib_jerk = [load(spatial_dir / "eval_sib_b1e-4.json")["rms_jerk_mean"],
                load(long_dir    / "eval_sib_long.json")["rms_jerk_mean"]]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["<b>Task Success Rate (%)</b>",
                                        "<b>Motion Jerk (lower = smoother)</b>"],
                        horizontal_spacing=0.14)

    for col, (van, sib, ylab) in enumerate([
        (van_sr,   sib_sr,   "Success Rate (%)"),
        (van_jerk, sib_jerk, "RMS Jerk"),
    ], start=1):
        fig.add_trace(go.Bar(
            name="Vanilla", x=suites, y=van,
            marker_color=C_VANILLA, width=0.3, offsetgroup="van",
            text=[f"{v:.1f}{'%' if col==1 else ''}" for v in van],
            textposition="outside",
            legendgroup="van", showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Bar(
            name="SIB (ours)", x=suites, y=sib,
            marker_color=C_SIB, width=0.3, offsetgroup="sib",
            text=[f"{v:.1f}{'%' if col==1 else ''}" for v in sib],
            textposition="outside",
            legendgroup="sib", showlegend=(col == 1),
        ), row=1, col=col)
        fig.update_yaxes(title_text=ylab, gridcolor=GRID, row=1, col=col)

    # Jerk reduction labels
    for i, suite in enumerate(suites):
        red = jerk_pct(sib_jerk[i], van_jerk[i])
        fig.add_annotation(
            x=suite, y=max(sib_jerk[i], van_jerk[i]) * 1.3,
            text=f"<b>−{red}%</b>",
            showarrow=False, font=dict(color=C_SIB, size=14),
            xref="x2", yref="y2",
        )

    fig.update_layout(
        title=dict(
            text="<b>Fig 5 · SIB vs Vanilla — Both Suites</b><br>"
                 "<sup>Success rate stays the same; jerk drops ~74% in both benchmarks</sup>",
            font=dict(size=18)),
        barmode="group",
        plot_bgcolor=BG, paper_bgcolor="white",
        font=FONT, height=460,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Assemble dashboard HTML
# ══════════════════════════════════════════════════════════════════════════════
def make_dashboard(out_dir: Path, spatial_dir: Path, long_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = [
        ("robustness",     "Fig 1 · Noise Robustness",  fig_robustness(spatial_dir)),
        ("jerk_animation", "Fig 2 · Jerk Animation",    fig_jerk_animation(spatial_dir)),
        ("per_task",       "Fig 3 · Per-Task Breakdown", fig_per_task(spatial_dir, long_dir)),
        ("rd_frontier",    "Fig 4 · RD Frontier",        fig_rd_frontier(spatial_dir)),
        ("cross_suite",    "Fig 5 · Cross-Suite Summary", fig_cross_suite(spatial_dir, long_dir)),
    ]

    # Save individual PNGs
    for name, title, fig in figs:
        fig.write_image(str(out_dir / f"{name}.png"), scale=2, width=960,
                        height=fig.layout.height)
        print(f"  saved {name}.png")

    # Build unified dashboard
    # First figure carries the full plotly.js CDN; rest are cdn-less
    divs = []
    for i, (name, title, fig) in enumerate(figs):
        incl = "cdn" if i == 0 else False
        divs.append((name, title, fig.to_html(full_html=False, include_plotlyjs=incl)))

    nav = "".join(
        f'<a href="#{n}">{t}</a>'
        for n, t, _ in divs
    )
    sections = "".join(
        f'<section id="{n}"><h2>{t}</h2>{div}</section>'
        for n, t, div in divs
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SIB-VLA · Week 1 Results</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body  {{ font-family: Inter, Arial, sans-serif; background: #F0F2F5; color: #2B2B2B; }}
    header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: #FFF; padding: 28px 48px;
    }}
    header h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -0.3px; }}
    header p  {{ margin-top: 6px; font-size: 13px; color: #AAB8C8; line-height: 1.6; }}
    nav {{
      background: #FFF; padding: 0 48px;
      border-bottom: 1px solid #DDE3EC;
      position: sticky; top: 0; z-index: 100;
      display: flex; gap: 0; overflow-x: auto;
    }}
    nav a {{
      display: inline-block; padding: 14px 18px;
      font-size: 13px; font-weight: 500;
      color: #5B8DB8; text-decoration: none;
      border-bottom: 3px solid transparent;
      white-space: nowrap;
    }}
    nav a:hover {{ border-bottom-color: #E07B39; color: #E07B39; }}
    section {{
      max-width: 980px; margin: 36px auto; padding: 0 24px;
      background: #FFF; border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }}
    section h2 {{ padding: 20px 24px 0; font-size: 15px; color: #888; font-weight: 500; }}
    .note {{
      background: #FFF8E7; border-left: 4px solid #E07B39;
      padding: 14px 20px; margin: 16px 48px 0;
      font-size: 13px; border-radius: 4px; color: #555;
    }}
  </style>
</head>
<body>
<header>
  <h1>SIB-VLA · Week 1 Results</h1>
  <p>
    Spectral Information Bottleneck applied to robot action chunk filtering.<br>
    Evaluated on LIBERO-Spatial and LIBERO-Long benchmarks · n=25 open-loop execution · SmolVLA backbone (frozen)
  </p>
</header>
<nav>{nav}</nav>
<div class="note">
  ℹ️  All evals use <strong>n=25</strong> (execute 25 of 50 predicted steps before re-querying — 25× cheaper than n=1).
  Published SmolVLA numbers (90% Spatial, 71% Long) are at n=1. Our absolute SR is lower for this reason, not a model issue.
</div>
{sections}
</body>
</html>"""

    out_path = out_dir / "sib_week1.html"
    out_path.write_text(html)
    print(f"\n  dashboard → {out_path}")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",     default="results/figures")
    ap.add_argument("--spatial", default="results/libero_spatial")
    ap.add_argument("--long",    default="results/long")
    args = ap.parse_args()
    make_dashboard(Path(args.out), Path(args.spatial), Path(args.long))
    print("Done.")


if __name__ == "__main__":
    main()
