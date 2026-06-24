#!/usr/bin/env python3
"""Generate updated paper figures from leakage-free rerun outputs.

Writes PDF and PNG versions to figures_updated/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("figures_updated")
OUT.mkdir(exist_ok=True)

LEAK = Path("leakage_free_out")

COL = "#3b6fb6"
RED = "#b54a45"
GRN = "#5d9741"
PUR = "#7655a6"
GRY = "#777777"
ORANGE = "#c47a2c"

W1, W2 = 3.45, 7.16

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "figure.dpi": 220,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT / (stem + '.pdf')}")
    print(f"wrote {OUT / (stem + '.png')}")


def load_sessions() -> pd.DataFrame:
    df = pd.read_csv("dr_sessions.csv")
    df = df[df["HvacMode"].astype(str).str.lower().eq("cool")].copy()
    df["Setback_Amplitude_Mean"] = pd.to_numeric(df["Setback_Amplitude_Mean"], errors="coerce")
    df = df[df["Setback_Amplitude_Mean"].between(0, 6)].copy()
    if "N_DR_Rows" in df.columns:
        df = df[pd.to_numeric(df["N_DR_Rows"], errors="coerce").fillna(0) >= 3].copy()
    y = df["OptOut_Immediate"]
    if y.dtype != bool:
        y = y.astype(str).str.lower().isin(["1", "true", "yes", "y"])
    df["Y"] = y.astype(int)
    df["Setback_Bin"] = pd.cut(
        df["Setback_Amplitude_Mean"],
        [0, 1, 2, 3, 4, 6],
        labels=["0-1", "1-2", "2-3", "3-4", "4-6"],
        include_lowest=True,
        right=True,
    ).astype(str)
    df["Cool_Reduction_Frac"] = pd.to_numeric(df["Cool_Reduction_Frac"], errors="coerce")
    df["Delivered_Completion"] = df["Cool_Reduction_Frac"] * (1 - df["Y"])
    return df


def fig2_setback() -> None:
    adj = pd.read_csv(LEAK / "adjusted_setback_no_duration_cdh.csv")
    bins = ["0-1", "1-2", "2-3", "3-4", "4-6"]
    adj = adj.set_index("Setback_Bin").loc[bins].reset_index()

    df = load_sessions()
    delivered = df.groupby("Setback_Bin", observed=True)["Delivered_Completion"].mean().reindex(bins)

    x = np.arange(len(bins))
    fig, ax1 = plt.subplots(figsize=(W1, 2.65))
    ax1.axvspan(1.5, 3.5, color="#000000", alpha=0.05, zorder=0)
    l1 = ax1.plot(x, adj["raw_optout"] * 100, marker="o", color=RED, label="Opt-out, raw")[0]
    l2 = ax1.plot(x, adj["adjusted_optout"] * 100, marker="s", linestyle="--", color=PUR, label="Opt-out, adjusted")[0]
    ax1.set_ylabel("Opt-out rate (%)")
    ax1.set_xlabel("Observed mean setback (degF)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins)
    ax1.set_ylim(0, 36)
    ax1.grid(axis="y", alpha=0.25, lw=0.5)

    ax2 = ax1.twinx()
    l3 = ax2.plot(x, delivered.values, marker="^", color=COL, label="Delivered flex.")[0]
    ax2.set_ylabel("Runtime-fraction flex.")
    ax2.set_ylim(0, 0.22)

    ax1.legend(handles=[l1, l2, l3], loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=7)
    ax1.text(2.0, 4.0, "lowest adjusted\nopt-out", ha="center", va="bottom", fontsize=6.5, color="#444444")
    save(fig, "fig2_setback_updated")


def fig3_reliability() -> None:
    gkf = pd.read_csv(LEAK / "groupkfold_full_reliability_deciles.csv")
    oot = pd.read_csv(LEAK / "out_of_time_full_reliability_deciles.csv")
    gm = pd.read_csv(LEAK / "groupkfold_metrics_leakage_free.csv")
    tm = pd.read_csv(LEAK / "out_of_time_metrics_leakage_free.csv")
    full = gm[gm["name"].eq("M3_full_plus_history")].iloc[0]
    time = tm.iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(W1, 5.15), constrained_layout=True)
    for ax, dat, title, met in [
        (axes[0], gkf, "(a) Held-out customers", full),
        (axes[1], oot, "(b) Future period", time),
    ]:
        ax.plot(dat["decile"], dat["observed_accept"] * 100, marker="o", color=COL, label="Observed")
        ax.plot(dat["decile"], dat["pred_accept"] * 100, marker="s", linestyle="--", color=RED, label="Predicted")
        ax.set_xticks(np.arange(1, 11))
        ax.set_ylim(0, 105)
        ax.set_xlabel("Predicted reliability decile")
        ax.set_ylabel("Acceptance rate (%)")
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.set_title(title)
        ax.text(
            0.03,
            0.95,
            f"AUC={met['auc']:.3f}\nBrier={met['brier']:.3f}\nECE={met['ece_optout']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )
    axes[0].legend(loc="lower right", frameon=False)
    save(fig, "fig3_reliability_updated")


def fig4_same_setback() -> None:
    dat = pd.read_csv(LEAK / "same_3_4_observed_mean_setback_bin.csv")
    order = ["low_risk", "medium_risk", "high_risk"]
    labels = ["Low", "Medium", "High"]
    d = dat.set_index("group").loc[order]
    x = np.arange(len(order))

    ratio = d.loc["low_risk", "completion_delivered"] / d.loc["high_risk", "completion_delivered"]
    ratio_lo = d["delivered_ratio_low_high_ci_lo"].iloc[0]
    ratio_hi = d["delivered_ratio_low_high_ci_hi"].iloc[0]
    red_lo = d["reduction_diff_low_high_ci_lo"].iloc[0]
    red_hi = d["reduction_diff_low_high_ci_hi"].iloc[0]

    fig, ax1 = plt.subplots(figsize=(W1, 2.75))
    bars = ax1.bar(x, d["completion_delivered"], color=[GRN, ORANGE, RED], width=0.58, label="Completion-adjusted")
    ax1.set_ylabel("Runtime-fraction flex.")
    ax1.set_xlabel("Pre-event risk group")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(d["baseline_relative_reduction"].max(), d["completion_delivered"].max()) * 1.35)
    ax1.grid(axis="y", alpha=0.25, lw=0.5)
    ax1.plot(x, d["baseline_relative_reduction"], marker="o", color="#222222", linestyle="--", label="Baseline-relative reduction")
    ax1.legend(loc="upper right", frameon=False)
    save(fig, "fig4_same_setback_updated")


def fig5_accounting() -> None:
    acc = pd.read_csv(LEAK / "accounting_benchmark_leakage_free.csv")
    pred = pd.read_csv(LEAK / "accounting_time_test_predictions.csv")

    method_order = [
        "acceptance_naive_observed_reduction",
        "behavior_aware_accept_only",
        "behavior_aware_full_pre_event",
        "observed_completion",
    ]
    labels = ["Naive\nobserved r", "Accept-\nweighted", "Full\npre-event", "Observed\ncompletion"]
    colors = [RED, COL, GRN, GRY]
    vals = acc.set_index("method").loc[method_order]

    nominal = pred["acceptance_naive_observed_reduction"].astype(float)
    accept_only = pred["behavior_aware_accept_only"].astype(float)
    obs = pred["observed_completion"].astype(float)
    p_acc = (accept_only / nominal.replace(0, np.nan)).clip(0, 1)
    tmp = pd.DataFrame({"p_acc": p_acc, "nominal": nominal, "obs": obs}).dropna()
    tmp["decile"] = pd.qcut(tmp["p_acc"], 10, labels=False, duplicates="drop") + 1
    dec = tmp.groupby("decile").agg(nominal=("nominal", "mean"), obs=("obs", "mean")).reset_index()
    dec["naive_error_pct"] = 100 * (dec["nominal"] - dec["obs"]) / dec["obs"]

    fig, axes = plt.subplots(2, 1, figsize=(W1, 5.2), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(method_order))
    ax.bar(x, vals["mean_per_session"], color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean runtime-fraction flex.")
    ax.set_ylim(0, vals["mean_per_session"].max() * 1.32)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("(a) Accounting estimates")
    for i, (mean, err) in enumerate(zip(vals["mean_per_session"], vals["pct_err_vs_observed_completion"])):
        ax.text(i, mean + 0.004, f"{mean:.3f}\n({err:+.1f}%)", ha="center", va="bottom", fontsize=6.5)

    ax = axes[1]
    ax.bar(dec["decile"], dec["naive_error_pct"], color=RED, width=0.7)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("Predicted acceptance decile")
    ax.set_ylabel("Naive overestimate (%)")
    ax.set_title("(b) Naive error by reliability")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    save(fig, "fig5_accounting_updated")


def main() -> None:
    fig2_setback()
    fig3_reliability()
    fig4_same_setback()
    fig5_accounting()


if __name__ == "__main__":
    main()
