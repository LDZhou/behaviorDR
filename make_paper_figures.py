#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py
=============================================================================
生成 IEEE PES Grid Edge 投稿论文的所有图（矢量 PDF），放到 ./figures/。
- Fig.1 框架图由作者手绘，本脚本【不】生成，论文里用 placeholder。
- Fig.2  setback dose-response（raw/adjusted opt-out + delivered flex）
- Fig.3  reliability 模型验证（OOF decile + calibration）
- Fig.4  同 3-4F setback 的 delivered 异质性（含 bootstrap 95% CI）
- Fig.5  容量核算误差 money figure（方法对比 + 按可靠性十分位的高估）

所有数字均来自已核实的结果表，来源在各处注释标注。
PDF 字体设为 Type-42（TrueType），避免 IEEE PDF eXpress 拒收 Type-3。

用法：
    python make_paper_figures.py
=============================================================================
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- IEEE 友好的全局样式 ----
matplotlib.rcParams.update({
    "pdf.fonttype": 42,      # 关键：嵌入 TrueType，避免 Type-3 被 IEEE 拒收
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
    "figure.dpi": 200,
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

COL = "#4472c4"   # blue
RED = "#c0504d"   # red
GRN = "#70ad47"   # green
GRY = "#9b9b9b"   # grey
PUR = "#8064a2"   # purple

# 单/双栏宽度（英寸）：IEEE 双栏单图 ~3.45in，跨栏 ~7.16in
W1, W2 = 3.45, 7.16


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / name)


# =============================================================================
# Fig.2  Setback dose-response   (来源: 02_adjusted_categorical_optout_fixed.csv,
#         behavior_model_out/01_setback_bin_composition_delivered_flex.csv)
# =============================================================================
def fig2():
    bins = ["0-1", "1-2", "2-3", "3-4", "4-6"]
    x = np.arange(5)
    raw_oo = [31.81, 21.52, 17.33, 20.44, 27.91]      # raw opt-out (%)
    adj_oo = [31.53, 23.26, 18.18, 17.90, 26.56]      # adjusted opt-out (%)
    delivered = [0.0563, 0.0959, 0.1235, 0.1807, 0.1112]  # delivered flex (runtime frac.)

    fig, ax1 = plt.subplots(figsize=(W1, 2.7))
    ax1.axvspan(1.5, 3.5, color="#000000", alpha=0.05, zorder=0)  # broad 2-4F minimum
    l1, = ax1.plot(x, raw_oo, marker="o", color=RED, label="Opt-out, raw")
    l2, = ax1.plot(x, adj_oo, marker="s", color=PUR, ls="--", label="Opt-out, adjusted")
    ax1.set_ylabel("Opt-out rate (%)")
    ax1.set_xlabel("Setback amplitude (\u00b0F)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins)
    ax1.set_ylim(0, 36)

    ax2 = ax1.twinx()
    l3, = ax2.plot(x, delivered, marker="^", color=COL, label="Delivered flex.")
    ax2.set_ylabel("Delivered flex. (runtime frac.)")
    ax2.set_ylim(0, 0.22)

    ax1.legend(handles=[l1, l2, l3], loc="upper center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    ax1.text(2.5, 2.5, "broad 2\u20134\u00b0F\nminimum", ha="center", va="bottom",
             fontsize=6.5, style="italic", color="#444444")
    save(fig, "fig2_setback.pdf")


# =============================================================================
# Fig.3  Reliability model validation   (来源: remaining_experiments_out/
#         groupkfold_reliability_deciles.csv, validation_metrics_summary.csv,
#         out_of_time_metrics.csv)
# =============================================================================
def fig3():
    dec = np.arange(1, 11)
    pred = np.array([21.19, 45.85, 65.63, 78.80, 86.37, 90.86, 93.82, 95.82, 97.15, 98.36])
    obs = np.array([22.70, 45.73, 64.10, 77.32, 84.81, 90.40, 93.42, 96.22, 97.39, 99.17])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(W2, 2.9))

    # (a) decile: predicted vs observed acceptance
    axa.plot(dec, obs, marker="o", color=COL, label="Observed acceptance")
    axa.plot(dec, pred, marker="s", color=RED, ls="--", label="Predicted acceptance")
    axa.set_xlabel("Predicted reliability decile")
    axa.set_ylabel("Acceptance rate (%)")
    axa.set_xticks(dec)
    axa.set_ylim(0, 105)
    axa.grid(axis="y", alpha=0.3, lw=0.5)
    axa.legend(loc="lower right", frameon=False)
    axa.set_title("(a) Reliability ranking", fontsize=8)

    # (b) calibration
    axb.plot([0, 100], [0, 100], ls="--", color=GRY, lw=1.0, label="Perfect")
    axb.scatter(pred, obs, s=22, color=COL, zorder=3, label="Model (OOF)")
    axb.set_xlabel("Predicted acceptance (%)")
    axb.set_ylabel("Observed acceptance (%)")
    axb.set_xlim(0, 105)
    axb.set_ylim(0, 105)
    axb.grid(alpha=0.3, lw=0.5)
    axb.legend(loc="upper left", frameon=False)
    axb.set_title("(b) Calibration", fontsize=8)
    txt = ("GroupKFold OOF\nAUC 0.863  ECE 0.009\nslope 0.99\n"
           "Out-of-time\nAUC 0.870  ECE 0.020")
    axb.text(0.97, 0.03, txt, transform=axb.transAxes, ha="right", va="bottom",
             fontsize=6.3, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.6))
    save(fig, "fig3_reliability.pdf")


# =============================================================================
# Fig.4  Same 3-4F setback, different delivered  (来源:
#         extra_experiments_out/B2_same_3_4_delivered_bootstrap.csv, B2_ratio_ci.csv)
# =============================================================================
def fig4():
    groups = ["Low\nrisk", "Medium", "High\nrisk"]
    x = np.arange(3)
    nominal = [0.2242, 0.2292, 0.2236]
    delivered = [0.2131, 0.1862, 0.1009]
    ci_lo = [0.1883, 0.1669, 0.0825]
    ci_hi = [0.2362, 0.2033, 0.1206]
    err = np.array([[d - lo for d, lo in zip(delivered, ci_lo)],
                    [hi - d for d, hi in zip(delivered, ci_hi)]])

    fig, ax = plt.subplots(figsize=(W1, 2.9))
    ax.bar(x - 0.19, nominal, width=0.38, color=GRY, edgecolor="black", lw=0.6,
           label="Nominal reduction")
    ax.bar(x + 0.19, delivered, width=0.38, yerr=err, capsize=3, color=COL,
           edgecolor="black", lw=0.6, label="Delivered flex. (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Cooling runtime fraction")
    ax.set_ylim(0, 0.34)
    # bracket spanning low<->high delivered bars with the ratio callout
    xl, xh, yb = 0.19, 2.19, 0.255
    ax.plot([xl, xl, xh, xh], [yb - 0.008, yb, yb, yb - 0.008], color="#666666", lw=0.8)
    ax.text(1.19, yb + 0.005, "2.11\u00d7  (95% CI 1.69\u20132.67)", ha="center", va="bottom",
            fontsize=6.8, color="#333333")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#cccccc")
    save(fig, "fig4_same_setback.pdf")


# =============================================================================
# Fig.5  Capacity accounting error  (来源: extra_experiments_out/
#         C_aggregation_error.csv, C_error_by_reliability_decile.csv)
# =============================================================================
def fig5():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(W2, 2.9))

    # (a) method comparison
    methods = ["Naive\n(nominal)", "Behavior\n(accept-\nonly)", "Behavior\n(full)", "Observed\n(truth)"]
    vals = [0.1480, 0.1050, 0.1035, 0.1012]
    errpct = ["+46%", "+3.7%", "+2.2%", "0%"]
    cols = [RED, COL, PUR, GRN]
    bars = axa.bar(range(4), vals, color=cols, edgecolor="black", lw=0.6)
    for i, (b, e) in enumerate(zip(bars, errpct)):
        axa.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, e,
                 ha="center", fontsize=6.8)
    axa.axhline(0.1012, ls="--", color=GRN, lw=0.8, alpha=0.7)
    axa.set_xticks(range(4))
    axa.set_xticklabels(methods)
    axa.set_ylabel("Mean delivered flex. / session")
    axa.set_ylim(0, 0.17)
    axa.set_title("(a) Capacity accounting (out-of-time)", fontsize=8)

    # (b) naive overestimate by reliability decile
    dec = np.arange(1, 11)
    overest = [571.3, 145.2, 62.6, 33.3, 17.1, 8.0, 7.3, 3.7, 1.3, -0.7]
    axb.bar(dec, overest, color=RED, edgecolor="black", lw=0.5)
    axb.axhline(0, color="black", lw=0.6)
    axb.set_xlabel("Predicted reliability decile")
    axb.set_ylabel("Naive overestimate of\ndelivered flex. (%)")
    axb.set_xticks(dec)
    axb.set_ylim(-20, 630)
    axb.text(1, 585, "571%", ha="center", va="bottom", fontsize=6.8, color=RED)
    axb.set_title("(b) Error concentrates in low-reliability users", fontsize=7.5)
    save(fig, "fig5_capacity.pdf")


if __name__ == "__main__":
    fig2()
    fig3()
    fig4()
    fig5()
    print("\nAll figures written to ./figures/  (Fig.1 framework is a manual placeholder).")
