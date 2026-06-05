#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extra_experiments_and_pack.py
=============================================================================
为「Behavior-Aware Delivered Flexibility」投稿(IEEE PES Grid Edge)补三组实验,
跑完自动打包。【不改主 pipeline，不碰 CATE】。只读 dr_sessions.csv。

实验：
  B1  特征消融的 history-only / no-history 基线
      —— 回答审稿人"AUC 0.86 是不是只在背用户历史"。
      同一套 GroupKFold 上比 full / history_only / no_history 三个模型。
  B2  headline 2× 的 user-clustered bootstrap CI
      —— 3-4°F 内 low/high risk 的 delivered_flex 点估计 + 95%CI（按用户重采样），
      并报三组 nominal_reduction（证明物理潜力相同）。【model-free，最稳】
  C   聚合误差 money figure（out-of-time 持出）
      —— 按报名数/nominal 估容量 vs 行为感知期望 vs 实际 delivered 的误差对比。
      在【观测 setback】上评估，不改 setback => 不外推、无 positivity 问题。
  D   分州外部效度
      —— 用 full 模型 OOF 预测，按州报 AUC/ECE/opt-out，暴露 VA/CS_DR 集中度。

打包：
  产出写到 extra_experiments_out/(+figs/)，最后压成 extra_experiments_bundle.zip。

用法（在含 dr_sessions.csv 的项目根目录）：
    python extra_experiments_and_pack.py
=============================================================================
"""

import json
import warnings
import zipfile
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    HAS_LGBM = False

DATA = Path("dr_sessions.csv")
OUT = Path("extra_experiments_out")
FIG = OUT / "figs"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_BOOT = 1000
SETBACK_BINS = [0, 1, 2, 3, 4, 6]
SETBACK_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-6"]

LINES = []
def log(s=""):
    print(s, flush=True)
    LINES.append(str(s))

# 与 behavior_model_validation.py 完全一致的候选特征
WEATHER_TIME = ["Tout_onset", "RH_onset", "GHI_onset", "dew_onset", "CDH_during",
                "temperature_2m", "relative_humidity_2m", "shortwave_radiation", "dew_point_2m",
                "Duration_Min", "Hour_of_Day", "Is_Weekend", "Month", "Hour_Bin",
                "DR_Type", "province_state", "country"]
SETBACK_FEATS = ["Setback", "Setback_sq", "Setback_Bin"]
BUILDING_BASELINE = ["floor_area_sqft", "building_age_yrs", "number_occupants", "has_heatpump",
                     "has_electric", "number_cool_stages", "number_heat_stages", "building_type",
                     "Baseline_Cool_Frac", "Avg_Baseline_Temp", "Setpoint_Cool_Start",
                     "Avg_Baseline_Cool", "weather_is_fallback"]
HISTORY = ["Prev_OptOut_Recomputed", "Prior_OptOut_Rate_Recomputed",
           "OptOut_Streak_Recomputed", "Session_Seq_Recomputed"]
CAT_COLS = {"Setback_Bin", "Hour_Bin", "DR_Type", "province_state", "country", "building_type"}


# ---------------------------------------------------------------- data prep
def safe_num(s):
    if s.dtype == bool:
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce")


def prep():
    if not DATA.exists():
        raise FileNotFoundError("dr_sessions.csv 不在当前目录")
    df = pd.read_csv(DATA)
    log("=" * 74)
    log("Extra experiments for behavior-aware delivered flexibility")
    log("=" * 74)
    log(f"raw rows: {len(df):,}")

    if "HvacMode" in df.columns:
        df = df[df["HvacMode"].astype(str).str.lower().eq("cool")].copy()
    if "OptOut_Immediate" in df.columns:
        ycol = "OptOut_Immediate"
    elif "Opted_Out" in df.columns:
        ycol = "Opted_Out"
    else:
        raise ValueError("找不到 opt-out 结局列")
    df["Y"] = (safe_num(df[ycol]).fillna(0) > 0).astype(int)

    df["Setback"] = safe_num(df["Setback_Amplitude_Mean"])
    df = df[df["Setback"].between(0, 6)].copy()
    if "N_DR_Rows" in df.columns:
        df = df[safe_num(df["N_DR_Rows"]) >= 3].copy()
    df["Setback_sq"] = df["Setback"] ** 2
    df["Setback_Bin"] = pd.cut(df["Setback"], bins=SETBACK_BINS, labels=SETBACK_LABELS,
                               include_lowest=True, right=True).astype(str)

    # time ordering
    tcol = None
    for c in ["Session_Start", "session_start", "DR_Start"]:
        if c in df.columns:
            tcol = c
            break
    if tcol:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.sort_values(["Identifier", tcol]).reset_index(drop=True)
    else:
        df = df.sort_values(["Identifier"]).reset_index(drop=True)

    # 时间补全
    if "Month" not in df.columns and tcol:
        df["Month"] = df[tcol].dt.month
    if "Is_Weekend" not in df.columns and tcol:
        df["Is_Weekend"] = df[tcol].dt.dayofweek.isin([5, 6]).astype(int)
    if "Hour_of_Day" not in df.columns and tcol:
        df["Hour_of_Day"] = df[tcol].dt.hour

    # reduction / delivered
    if "Cool_Reduction_Frac" in df.columns:
        df["Cool_Reduction_Frac"] = safe_num(df["Cool_Reduction_Frac"])
        df["Delivered"] = df["Cool_Reduction_Frac"] * (1 - df["Y"])

    # 历史特征：仅用此前事件，按时间重算
    g = df.groupby("Identifier", sort=False)
    df["Session_Seq_Recomputed"] = g.cumcount() + 1
    df["Prev_OptOut_Recomputed"] = g["Y"].shift(1)
    cum = g["Y"].cumsum()
    n_seen = g.cumcount() + 1
    pop = float(df["Y"].mean())
    df["Prior_OptOut_Rate_Recomputed"] = np.where(n_seen - 1 > 0, (cum - df["Y"]) / (n_seen - 1), pop)
    streaks = []
    for _, sub in df.groupby("Identifier", sort=False):
        cur = 0
        for v in sub["Y"].values:
            streaks.append(cur)
            cur = cur + 1 if v == 1 else 0
    df["OptOut_Streak_Recomputed"] = streaks

    log(f"filtered cooling sessions: {len(df):,}; users: {df['Identifier'].nunique():,}; "
        f"opt-out: {df['Y'].mean():.4f}")
    return df, tcol


def avail(df, cands):
    return [c for c in cands if c in df.columns]


# ---------------------------------------------------------------- modeling
def make_clf():
    if HAS_LGBM:
        return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31,
                              subsample=0.85, colsample_bytree=0.85, min_child_samples=80,
                              reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=31,
                                          min_samples_leaf=80, random_state=RANDOM_STATE)


def make_reg():
    if HAS_LGBM:
        return LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                             subsample=0.85, colsample_bytree=0.85, min_child_samples=120,
                             reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31,
                                         min_samples_leaf=100, random_state=RANDOM_STATE)


def encode(train, test, feats):
    num = [c for c in feats if c not in CAT_COLS]
    cat = [c for c in feats if c in CAT_COLS]
    Xtr, Xte = [], []
    if num:
        a = train[num].apply(safe_num)
        b = test[num].apply(safe_num)
        med = a.median()
        Xtr.append(a.fillna(med).astype(float))
        Xte.append(b.fillna(med).astype(float))
    if cat:
        a = pd.get_dummies(train[cat].astype(str).fillna("M"), columns=cat, dummy_na=False)
        b = pd.get_dummies(test[cat].astype(str).fillna("M"), columns=cat, dummy_na=False)
        a, b = a.align(b, join="left", axis=1, fill_value=0)
        Xtr.append(a.astype(float))
        Xte.append(b.astype(float))
    return pd.concat(Xtr, axis=1), pd.concat(Xte, axis=1)


def ece(y, p, nb=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["b"] = pd.qcut(d["p"].rank(method="first"), nb, labels=False, duplicates="drop")
    return float(sum(len(g) / len(d) * abs(g["y"].mean() - g["p"].mean()) for _, g in d.groupby("b")))


def cal_slope(y, p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(solver="lbfgs").fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def gkf_oof(df, feats):
    """同一套 GroupKFold 上产出 OOF 概率。"""
    y = df["Y"].astype(int).values
    groups = df["Identifier"].astype(str).values
    oof = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, te in gkf.split(df, y, groups):
        Xtr, Xte = encode(df.iloc[tr], df.iloc[te], feats)
        m = make_clf()
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def metric_row(name, y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ci, cs = cal_slope(y, p)
    return {"model": name, "n": len(y), "auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, p, labels=[0, 1]), "ece": ece(y, p),
            "cal_intercept": ci, "cal_slope": cs}


# ---------------------------------------------------------------- B1 ablation
def exp_B1(df):
    log("\n[B1] history-only / no-history ablation (GroupKFold OOF)")
    full = avail(df, WEATHER_TIME + SETBACK_FEATS + BUILDING_BASELINE + HISTORY)
    hist = avail(df, HISTORY)
    nohist = avail(df, WEATHER_TIME + SETBACK_FEATS + BUILDING_BASELINE)
    sets = {"full": full, "history_only": hist, "no_history": nohist}
    rows, oof_full = [], None
    y = df["Y"].astype(int).values
    for name, feats in sets.items():
        oof = gkf_oof(df, feats)
        r = metric_row(name, y, oof)
        r["n_features"] = len(feats)
        rows.append(r)
        log(f"  {name:13s}: AUC={r['auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
            f"Brier={r['brier']:.4f}  ECE={r['ece']:.4f}  ({len(feats)} feats)")
        if name == "full":
            oof_full = oof
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "B1_history_ablation.csv", index=False)

    plt.figure(figsize=(6.2, 4.0))
    plt.bar(tab["model"], tab["auc"], color=["#4472c4", "#c0504d", "#70ad47"], edgecolor="black")
    for i, v in enumerate(tab["auc"]):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(0.5, 1.0)
    plt.ylabel("OOF AUC")
    plt.title("Reliability model: full vs history-only vs no-history")
    plt.tight_layout()
    plt.savefig(FIG / "B1_history_ablation.png", dpi=200)
    plt.close()
    return oof_full


# ---------------------------------------------------------------- B2 bootstrap 2x
def risk_group(prev, prior):
    if prev == 0 and prior <= 0.10:
        return "low_risk"
    if prev == 1 or prior > 0.40:
        return "high_risk"
    return "medium_risk"


def exp_B2(df):
    log("\n[B2] 3-4F headline 2x with user-clustered bootstrap CI")
    if "Delivered" not in df.columns:
        log("  Cool_Reduction_Frac 缺失，跳过 B2")
        return
    sub = df[(df["Setback_Bin"] == "3-4") & df["Delivered"].notna()].copy()
    pop = float(df["Y"].mean())
    prev = sub["Prev_OptOut_Recomputed"].fillna(0).astype(int).values
    prior = sub["Prior_OptOut_Rate_Recomputed"].fillna(pop).values
    sub["grp"] = [risk_group(p, r) for p, r in zip(prev, prior)]

    groups = ["low_risk", "medium_risk", "high_risk"]
    point = {}
    for grp in groups:
        s = sub[sub["grp"] == grp]
        point[grp] = {"n": len(s), "users": s["Identifier"].nunique(),
                      "delivered": s["Delivered"].mean(),
                      "nominal": s["Cool_Reduction_Frac"].mean(),
                      "optout": s["Y"].mean()}

    # user-clustered bootstrap
    users = sub["Identifier"].values
    uniq = np.unique(users)
    rows_by_user = {u: np.where(users == u)[0] for u in uniq}
    deliv = sub["Delivered"].values
    grp_arr = sub["grp"].values
    rng = np.random.default_rng(RANDOM_STATE)

    boot = {g: [] for g in groups}
    boot_ratio, boot_diff = [], []
    for _ in range(N_BOOT):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_user[u] for u in samp])
        gd, dv = grp_arr[idx], deliv[idx]
        means = {}
        for g in groups:
            mask = gd == g
            means[g] = dv[mask].mean() if mask.any() else np.nan
            boot[g].append(means[g])
        if means["high_risk"] and means["high_risk"] > 0:
            boot_ratio.append(means["low_risk"] / means["high_risk"])
            boot_diff.append(means["low_risk"] - means["high_risk"])

    def ci(a):
        a = np.array([x for x in a if np.isfinite(x)])
        return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    out = []
    for g in groups:
        lo, hi = ci(boot[g])
        out.append({"group": g, **point[g], "delivered_ci_lo": lo, "delivered_ci_hi": hi})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "B2_same_3_4_delivered_bootstrap.csv", index=False)

    r_lo, r_hi = ci(boot_ratio)
    d_lo, d_hi = ci(boot_diff)
    ratio_pt = point["low_risk"]["delivered"] / point["high_risk"]["delivered"]
    diff_pt = point["low_risk"]["delivered"] - point["high_risk"]["delivered"]
    pd.DataFrame([{"metric": "low/high ratio", "point": ratio_pt, "ci_lo": r_lo, "ci_hi": r_hi},
                  {"metric": "low-high diff", "point": diff_pt, "ci_lo": d_lo, "ci_hi": d_hi}]
                 ).to_csv(OUT / "B2_ratio_ci.csv", index=False)

    log(res.to_string(index=False))
    log(f"  low/high delivered ratio = {ratio_pt:.2f}x  95%CI [{r_lo:.2f}, {r_hi:.2f}]")
    log(f"  low-high delivered diff  = {diff_pt:.3f}    95%CI [{d_lo:.3f}, {d_hi:.3f}]")
    log(f"  nominal_reduction by group (应近似相同): "
        f"low={point['low_risk']['nominal']:.3f}, med={point['medium_risk']['nominal']:.3f}, "
        f"high={point['high_risk']['nominal']:.3f}")

    plt.figure(figsize=(6.4, 4.2))
    x = np.arange(3)
    dv = [point[g]["delivered"] for g in groups]
    nv = [point[g]["nominal"] for g in groups]
    err = np.array([[point[g]["delivered"] - ci(boot[g])[0], ci(boot[g])[1] - point[g]["delivered"]]
                    for g in groups]).T
    plt.bar(x - 0.18, nv, width=0.36, label="Nominal reduction", color="#bbbbbb", edgecolor="black")
    plt.bar(x + 0.18, dv, width=0.36, yerr=err, capsize=4, label="Delivered flexibility",
            color="#4472c4", edgecolor="black")
    plt.xticks(x, ["Low risk", "Medium", "High risk"])
    plt.ylabel("Cooling runtime fraction")
    plt.title("Same 3-4F setback: identical physics, behavior drives delivered (95% CI)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "B2_same_3_4_delivered.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------- C money figure
def exp_C(df, tcol):
    log("\n[C] aggregation-error money figure (out-of-time)")
    if "Delivered" not in df.columns or tcol is None:
        log("  缺 Cool_Reduction_Frac 或时间列，跳过 C")
        return
    d = df.dropna(subset=[tcol, "Cool_Reduction_Frac"]).copy()
    cutoff = d[tcol].quantile(0.70)
    tr = d[d[tcol] <= cutoff].copy()
    te = d[d[tcol] > cutoff].copy()
    log(f"  cutoff={cutoff}; train n={len(tr):,}, test n={len(te):,}")

    feats = avail(df, WEATHER_TIME + SETBACK_FEATS + BUILDING_BASELINE + HISTORY)
    # acceptance（事前）
    Xtr, Xte = encode(tr, te, feats)
    clf = make_clf()
    clf.fit(Xtr, tr["Y"].astype(int).values)
    p_oo = clf.predict_proba(Xte)[:, 1]
    p_acc = 1 - p_oo

    # reduction（仅在非 opt-out 训练，观测 setback 上预测，不改 setback）
    tr_red = tr[tr["Y"] == 0]
    Xtr_r, Xte_r = encode(tr_red, te, feats)
    reg = make_reg()
    reg.fit(Xtr_r, tr_red["Cool_Reduction_Frac"].astype(float).values)
    pred_red = np.clip(reg.predict(Xte_r), -1, 1)

    nominal = te["Cool_Reduction_Frac"].values            # 按报名/nominal：假设 100% 接受
    truth = te["Delivered"].values                        # 实际 delivered
    accept_aware = p_acc * nominal                         # 行为修正（用观测 reduction）
    full_fcast = p_acc * pred_red                          # 全事前预测（reduction 也预测）

    m_nom, m_tru = nominal.mean(), truth.mean()
    m_acc, m_full = accept_aware.mean(), full_fcast.mean()

    def err(pred_mean):
        return (pred_mean - m_tru) / m_tru * 100.0

    summ = pd.DataFrame([
        {"method": "naive_nominal (enrolled count)", "mean_per_session": m_nom, "pct_err_vs_truth": err(m_nom)},
        {"method": "behavior_aware_accept_only",     "mean_per_session": m_acc, "pct_err_vs_truth": err(m_acc)},
        {"method": "behavior_aware_full_forecast",   "mean_per_session": m_full, "pct_err_vs_truth": err(m_full)},
        {"method": "observed_delivered (truth)",     "mean_per_session": m_tru, "pct_err_vs_truth": 0.0},
    ])

    # user-clustered bootstrap CI for the two error %
    users = te["Identifier"].values
    uniq = np.unique(users)
    rbu = {u: np.where(users == u)[0] for u in uniq}
    rng = np.random.default_rng(RANDOM_STATE)
    e_nom, e_acc = [], []
    for _ in range(N_BOOT):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rbu[u] for u in samp])
        t = truth[idx].mean()
        e_nom.append((nominal[idx].mean() - t) / t * 100)
        e_acc.append((accept_aware[idx].mean() - t) / t * 100)
    summ.loc[summ["method"].str.startswith("naive"), "err_ci_lo"] = np.quantile(e_nom, 0.025)
    summ.loc[summ["method"].str.startswith("naive"), "err_ci_hi"] = np.quantile(e_nom, 0.975)
    summ.loc[summ["method"].str.startswith("behavior_aware_accept"), "err_ci_lo"] = np.quantile(e_acc, 0.025)
    summ.loc[summ["method"].str.startswith("behavior_aware_accept"), "err_ci_hi"] = np.quantile(e_acc, 0.975)
    summ.to_csv(OUT / "C_aggregation_error.csv", index=False)
    log(summ.to_string(index=False))

    # 按预测可靠性十分位，看 naive 在哪最离谱
    te2 = te.copy()
    te2["pred_accept"] = p_acc
    te2["nominal"] = nominal
    te2["truth"] = truth
    te2["dec"] = pd.qcut(te2["pred_accept"].rank(method="first"), 10, labels=False) + 1
    dec = te2.groupby("dec").agg(n=("truth", "size"),
                                 pred_accept=("pred_accept", "mean"),
                                 nominal=("nominal", "mean"),
                                 truth=("truth", "mean")).reset_index()
    dec["naive_overest_pct"] = (dec["nominal"] - dec["truth"]) / dec["truth"] * 100
    dec.to_csv(OUT / "C_error_by_reliability_decile.csv", index=False)

    plt.figure(figsize=(6.4, 4.2))
    mth = ["naive\nnominal", "behavior\naccept-only", "behavior\nfull", "observed\n(truth)"]
    vals = [m_nom, m_acc, m_full, m_tru]
    cols = ["#c0504d", "#4472c4", "#8064a2", "#70ad47"]
    plt.bar(mth, vals, color=cols, edgecolor="black")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("Mean delivered flexibility per session")
    plt.title("Capacity accounting error (out-of-time test)")
    plt.tight_layout()
    plt.savefig(FIG / "C_aggregation_error.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------- D per-state
def exp_D(df, oof_full):
    log("\n[D] per-state external validity (full-model OOF)")
    if oof_full is None or "province_state" not in df.columns:
        log("  缺 OOF 或 province_state，跳过 D")
        return
    d = df.copy()
    d["pred_oo"] = oof_full
    rows = []
    for st, s in d.groupby("province_state"):
        y = s["Y"].astype(int).values
        p = s["pred_oo"].values
        rows.append({"state": str(st), "n": len(s), "users": s["Identifier"].nunique(),
                     "share": len(s) / len(d), "optout_rate": y.mean(),
                     "auc": roc_auc_score(y, p) if (len(s) >= 100 and len(np.unique(y)) == 2) else np.nan,
                     "ece": ece(y, p) if len(s) >= 100 else np.nan})
    tab = pd.DataFrame(rows).sort_values("n", ascending=False)
    tab.to_csv(OUT / "D_per_state_validity.csv", index=False)
    log(tab.head(12).to_string(index=False))
    top = tab.iloc[0]
    log(f"  top state {top['state']} 占 {top['share']:.1%} 的 session（外部效度需 hedge）")


# ---------------------------------------------------------------- pack
def pack():
    files = [p for p in OUT.rglob("*") if p.is_file()]
    report = OUT / "extra_experiments_report.txt"
    report.write_text("\n".join(LINES), encoding="utf-8")
    if report not in files:
        files.append(report)
    manifest = [{"path": str(f.relative_to(OUT)), "size_kb": round(f.stat().st_size / 1024, 1)}
                for f in files]
    (OUT / "_manifest.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"),
                    "lgbm": HAS_LGBM, "n_files": len(files) + 1, "files": manifest},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    zip_path = Path("extra_experiments_bundle.zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in OUT.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(OUT.parent))
    log("\n" + "=" * 74)
    log(f"打包完成: {zip_path.resolve()}  ({zip_path.stat().st_size/1024:.1f} KB)")
    log("把 extra_experiments_bundle.zip 发回来即可。")
    log("=" * 74)


def main():
    df, tcol = prep()
    oof_full = exp_B1(df)
    exp_B2(df)
    exp_C(df, tcol)
    exp_D(df, oof_full)
    pack()


if __name__ == "__main__":
    main()
