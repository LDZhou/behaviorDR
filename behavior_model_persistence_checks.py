#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_persistence_causality_sanity.py

Additional experiments for the ecobee smart thermostat DR project.

A. Persistence / causality-oriented sanity checks
   1. First-opt-out event study
   2. Matched first-opt-out comparison
   3. User fixed-effect lag LPM
   4. Lead/placebo comparison

B. Reliability-score prediction sanity checks
   5. User-level GroupKFold LightGBM validation
   6. Reliability decile validation
   7. Calibration curve / ECE / calibration slope
   8. Out-of-time validation
   9. Same-setback 3--4°F heterogeneity by predicted reliability score

Important:
These analyses improve credibility but do NOT prove causality.
They test whether recent opt-out has extra predictive power beyond stable user type
and whether the reliability score is useful/calibrated.

Run:
    python 14_persistence_causality_sanity.py

Expected input:
    dr_sessions.csv

Outputs:
    persistence_sanity_out/
"""

import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGBM = False


OUTDIR = Path("persistence_sanity_out")
FIGDIR = OUTDIR / "figs"
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

ID_COL = "Identifier"
INPUT_CANDIDATES = ["dr_sessions.csv", "dr_sessions_v2.csv"]
Y_PRIORITY = ["OptOut_Immediate", "Y_immediate", "OptOut", "OptOut_Original", "OptOut_30min"]
TIME_PRIORITY = ["Session_Start", "Start", "Event_Start", "date_time_start"]
SETBACK_PRIORITY = ["Setback_Amplitude_Mean", "Setback", "Setback_F", "Setback_Amplitude"]


def find_file():
    for p in INPUT_CANDIDATES:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError("Cannot find dr_sessions.csv or dr_sessions_v2.csv")


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Cannot find any of: {candidates}")
    return None


def make_binary(s):
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "hold", "optout", "opt_out"]).astype(int)


def load_sessions():
    path = find_file()
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df):,} rows")

    if ID_COL not in df.columns:
        raise KeyError(f"Missing {ID_COL}")

    time_col = pick_col(df, TIME_PRIORITY, required=False)
    if time_col is None:
        df["_time"] = np.arange(len(df))
        time_col = "_time"
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    y_col = pick_col(df, Y_PRIORITY, required=True)
    df["_Y"] = make_binary(df[y_col])

    setback_col = pick_col(df, SETBACK_PRIORITY, required=False)
    if setback_col is None:
        df["_Setback"] = np.nan
    else:
        df["_Setback"] = pd.to_numeric(df[setback_col], errors="coerce")

    # Cooling-only if possible
    if "HvacMode" in df.columns:
        m = df["HvacMode"].astype(str).str.lower().str.contains("cool", na=False)
        if m.sum() > 1000:
            df = df.loc[m].copy()
            print(f"Filtered cooling sessions: {len(df):,}")

    df = df.dropna(subset=[ID_COL]).copy()
    df = df.sort_values([ID_COL, time_col]).reset_index(drop=True)

    # recompute history using only prior events
    g = df.groupby(ID_COL, sort=False)
    df["_SessionSeq"] = g.cumcount() + 1
    df["_PriorCount"] = g.cumcount()
    df["_PrevOptOut"] = g["_Y"].shift(1).fillna(0).astype(int)
    df["_PriorOptOutCount"] = g["_Y"].cumsum() - df["_Y"]
    df["_PriorOptOutRate"] = np.where(
        df["_PriorCount"] > 0,
        df["_PriorOptOutCount"] / df["_PriorCount"],
        0.0,
    )

    streaks = []
    for _, sub in df.groupby(ID_COL, sort=False):
        cur = 0
        for y in sub["_Y"].values:
            streaks.append(cur)
            cur = cur + 1 if y == 1 else 0
    df["_OptOutStreak"] = streaks

    df["_NextOptOut"] = g["_Y"].shift(-1)
    df["_HasNext"] = df["_NextOptOut"].notna()
    df["_NextOptOut"] = df["_NextOptOut"].astype(float)
    df["_Prev2OptOut"] = g["_Y"].shift(2)
    df["_Next2OptOut"] = g["_Y"].shift(-2)

    bins = [0, 1, 2, 3, 4, 6]
    labels = ["0-1", "1-2", "2-3", "3-4", "4-6"]
    df["_SetbackClip"] = df["_Setback"].clip(0, 6)
    df["_Setback_Bin"] = pd.cut(df["_SetbackClip"], bins=bins, labels=labels, include_lowest=True)

    print(f"Working sample: {len(df):,} sessions, {df[ID_COL].nunique():,} users")
    print(f"Opt-out rate: {df['_Y'].mean():.2%}")
    return df, time_col


def feature_groups(df):
    weather_time = [
        "Tout_onset", "RH_onset", "GHI_onset", "CDH_during",
        "Duration_Min", "Hour_of_Day", "Is_Weekend", "Month",
        "Hour_Bin", "DR_Type", "province_state", "country",
    ]
    setback = ["_Setback", "_Setback_Bin"]
    building = [
        "floor_area_sqft", "building_age_yrs", "number_occupants",
        "has_heatpump", "has_electric", "number_cool_stages",
        "building_type", "Baseline_Cool_Frac", "Avg_Baseline_Temp",
        "Setpoint_Cool_Start", "Avg_Baseline_Cool", "weather_is_fallback",
    ]
    history = ["_PrevOptOut", "_PriorOptOutRate", "_OptOutStreak", "_SessionSeq"]
    groups = {
        "weather_time": [c for c in weather_time if c in df.columns],
        "setback": [c for c in setback if c in df.columns],
        "building_baseline": [c for c in building if c in df.columns],
        "user_history": [c for c in history if c in df.columns],
    }
    return groups


def make_preprocess(df, features):
    num, cat = [], []
    for c in features:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    transformers = []
    if num:
        transformers.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    return ColumnTransformer(transformers, remainder="drop")


def make_model():
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=40,
            reg_lambda=1.0,
            objective="binary",
            random_state=42,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42,
    )


def ece_score(y_true, p, n_bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p <= hi) if hi == 1 else (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred = float(p[mask].mean())
        obs = float(y_true[mask].mean())
        ece += (n / len(y_true)) * abs(obs - pred)
        rows.append({"bin_low": lo, "bin_high": hi, "n": n, "pred": pred, "obs": obs, "abs_error": abs(obs - pred)})
    return ece, pd.DataFrame(rows)


def calibration_intercept_slope(y, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p)
    try:
        res = sm.Logit(y, X).fit(disp=False, maxiter=200)
        return float(res.params[0]), float(res.params[1])
    except Exception:
        return np.nan, np.nan


def metric_dict(y, p, name):
    out = {"model": name, "n": len(y), "event_rate": float(np.mean(y))}
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    out["pr_auc"] = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan
    out["brier"] = brier_score_loss(y, p)
    out["logloss"] = log_loss(y, p, labels=[0, 1])
    ece, _ = ece_score(y, p)
    out["ece"] = ece
    inter, slope = calibration_intercept_slope(y, p)
    out["calibration_intercept"] = inter
    out["calibration_slope"] = slope
    return out


def encode_for_sm(d, covs, min_cat=50):
    out = d.copy()
    for c in covs:
        if c not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(pd.to_numeric(out[c], errors="coerce").median())
        else:
            vc = out[c].astype(str).value_counts()
            keep = set(vc[vc >= min_cat].index)
            out[c] = out[c].astype(str).where(out[c].astype(str).isin(keep), "Other")
            dd = pd.get_dummies(out[c], prefix=c, drop_first=True, dtype=float)
            out = pd.concat([out.drop(columns=[c]), dd], axis=1)
    return out


def covariates_for_controls(df):
    candidates = [
        "_Setback", "Duration_Min", "Tout_onset", "RH_onset", "GHI_onset", "CDH_during",
        "Month", "Hour_of_Day", "Is_Weekend", "Baseline_Cool_Frac", "Avg_Baseline_Temp",
        "Setpoint_Cool_Start", "_SessionSeq", "province_state", "DR_Type", "_Setback_Bin",
    ]
    return [c for c in candidates if c in df.columns]


def first_optout_event_study(df):
    print("\n[1] First-opt-out event study")
    d = df[df["_HasNext"]].copy()
    risk = d[d["_PriorOptOutCount"] == 0].copy()
    risk["first_optout_at_t"] = risk["_Y"].astype(int)
    risk["next_optout"] = risk["_NextOptOut"].astype(int)

    rows = []
    for tr, sub in risk.groupby("first_optout_at_t"):
        rows.append({
            "group": "first_optout_at_t" if tr == 1 else "stay_at_t_no_prior_optout",
            "n": len(sub),
            "users": sub[ID_COL].nunique(),
            "next_optout_rate": sub["next_optout"].mean(),
            "mean_setback": sub["_Setback"].mean(),
            "mean_duration": sub["Duration_Min"].mean() if "Duration_Min" in sub.columns else np.nan,
            "mean_tout": sub["Tout_onset"].mean() if "Tout_onset" in sub.columns else np.nan,
        })
    raw = pd.DataFrame(rows)
    raw.to_csv(OUTDIR / "first_optout_event_study_raw.csv", index=False)

    model_rows = []
    if risk["first_optout_at_t"].nunique() == 2 and risk["next_optout"].nunique() == 2:
        covs = covariates_for_controls(risk)
        tmp = risk[["next_optout", "first_optout_at_t", ID_COL] + covs].copy()
        tmp = encode_for_sm(tmp, covs)
        y = tmp["next_optout"].astype(int)
        X = sm.add_constant(tmp.drop(columns=["next_optout", ID_COL]), has_constant="add")
        try:
            res = sm.Logit(y, X).fit(disp=False, maxiter=200)
            coef = res.params.get("first_optout_at_t", np.nan)
            model_rows.append({
                "model": "next_optout_logit_controls",
                "term": "first_optout_at_t",
                "coef": coef,
                "odds_ratio": float(np.exp(coef)) if pd.notna(coef) else np.nan,
                "pvalue": res.pvalues.get("first_optout_at_t", np.nan),
                "n": len(tmp),
                "users": tmp[ID_COL].nunique(),
            })
        except Exception as e:
            model_rows.append({"model": "next_optout_logit_controls", "error": repr(e), "n": len(tmp)})

    model = pd.DataFrame(model_rows)
    model.to_csv(OUTDIR / "first_optout_event_study_model.csv", index=False)

    if not raw.empty:
        plt.figure(figsize=(6.3, 4.2))
        plt.bar(raw["group"], raw["next_optout_rate"] * 100)
        plt.ylabel("Next-event opt-out rate (%)")
        plt.title("Next opt-out after first opt-out vs stay")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(FIGDIR / "fig_first_optout_event_study.png", dpi=200)
        plt.close()

    print(raw.to_string(index=False))
    if not model.empty:
        print(model.to_string(index=False))
    return raw, model


def matched_first_optout(df):
    print("\n[2] Matched first-opt-out comparison")
    d = df[df["_HasNext"]].copy()
    risk = d[d["_PriorOptOutCount"] == 0].copy()
    risk["first_optout_at_t"] = risk["_Y"].astype(int)
    risk["next_optout"] = risk["_NextOptOut"].astype(int)

    if risk["first_optout_at_t"].nunique() < 2:
        print("  Not enough variation.")
        return pd.DataFrame()

    covs = covariates_for_controls(risk)
    pre = make_preprocess(risk, covs)
    X = pre.fit_transform(risk[covs])
    t = risk["first_optout_at_t"].values

    clf = make_model()
    clf.fit(X, t)
    ps = clf.predict_proba(X)[:, 1]
    risk["_ps_first_optout"] = ps

    treated = risk[risk["first_optout_at_t"] == 1].copy()
    control = risk[risk["first_optout_at_t"] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["_ps_first_optout"]].values)
    dist, idx = nn.kneighbors(treated[["_ps_first_optout"]].values)
    mc = control.iloc[idx[:, 0]].copy()

    matched = pd.DataFrame({
        "treated_index": treated.index.values,
        "control_index": mc.index.values,
        "treated_ps": treated["_ps_first_optout"].values,
        "control_ps": mc["_ps_first_optout"].values,
        "abs_ps_diff": np.abs(treated["_ps_first_optout"].values - mc["_ps_first_optout"].values),
        "treated_next_optout": treated["next_optout"].values,
        "control_next_optout": mc["next_optout"].values,
        "treated_setback": treated["_Setback"].values,
        "control_setback": mc["_Setback"].values,
    })
    matched.to_csv(OUTDIR / "matched_first_optout_pairs.csv", index=False)

    caliper = matched["abs_ps_diff"].quantile(0.95)
    matched_cal = matched[matched["abs_ps_diff"] <= caliper]
    summary = pd.DataFrame([
        {
            "sample": "all_matched",
            "treated_n": len(matched),
            "treated_next_optout": matched["treated_next_optout"].mean(),
            "control_next_optout": matched["control_next_optout"].mean(),
            "difference": matched["treated_next_optout"].mean() - matched["control_next_optout"].mean(),
            "mean_abs_ps_diff": matched["abs_ps_diff"].mean(),
        },
        {
            "sample": "caliper_95pct",
            "treated_n": len(matched_cal),
            "treated_next_optout": matched_cal["treated_next_optout"].mean(),
            "control_next_optout": matched_cal["control_next_optout"].mean(),
            "difference": matched_cal["treated_next_optout"].mean() - matched_cal["control_next_optout"].mean(),
            "mean_abs_ps_diff": matched_cal["abs_ps_diff"].mean(),
        }
    ])
    summary.to_csv(OUTDIR / "matched_first_optout_summary.csv", index=False)

    plt.figure(figsize=(6.2, 4.2))
    vals = [summary.loc[0, "control_next_optout"], summary.loc[0, "treated_next_optout"]]
    plt.bar(["Matched stay at t", "First opt-out at t"], np.array(vals) * 100)
    plt.ylabel("Next-event opt-out rate (%)")
    plt.title("Matched comparison: next opt-out")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_matched_first_optout.png", dpi=200)
    plt.close()

    print(summary.to_string(index=False))
    return summary


def user_fe_lag_lpm(df):
    print("\n[3] User fixed-effect lag LPM")
    d = df[df["_PriorCount"] > 0].copy()
    covs = ["_PrevOptOut", "_PriorOptOutRate", "_Setback", "Duration_Min", "Tout_onset", "Month", "Hour_of_Day", "Is_Weekend"]
    covs = [c for c in covs if c in d.columns]
    for c in covs:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].fillna(d[c].median())

    counts = d.groupby(ID_COL)["_Y"].agg(["count", "nunique"])
    keep = counts[(counts["count"] >= 2) & (counts["nunique"] >= 2)].index
    d = d[d[ID_COL].isin(keep)].copy()
    if len(d) < 100:
        out = pd.DataFrame([{"model": "user_fe_lpm", "error": "not enough within-user variation"}])
        out.to_csv(OUTDIR / "user_fe_lag_lpm.csv", index=False)
        return out

    y_dm = d["_Y"] - d.groupby(ID_COL)["_Y"].transform("mean")
    X_dm = pd.DataFrame(index=d.index)
    for c in covs:
        X_dm[c] = d[c] - d.groupby(ID_COL)[c].transform("mean")
    use = [c for c in covs if X_dm[c].std() > 1e-8]
    X = sm.add_constant(X_dm[use], has_constant="add")
    res = sm.OLS(y_dm, X).fit(cov_type="cluster", cov_kwds={"groups": d[ID_COL].values})

    rows = []
    for term in ["_PrevOptOut", "_PriorOptOutRate", "_Setback"]:
        if term in res.params.index:
            rows.append({
                "model": "user_fe_lpm_within",
                "term": term,
                "coef": res.params[term],
                "se": res.bse[term],
                "pvalue": res.pvalues[term],
                "n": len(d),
                "users": d[ID_COL].nunique(),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTDIR / "user_fe_lag_lpm.csv", index=False)
    print(tab.to_string(index=False))
    return tab


def lead_placebo(df):
    print("\n[4] Lead/placebo comparison")
    d = df[(df["_PriorCount"] > 0) & df["_NextOptOut"].notna()].copy()
    d["_NextOptOut_int"] = d["_NextOptOut"].astype(int)
    covs = ["_PrevOptOut", "_NextOptOut_int", "_PriorOptOutRate", "_Setback", "Duration_Min", "Tout_onset", "Month", "Hour_of_Day", "Is_Weekend", "province_state", "DR_Type"]
    covs = [c for c in covs if c in d.columns]
    tmp = d[["_Y", ID_COL] + covs].copy()
    tmp = encode_for_sm(tmp, covs)
    y = tmp["_Y"].astype(int)
    X = sm.add_constant(tmp.drop(columns=["_Y", ID_COL]), has_constant="add")
    rows = []
    try:
        res = sm.Logit(y, X).fit(disp=False, maxiter=200)
        for term in ["_PrevOptOut", "_NextOptOut_int", "_PriorOptOutRate"]:
            if term in res.params.index:
                rows.append({
                    "model": "lead_placebo_logit",
                    "term": term,
                    "coef": res.params[term],
                    "odds_ratio": float(np.exp(res.params[term])),
                    "pvalue": res.pvalues[term],
                    "n": len(tmp),
                    "users": tmp[ID_COL].nunique(),
                })
    except Exception as e:
        rows.append({"model": "lead_placebo_logit", "error": repr(e), "n": len(tmp), "users": tmp[ID_COL].nunique()})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTDIR / "lead_placebo_test.csv", index=False)
    print(tab.to_string(index=False))
    return tab


def fit_groupkfold(df, features):
    print("\n[5] User-level GroupKFold reliability score validation")
    d = df.copy()
    y = d["_Y"].astype(int).values
    groups = d[ID_COL].values
    oof = np.full(len(d), np.nan)
    fold_rows = []
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(d, y, groups), start=1):
        pre = make_preprocess(d.iloc[tr], features)
        pipe = Pipeline([("prep", pre), ("model", make_model())])
        pipe.fit(d.iloc[tr][features], y[tr])
        p = pipe.predict_proba(d.iloc[te][features])[:, 1]
        oof[te] = p
        fold_rows.append({
            "fold": fold,
            "train_users": len(np.unique(groups[tr])),
            "test_users": len(np.unique(groups[te])),
            "test_n": len(te),
        })
        print(f"  fold {fold}: train users={fold_rows[-1]['train_users']:,}, test users={fold_rows[-1]['test_users']:,}")
    d["_PredOptOut"] = oof
    d["_ReliabilityScore"] = 1 - oof

    metrics = metric_dict(y, oof, "groupkfold_full_model")
    pd.DataFrame([metrics]).to_csv(OUTDIR / "groupkfold_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTDIR / "groupkfold_folds.csv", index=False)
    print("  Metrics:", metrics)
    return d, metrics


def decile_and_calibration(df_pred, prefix="groupkfold"):
    d = df_pred.copy()
    d["_decile"] = pd.qcut(d["_ReliabilityScore"].rank(method="first"), 10, labels=False) + 1
    rows = []
    for dec, sub in d.groupby("_decile"):
        rows.append({
            "decile": int(dec),
            "n": len(sub),
            "predicted_acceptance": sub["_ReliabilityScore"].mean(),
            "observed_acceptance": 1 - sub["_Y"].mean(),
            "predicted_optout": sub["_PredOptOut"].mean(),
            "observed_optout": sub["_Y"].mean(),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTDIR / f"{prefix}_reliability_deciles.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(tab["decile"], tab["observed_acceptance"] * 100, marker="o", label="Observed acceptance")
    plt.plot(tab["decile"], tab["predicted_acceptance"] * 100, marker="s", label="Predicted acceptance")
    plt.xlabel("Predicted reliability decile")
    plt.ylabel("Acceptance rate (%)")
    plt.title("Reliability score validation")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / f"{prefix}_reliability_deciles.png", dpi=200)
    plt.close()

    ece, caltab = ece_score(d["_Y"].values, d["_PredOptOut"].values)
    caltab.to_csv(OUTDIR / f"{prefix}_calibration_table.csv", index=False)
    plt.figure(figsize=(5.4, 5))
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.scatter(caltab["pred"], caltab["obs"], s=np.maximum(25, caltab["n"] / caltab["n"].max() * 180))
    plt.xlabel("Predicted opt-out probability")
    plt.ylabel("Observed opt-out rate")
    plt.title("Calibration")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / f"{prefix}_calibration.png", dpi=200)
    plt.close()
    return tab, caltab


def same_setback_heterogeneity(df_pred):
    print("\n[6] Same 3-4°F setback heterogeneity")
    d = df_pred[df_pred["_Setback_Bin"].astype(str) == "3-4"].copy()
    if len(d) < 100:
        out = pd.DataFrame([{"error": "not enough 3-4°F sessions"}])
        out.to_csv(OUTDIR / "same_3_4_setback_heterogeneity.csv", index=False)
        return out

    d["_reliability_tertile"] = pd.qcut(
        d["_ReliabilityScore"].rank(method="first"),
        3,
        labels=["Low predicted reliability", "Medium", "High predicted reliability"],
    )
    d["_history_group"] = np.where(
        (d["_PrevOptOut"] == 1) | (d["_PriorOptOutRate"] >= 0.4),
        "High history risk",
        np.where(
            (d["_PriorOptOutRate"] <= 0.05) & (d["_PrevOptOut"] == 0),
            "Low history risk",
            "Medium history risk",
        ),
    )

    rows = []
    for var in ["_reliability_tertile", "_history_group"]:
        for group, sub in d.groupby(var):
            rows.append({
                "grouping": var,
                "group": str(group),
                "n": len(sub),
                "users": sub[ID_COL].nunique(),
                "predicted_acceptance": sub["_ReliabilityScore"].mean(),
                "observed_acceptance": 1 - sub["_Y"].mean(),
                "observed_optout": sub["_Y"].mean(),
                "mean_prior_optout_rate": sub["_PriorOptOutRate"].mean(),
                "mean_prev_optout": sub["_PrevOptOut"].mean(),
                "mean_setback": sub["_Setback"].mean(),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTDIR / "same_3_4_setback_heterogeneity.csv", index=False)

    plot_tab = tab[tab["grouping"] == "_reliability_tertile"].copy()
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(plot_tab["group"], plot_tab["observed_acceptance"] * 100)
    plt.ylabel("Observed acceptance rate (%)")
    plt.title("Same 3-4°F setback: different reliability")
    plt.ylim(0, 105)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_same_3_4_setback_reliability.png", dpi=200)
    plt.close()

    print(tab.to_string(index=False))
    return tab


def behavior_accounting_by_decile(df_pred):
    print("\n[7] Behavior accounting by reliability decile")
    d = df_pred.copy()
    if "Cool_Reduction_Frac" not in d.columns:
        alt = [c for c in d.columns if "cool" in c.lower() and "reduction" in c.lower()]
        if not alt:
            print("  No Cool_Reduction_Frac column found; skipped.")
            return pd.DataFrame()
        d["Cool_Reduction_Frac"] = pd.to_numeric(d[alt[0]], errors="coerce")

    d["Cool_Reduction_Frac"] = pd.to_numeric(d["Cool_Reduction_Frac"], errors="coerce")
    d = d.dropna(subset=["Cool_Reduction_Frac", "_ReliabilityScore"]).copy()
    d["_observed_delivered"] = d["Cool_Reduction_Frac"] * (1 - d["_Y"])
    d["_predicted_delivered_simple"] = d["Cool_Reduction_Frac"] * d["_ReliabilityScore"]
    d["_decile"] = pd.qcut(d["_ReliabilityScore"].rank(method="first"), 10, labels=False) + 1

    rows = []
    for dec, sub in d.groupby("_decile"):
        rows.append({
            "decile": int(dec),
            "n": len(sub),
            "users": sub[ID_COL].nunique(),
            "predicted_acceptance": sub["_ReliabilityScore"].mean(),
            "observed_acceptance": 1 - sub["_Y"].mean(),
            "nominal_reduction": sub["Cool_Reduction_Frac"].mean(),
            "observed_delivered": sub["_observed_delivered"].mean(),
            "predicted_delivered_simple": sub["_predicted_delivered_simple"].mean(),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTDIR / "behavior_accounting_by_reliability_decile.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(tab["decile"], tab["nominal_reduction"], marker="o", label="Nominal reduction")
    plt.plot(tab["decile"], tab["observed_delivered"], marker="s", label="Observed delivered")
    plt.plot(tab["decile"], tab["predicted_delivered_simple"], marker="^", label="Predicted delivered")
    plt.xlabel("Predicted reliability decile")
    plt.ylabel("Mean value")
    plt.title("Behavior-adjusted accounting by reliability decile")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_behavior_accounting_by_reliability_decile.png", dpi=200)
    plt.close()
    return tab


def out_of_time(df, features, time_col):
    print("\n[8] Out-of-time validation")
    if time_col == "_time":
        print("  No real time column; skipped.")
        return pd.DataFrame()
    d = df.dropna(subset=[time_col]).copy()
    cutoff = d[time_col].quantile(0.70)
    train = d[d[time_col] <= cutoff].copy()
    test = d[d[time_col] > cutoff].copy()
    print(f"  cutoff={cutoff}")
    print(f"  train n={len(train):,}, users={train[ID_COL].nunique():,}")
    print(f"  test  n={len(test):,}, users={test[ID_COL].nunique():,}")

    pre = make_preprocess(train, features)
    pipe = Pipeline([("prep", pre), ("model", make_model())])
    pipe.fit(train[features], train["_Y"].astype(int))
    p = pipe.predict_proba(test[features])[:, 1]
    test["_PredOptOut"] = p
    test["_ReliabilityScore"] = 1 - p

    metrics = metric_dict(test["_Y"].astype(int).values, p, "out_of_time_full_model")
    metrics["cutoff"] = str(cutoff)
    metrics["train_n"] = len(train)
    metrics["test_n"] = len(test)
    pd.DataFrame([metrics]).to_csv(OUTDIR / "out_of_time_metrics.csv", index=False)

    decile_and_calibration(test, prefix="out_of_time")
    print("  Metrics:", metrics)
    return pd.DataFrame([metrics])


def main():
    print("=" * 80)
    print("Persistence / causality sanity + reliability score validation")
    print("=" * 80)

    df, time_col = load_sessions()
    groups = feature_groups(df)
    features = list(dict.fromkeys(groups["weather_time"] + groups["setback"] + groups["building_baseline"] + groups["user_history"]))

    print("\nFeature groups:")
    for k, v in groups.items():
        print(f"  {k}: {v}")

    first_optout_event_study(df)
    matched_first_optout(df)
    user_fe_lag_lpm(df)
    lead_placebo(df)

    pred, metrics = fit_groupkfold(df, features)
    decile_and_calibration(pred, prefix="groupkfold")
    same_setback_heterogeneity(pred)
    behavior_accounting_by_decile(pred)
    out_of_time(df, features, time_col)

    # Summary metrics
    summary = []
    for fname in ["groupkfold_metrics.csv", "out_of_time_metrics.csv"]:
        p = OUTDIR / fname
        if p.exists():
            summary.append(pd.read_csv(p))
    if summary:
        pd.concat(summary, ignore_index=True, sort=False).to_csv(OUTDIR / "validation_metrics_summary.csv", index=False)

    with open(OUTDIR / "README_results_interpretation.txt", "w", encoding="utf-8") as f:
        f.write("Persistence / causality sanity + reliability score validation\n")
        f.write("=" * 72 + "\n\n")
        f.write("Important: these analyses do NOT prove causality.\n")
        f.write("They test whether recent opt-out marks a high-risk behavioral state\n")
        f.write("beyond stable user type and event conditions.\n\n")
        f.write("Key files:\n")
        for fn in [
            "first_optout_event_study_raw.csv",
            "first_optout_event_study_model.csv",
            "matched_first_optout_summary.csv",
            "user_fe_lag_lpm.csv",
            "lead_placebo_test.csv",
            "groupkfold_metrics.csv",
            "groupkfold_reliability_deciles.csv",
            "groupkfold_calibration_table.csv",
            "same_3_4_setback_heterogeneity.csv",
            "behavior_accounting_by_reliability_decile.csv",
            "out_of_time_metrics.csv",
        ]:
            f.write(f"- {fn}\n")

    print("\nDone.")
    print(f"Outputs written to: {OUTDIR.resolve()}")
    print("\nRecommended files to inspect:")
    for fn in [
        "first_optout_event_study_raw.csv",
        "matched_first_optout_summary.csv",
        "user_fe_lag_lpm.csv",
        "lead_placebo_test.csv",
        "groupkfold_metrics.csv",
        "groupkfold_reliability_deciles.csv",
        "same_3_4_setback_heterogeneity.csv",
        "out_of_time_metrics.csv",
    ]:
        print(f"  {OUTDIR / fn}")


if __name__ == "__main__":
    main()
