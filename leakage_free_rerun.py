#!/usr/bin/env python3
"""
Leakage-free sensitivity rerun for the ecobee behavior-aware DR paper.

This script intentionally writes to leakage_free_out/ and does not modify the
existing result folders. It removes post-dispatch / realized-duration features
from the reliability model and uses an event-onset setback proxy when available.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This rerun expects lightgbm in the HPC analysis environment") from exc

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    HAS_SM = True
except Exception:
    HAS_SM = False


DATA = Path("dr_sessions.csv")
OUT = Path("leakage_free_out")
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
SETBACK_BINS = [0, 1, 2, 3, 4, 6]
SETBACK_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-6"]


def log(msg: str = "") -> None:
    print(msg, flush=True)
    REPORT.append(str(msg))


REPORT: list[str] = []


@dataclass
class Metrics:
    name: str
    n: int
    users: int
    auc: float
    pr_auc_optout: float
    brier: float
    logloss: float
    ece_optout: float
    cal_intercept: float
    cal_slope: float
    n_features: int


def as_bool_int(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"]).astype(int)


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p).astype(float), 1e-6, 1 - 1e-6)
    y = np.asarray(y).astype(int)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(lp, y)
        return float(lr.intercept_[0]), float(lr.coef_[0][0])
    except Exception:
        return float("nan"), float("nan")


def metric_row(name: str, y: np.ndarray, p: np.ndarray, groups: Iterable, n_features: int) -> Metrics:
    ci, cs = calibration_intercept_slope(y, p)
    return Metrics(
        name=name,
        n=int(len(y)),
        users=int(pd.Series(groups).nunique()),
        auc=float(roc_auc_score(y, p)),
        pr_auc_optout=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        ece_optout=ece_score(y, p),
        cal_intercept=ci,
        cal_slope=cs,
        n_features=int(n_features),
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    log("=" * 78)
    log("Leakage-free behavior-aware DR rerun")
    log("=" * 78)
    log(f"Loaded raw rows: {len(df):,}")

    if "HvacMode" in df.columns:
        df = df[df["HvacMode"].astype(str).str.lower().eq("cool")].copy()
    df["Setback_Amplitude_Mean"] = pd.to_numeric(df["Setback_Amplitude_Mean"], errors="coerce")
    df = df[df["Setback_Amplitude_Mean"].between(0, 6)].copy()
    if "N_DR_Rows" in df.columns:
        df = df[pd.to_numeric(df["N_DR_Rows"], errors="coerce").fillna(0) >= 3].copy()

    df["Session_Start"] = pd.to_datetime(df["Session_Start"], errors="coerce")
    df = df.sort_values(["Identifier", "Session_Start"]).reset_index(drop=True)
    df["Y"] = as_bool_int(df["OptOut_Immediate"] if "OptOut_Immediate" in df.columns else df["Opted_Out"])

    # Existing observed event-mean setback is retained for descriptive bins only.
    df["Setback_Mean"] = df["Setback_Amplitude_Mean"].astype(float)
    df["Setback_Mean_Bin"] = pd.cut(
        df["Setback_Mean"], SETBACK_BINS, labels=SETBACK_LABELS, include_lowest=True, right=True
    ).astype(object)

    # Event-onset proxy: available at the first DR record, unlike the event mean.
    df["Setback_Onset"] = pd.to_numeric(df["Setpoint_Cool_Start"], errors="coerce") - pd.to_numeric(
        df["Normal_Setpoint_Cool"], errors="coerce"
    )
    df["Setback_Onset_sq"] = df["Setback_Onset"] ** 2
    onset_bin = pd.cut(
        df["Setback_Onset"], SETBACK_BINS, labels=SETBACK_LABELS, include_lowest=True, right=True
    )
    df["Setback_Onset_Bin"] = onset_bin.astype(object).where(onset_bin.notna(), "outside_0_6")

    df["Cool_Reduction_Frac"] = pd.to_numeric(df["Cool_Reduction_Frac"], errors="coerce")
    df["Delivered_Completion"] = df["Cool_Reduction_Frac"] * (1 - df["Y"])

    # Chronological histories. Leave first-event prior as NaN so fold/time train
    # imputers supply training-only values.
    df["Session_Seq_Recomputed"] = df.groupby("Identifier").cumcount() + 1
    df["Prev_OptOut_Recomputed"] = df.groupby("Identifier")["Y"].shift(1)
    cum_oo = df.groupby("Identifier")["Y"].cumsum()
    n_seen = df.groupby("Identifier").cumcount() + 1
    df["Prior_OptOut_Rate_Recomputed"] = np.where(n_seen > 1, (cum_oo - df["Y"]) / (n_seen - 1), np.nan)
    streaks: list[int] = []
    for _, g in df.groupby("Identifier", sort=False):
        st = 0
        for y in g["Y"].astype(int).values:
            streaks.append(st)
            st = st + 1 if y == 1 else 0
    df["OptOut_Streak_Recomputed"] = streaks

    if "Dew_onset" not in df.columns and "dew_onset" in df.columns:
        df["Dew_onset"] = df["dew_onset"]

    log(
        f"Filtered cooling sample: {len(df):,} sessions, "
        f"{df['Identifier'].nunique():,} users, opt-out={df['Y'].mean():.4f}"
    )
    log(f"Onset setback available for {df['Setback_Onset'].notna().mean():.1%} of sessions")
    return df


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    weather_time = available(
        df,
        [
            "Tout_onset",
            "RH_onset",
            "GHI_onset",
            "Dew_onset",
            "Hour_of_Day",
            "Is_Weekend",
            "Month",
            "Hour_Bin",
            "DR_Type",
            "province_state",
            "country",
        ],
    )
    onset_setback = available(df, ["Setback_Onset", "Setback_Onset_sq", "Setback_Onset_Bin"])
    building_baseline = available(
        df,
        [
            "floor_area_sqft",
            "building_age_yrs",
            "number_occupants",
            "has_heatpump",
            "has_electric",
            "number_cool_stages",
            "building_type",
            "Baseline_Cool_Frac",
            "Avg_Baseline_Temp",
            "Setpoint_Cool_Start",
            "Avg_Baseline_Cool",
            "weather_is_fallback",
        ],
    )
    history = available(
        df,
        [
            "Prev_OptOut_Recomputed",
            "Prior_OptOut_Rate_Recomputed",
            "OptOut_Streak_Recomputed",
            "Session_Seq_Recomputed",
        ],
    )
    return {
        "M0_weather_time_onset_only": weather_time,
        "M1_plus_onset_setback": weather_time + onset_setback,
        "M2_plus_building_baseline": weather_time + onset_setback + building_baseline,
        "M3_full_plus_history": weather_time + onset_setback + building_baseline + history,
        "history_only": history,
        "no_history": weather_time + onset_setback + building_baseline,
        "full_no_setback": weather_time + building_baseline + history,
    }


def make_preprocessor(train: pd.DataFrame, feats: list[str]) -> ColumnTransformer:
    cat_cols = [
        c
        for c in feats
        if train[c].dtype == object or str(train[c].dtype).startswith("category") or train[c].dtype == bool
    ]
    num_cols = [c for c in feats if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def make_classifier() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
    )


def make_regressor() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
    )


def fit_pipe(train: pd.DataFrame, feats: list[str], kind: str = "clf") -> Pipeline:
    model = make_classifier() if kind == "clf" else make_regressor()
    return Pipeline([("pre", make_preprocessor(train, feats)), ("model", model)])


def groupkfold_predict(df: pd.DataFrame, feats: list[str], name: str) -> tuple[pd.DataFrame, Metrics]:
    valid = df[["Identifier", "Y"] + feats].dropna(subset=["Identifier", "Y"]).copy()
    y = valid["Y"].astype(int).values
    groups = valid["Identifier"].values
    p = np.full(len(valid), np.nan)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for k, (tr_idx, te_idx) in enumerate(gkf.split(valid, y, groups), start=1):
        tr, te = valid.iloc[tr_idx], valid.iloc[te_idx]
        pipe = fit_pipe(tr, feats, "clf")
        pipe.fit(tr[feats], tr["Y"].astype(int).values)
        p[te_idx] = pipe.predict_proba(te[feats])[:, 1]
        log(f"  {name} fold {k}: train users={tr['Identifier'].nunique():,}, test users={te['Identifier'].nunique():,}")
    pred = valid[["Identifier", "Y"]].copy()
    pred["pred_optout"] = p
    return pred, metric_row(name, y, p, groups, len(feats))


def time_split_predict(df: pd.DataFrame, feats: list[str], name: str) -> tuple[pd.DataFrame, Metrics, pd.DataFrame, pd.DataFrame]:
    valid = df[["Identifier", "Y", "Session_Start"] + feats].dropna(subset=["Identifier", "Y", "Session_Start"]).copy()
    valid = valid.sort_values("Session_Start").reset_index(drop=True)
    cut = int(math.floor(len(valid) * 0.70))
    train, test = valid.iloc[:cut].copy(), valid.iloc[cut:].copy()
    pipe = fit_pipe(train, feats, "clf")
    pipe.fit(train[feats], train["Y"].astype(int).values)
    p = pipe.predict_proba(test[feats])[:, 1]
    pred = test[["Identifier", "Y", "Session_Start"]].copy()
    pred["pred_optout"] = p
    met = metric_row(name, test["Y"].astype(int).values, p, test["Identifier"].values, len(feats))
    log(
        f"  time split {name}: train n={len(train):,}, test n={len(test):,}, "
        f"cutoff={test['Session_Start'].min()}"
    )
    return pred, met, train, test


def reliability_deciles(df: pd.DataFrame, pred: pd.DataFrame, label: str) -> pd.DataFrame:
    d = pred.copy()
    d["pred_accept"] = 1 - d["pred_optout"]
    d["observed_accept"] = 1 - d["Y"].astype(int)
    d["decile"] = pd.qcut(d["pred_accept"], 10, labels=False, duplicates="drop") + 1
    out = d.groupby("decile").agg(
        n=("Y", "count"),
        pred_accept=("pred_accept", "mean"),
        observed_accept=("observed_accept", "mean"),
        optout=("Y", "mean"),
    ).reset_index()
    out.to_csv(OUT / f"{label}_reliability_deciles.csv", index=False)
    return out


def bootstrap_mean_errors(d: pd.DataFrame, value_cols: list[str], truth_col: str, n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    cols = [truth_col] + value_cols
    agg = d.groupby("Identifier", sort=False)[cols].agg(["sum", "count"])
    users = agg.index.to_numpy()
    truth_sum = agg[(truth_col, "sum")].to_numpy(float)
    truth_count = agg[(truth_col, "count")].to_numpy(float)
    value_sums = {c: agg[(c, "sum")].to_numpy(float) for c in value_cols}
    value_counts = {c: agg[(c, "count")].to_numpy(float) for c in value_cols}
    rows = []
    for b in range(n_boot):
        draw = rng.integers(0, len(users), size=len(users))
        truth = truth_sum[draw].sum() / truth_count[draw].sum()
        row = {"boot": b}
        for c in value_cols:
            mean = value_sums[c][draw].sum() / value_counts[c][draw].sum()
            row[f"{c}_pct_err"] = 100 * (mean - truth) / truth
        rows.append(row)
    boot = pd.DataFrame(rows)
    summ = []
    for c in value_cols:
        vals = boot[f"{c}_pct_err"].dropna()
        summ.append(
            {
                "method": c,
                "err_ci_lo": float(vals.quantile(0.025)),
                "err_ci_hi": float(vals.quantile(0.975)),
            }
        )
    return pd.DataFrame(summ)


def accounting_benchmark(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    log("\n[Accounting] Out-of-time delivered-flexibility benchmark")
    base_cols = ["Identifier", "Y", "Session_Start", "Cool_Reduction_Frac", "Delivered_Completion"] + feats
    valid = df[base_cols].dropna(subset=["Identifier", "Y", "Session_Start", "Cool_Reduction_Frac"]).copy()
    valid = valid.sort_values("Session_Start").reset_index(drop=True)
    cut = int(math.floor(len(valid) * 0.70))
    train, test = valid.iloc[:cut].copy(), valid.iloc[cut:].copy()

    clf = fit_pipe(train, feats, "clf")
    clf.fit(train[feats], train["Y"].astype(int).values)
    p_acc = 1 - clf.predict_proba(test[feats])[:, 1]

    train_red = train[train["Y"].astype(int) == 0].dropna(subset=["Cool_Reduction_Frac"]).copy()
    reg = fit_pipe(train_red, feats, "reg")
    reg.fit(train_red[feats], train_red["Cool_Reduction_Frac"].astype(float).values)
    pred_red = np.clip(reg.predict(test[feats]), -1, 1)

    test = test.copy()
    test["observed_completion"] = test["Delivered_Completion"]
    test["acceptance_naive_observed_reduction"] = test["Cool_Reduction_Frac"]
    test["behavior_aware_accept_only"] = p_acc * test["Cool_Reduction_Frac"].astype(float).values
    test["behavior_aware_full_pre_event"] = p_acc * pred_red

    methods = [
        "acceptance_naive_observed_reduction",
        "behavior_aware_accept_only",
        "behavior_aware_full_pre_event",
        "observed_completion",
    ]
    truth = test["observed_completion"].mean()
    rows = []
    for m in methods:
        mean = test[m].mean()
        rows.append(
            {
                "method": m,
                "mean_per_session": float(mean),
                "pct_err_vs_observed_completion": float(100 * (mean - truth) / truth),
                "n": int(len(test)),
                "users": int(test["Identifier"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    ci = bootstrap_mean_errors(
        test,
        [
            "acceptance_naive_observed_reduction",
            "behavior_aware_accept_only",
            "behavior_aware_full_pre_event",
        ],
        "observed_completion",
    )
    out = out.merge(ci, how="left", left_on="method", right_on="method")
    out.to_csv(OUT / "accounting_benchmark_leakage_free.csv", index=False)
    test[["Identifier", "Y", "Session_Start"] + methods].to_csv(OUT / "accounting_time_test_predictions.csv", index=False)
    log(out.to_string(index=False))
    return out


def same_setback_heterogeneity(df: pd.DataFrame, bin_col: str, label: str) -> pd.DataFrame:
    log(f"\n[Same setback] Heterogeneity for {label}")
    sub = df[df[bin_col].astype(str).eq("3-4")].dropna(subset=["Cool_Reduction_Frac"]).copy()
    prev = sub["Prev_OptOut_Recomputed"].fillna(0)
    prior = sub["Prior_OptOut_Rate_Recomputed"]
    sub["risk_group"] = "medium_risk"
    sub.loc[(prev == 0) & (prior <= 0.10), "risk_group"] = "low_risk"
    sub.loc[(prev == 1) | (prior > 0.40), "risk_group"] = "high_risk"
    order = ["low_risk", "medium_risk", "high_risk"]
    rows = []
    for g in order:
        s = sub[sub["risk_group"] == g]
        if len(s) == 0:
            continue
        rows.append(
            {
                "group": g,
                "n": int(len(s)),
                "users": int(s["Identifier"].nunique()),
                "completion_delivered": float(s["Delivered_Completion"].mean()),
                "baseline_relative_reduction": float(s["Cool_Reduction_Frac"].mean()),
                "optout": float(s["Y"].mean()),
            }
        )
    out = pd.DataFrame(rows)

    rng = np.random.default_rng(RANDOM_STATE)
    user_group = sub.groupby(["Identifier", "risk_group"], sort=False).agg(
        delivered_sum=("Delivered_Completion", "sum"),
        reduction_sum=("Cool_Reduction_Frac", "sum"),
        y_sum=("Y", "sum"),
        n=("Y", "count"),
    ).reset_index()
    users = sub["Identifier"].dropna().unique()
    user_pos = {u: i for i, u in enumerate(users)}
    buckets = {}
    for g in order:
        ug = user_group[user_group["risk_group"] == g]
        idx = np.array([user_pos[u] for u in ug["Identifier"].values], dtype=int)
        arr = np.zeros((len(users), 4), dtype=float)
        arr[idx, 0] = ug["delivered_sum"].to_numpy(float)
        arr[idx, 1] = ug["reduction_sum"].to_numpy(float)
        arr[idx, 2] = ug["y_sum"].to_numpy(float)
        arr[idx, 3] = ug["n"].to_numpy(float)
        buckets[g] = arr

    def boot_mean(draw: np.ndarray, group: str, col: int) -> float:
        arr = buckets[group]
        denom = arr[draw, 3].sum()
        if denom <= 0:
            return np.nan
        return float(arr[draw, col].sum() / denom)

    boot = []
    for b in range(1000):
        draw = rng.integers(0, len(users), size=len(users))
        row = {"boot": b}
        low_del = boot_mean(draw, "low_risk", 0)
        med_del = boot_mean(draw, "medium_risk", 0)
        high_del = boot_mean(draw, "high_risk", 0)
        low_red = boot_mean(draw, "low_risk", 1)
        med_red = boot_mean(draw, "medium_risk", 1)
        high_red = boot_mean(draw, "high_risk", 1)
        low_y = boot_mean(draw, "low_risk", 2)
        high_y = boot_mean(draw, "high_risk", 2)
        row["delivered_ratio_low_high"] = low_del / high_del
        row["delivered_diff_low_high"] = low_del - high_del
        row["reduction_diff_low_high"] = low_red - high_red
        row["optout_diff_high_low"] = high_y - low_y
        row["reduction_diff_low_medium"] = low_red - med_red
        row["reduction_diff_medium_high"] = med_red - high_red
        boot.append(row)
    boot = pd.DataFrame(boot)
    for c in boot.columns:
        if c == "boot":
            continue
        out.loc[:, f"{c}_ci_lo"] = float(boot[c].quantile(0.025))
        out.loc[:, f"{c}_ci_hi"] = float(boot[c].quantile(0.975))

    out.to_csv(OUT / f"same_3_4_{label}.csv", index=False)
    boot.to_csv(OUT / f"same_3_4_{label}_bootstrap.csv", index=False)
    log(out.to_string(index=False))
    return out


def adjusted_setback_model(df: pd.DataFrame) -> None:
    if not HAS_SM:
        log("\n[Adjusted setback] statsmodels unavailable, skipped")
        return
    log("\n[Adjusted setback] Logistic model without duration/CDH/building controls")
    d = df.dropna(subset=["Y", "Setback_Mean_Bin", "Tout_onset", "Identifier"]).copy()
    d = d[d["Setback_Mean_Bin"].astype(str).isin(SETBACK_LABELS)].copy()
    controls = ["C(Setback_Mean_Bin, Treatment(reference='2-3'))", "Tout_onset"]
    for c in ["RH_onset", "GHI_onset", "Hour_Bin", "Month", "DR_Type", "province_state"]:
        if c in d.columns:
            controls.append(f"C({c})" if d[c].dtype == object or c in ["Hour_Bin", "DR_Type", "province_state"] else c)
    fml = "Y ~ " + " + ".join(controls)
    try:
        model = smf.glm(fml, data=d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d["Identifier"]}
        )
        rows = []
        raw = d.groupby("Setback_Mean_Bin", observed=True)["Y"].mean()
        for b in SETBACK_LABELS:
            tmp = d.copy()
            tmp["Setback_Mean_Bin"] = b
            rows.append(
                {
                    "Setback_Bin": b,
                    "n": int((d["Setback_Mean_Bin"].astype(str) == b).sum()),
                    "raw_optout": float(raw.loc[b]) if b in raw.index else np.nan,
                    "adjusted_optout": float(model.predict(tmp).mean()),
                }
            )
        pred = pd.DataFrame(rows)
        pred.to_csv(OUT / "adjusted_setback_no_duration_cdh.csv", index=False)
        coef = pd.DataFrame(
            {
                "term": model.params.index,
                "coef": model.params.values,
                "se": model.bse.values,
                "pvalue": model.pvalues.values,
            }
        )
        coef.to_csv(OUT / "adjusted_setback_no_duration_cdh_coefficients.csv", index=False)
        log(pred.to_string(index=False))
    except Exception as exc:
        log(f"  adjusted setback model failed: {type(exc).__name__}: {exc}")


def main() -> None:
    df = load_data()
    fs = feature_sets(df)
    (OUT / "feature_sets.json").write_text(json.dumps(fs, indent=2), encoding="utf-8")
    log("\nFeature sets:")
    for name, feats in fs.items():
        log(f"  {name}: {len(feats)} features -> {feats}")

    log("\n[GroupKFold] Acceptance model rerun")
    metrics: list[Metrics] = []
    full_pred = None
    for name in [
        "M0_weather_time_onset_only",
        "M1_plus_onset_setback",
        "M2_plus_building_baseline",
        "M3_full_plus_history",
        "history_only",
        "no_history",
        "full_no_setback",
    ]:
        pred, met = groupkfold_predict(df, fs[name], name)
        pred.to_csv(OUT / f"groupkfold_predictions_{name}.csv", index=False)
        metrics.append(met)
        log(f"  {name}: AUC={met.auc:.4f}, PR-AUC={met.pr_auc_optout:.4f}, Brier={met.brier:.4f}, ECE={met.ece_optout:.4f}")
        if name == "M3_full_plus_history":
            full_pred = pred
            reliability_deciles(df, pred, "groupkfold_full")

    metrics_df = pd.DataFrame([asdict(m) for m in metrics])
    metrics_df.to_csv(OUT / "groupkfold_metrics_leakage_free.csv", index=False)

    log("\n[Out-of-time] Full model rerun")
    time_pred, time_met, _, _ = time_split_predict(df, fs["M3_full_plus_history"], "M3_full_plus_history_time")
    time_pred.to_csv(OUT / "out_of_time_predictions_full.csv", index=False)
    pd.DataFrame([asdict(time_met)]).to_csv(OUT / "out_of_time_metrics_leakage_free.csv", index=False)
    reliability_deciles(df, time_pred, "out_of_time_full")
    log(
        f"  out-of-time full: AUC={time_met.auc:.4f}, PR-AUC={time_met.pr_auc_optout:.4f}, "
        f"Brier={time_met.brier:.4f}, ECE={time_met.ece_optout:.4f}"
    )

    accounting_benchmark(df, fs["M3_full_plus_history"])
    same_setback_heterogeneity(df, "Setback_Mean_Bin", "observed_mean_setback_bin")
    same_setback_heterogeneity(df, "Setback_Onset_Bin", "onset_setback_bin")
    adjusted_setback_model(df)

    (OUT / "rerun_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    log("\nDone. Outputs written to leakage_free_out/")


if __name__ == "__main__":
    main()
