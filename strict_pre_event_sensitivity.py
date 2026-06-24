#!/usr/bin/env python3
"""Strict pre-event sensitivity checks for the ecobee DR paper.

Outputs are written to strict_pre_event_out/. This script is intentionally
separate from the main analysis and from leakage_free_out/.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

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
from lightgbm import LGBMClassifier, LGBMRegressor

DATA = Path("dr_sessions.csv")
OUT = Path("strict_pre_event_out")
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
SETBACK_BINS = [0, 1, 2, 3, 4, 6]
SETBACK_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-6"]
REPORT: list[str] = []


@dataclass
class Metrics:
    cohort: str
    model: str
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


def log(s: str = "") -> None:
    print(s, flush=True)
    REPORT.append(str(s))


def as_bool_int(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"]).astype(int)


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    out = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if mask.any():
            out += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(out)


def cal_int_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(lp, y.astype(int))
        return float(lr.intercept_[0]), float(lr.coef_[0][0])
    except Exception:
        return float("nan"), float("nan")


def metric(cohort: str, model: str, y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_features: int) -> Metrics:
    ci, cs = cal_int_slope(y, p)
    return Metrics(
        cohort=cohort,
        model=model,
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


def load_base() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df["HvacMode"].astype(str).str.lower().eq("cool")].copy()
    if "N_DR_Rows" in df.columns:
        df = df[pd.to_numeric(df["N_DR_Rows"], errors="coerce").fillna(0) >= 3].copy()
    df["Session_Start"] = pd.to_datetime(df["Session_Start"], errors="coerce")
    df = df.sort_values(["Identifier", "Session_Start"]).reset_index(drop=True)
    df["Y"] = as_bool_int(df["OptOut_Immediate"] if "OptOut_Immediate" in df.columns else df["Opted_Out"])
    df["Cool_Reduction_Frac"] = pd.to_numeric(df["Cool_Reduction_Frac"], errors="coerce")
    df["Delivered_Completion"] = df["Cool_Reduction_Frac"] * (1 - df["Y"])
    df["Setback_Baseline_Onset"] = pd.to_numeric(df["Setpoint_Cool_Start"], errors="coerce") - pd.to_numeric(
        df["Baseline_Setpoint_Cool"], errors="coerce"
    )
    df["Setback_Baseline_Onset_sq"] = df["Setback_Baseline_Onset"] ** 2
    bin_ser = pd.cut(df["Setback_Baseline_Onset"], SETBACK_BINS, labels=SETBACK_LABELS, include_lowest=True)
    df["Setback_Baseline_Onset_Bin"] = bin_ser.astype(object).where(bin_ser.notna(), "outside_0_6")

    df["Session_Seq_Recomputed"] = df.groupby("Identifier").cumcount() + 1
    df["Prev_OptOut_Recomputed"] = df.groupby("Identifier")["Y"].shift(1)
    cum = df.groupby("Identifier")["Y"].cumsum()
    n_seen = df.groupby("Identifier").cumcount() + 1
    df["Prior_OptOut_Rate_Recomputed"] = np.where(n_seen > 1, (cum - df["Y"]) / (n_seen - 1), np.nan)
    streaks: list[int] = []
    for _, g in df.groupby("Identifier", sort=False):
        st = 0
        for y in g["Y"].astype(int).values:
            streaks.append(st)
            st = st + 1 if y else 0
    df["OptOut_Streak_Recomputed"] = streaks
    return df


def present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    weather_time = present(
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
    setback = ["Setback_Baseline_Onset", "Setback_Baseline_Onset_sq", "Setback_Baseline_Onset_Bin"]
    building_baseline = present(
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
    history = [
        "Prev_OptOut_Recomputed",
        "Prior_OptOut_Rate_Recomputed",
        "OptOut_Streak_Recomputed",
        "Session_Seq_Recomputed",
    ]
    return {
        "weather_time": weather_time,
        "plus_baseline_onset_setback": weather_time + setback,
        "no_history": weather_time + setback + building_baseline,
        "full": weather_time + setback + building_baseline + history,
        "history_only": history,
        "full_no_setback": weather_time + building_baseline + history,
    }


def preprocessor(train: pd.DataFrame, feats: list[str]) -> ColumnTransformer:
    cats = [c for c in feats if train[c].dtype == object or str(train[c].dtype).startswith("category") or train[c].dtype == bool]
    nums = [c for c in feats if c not in cats]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), nums),
            (
                "cat",
                Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                cats,
            ),
        ]
    )


def clf_pipe(train: pd.DataFrame, feats: list[str]) -> Pipeline:
    clf = LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    return Pipeline([("pre", preprocessor(train, feats)), ("clf", clf)])


def reg_pipe(train: pd.DataFrame, feats: list[str]) -> Pipeline:
    reg = LGBMRegressor(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    return Pipeline([("pre", preprocessor(train, feats)), ("reg", reg)])


def groupkfold(df: pd.DataFrame, feats: list[str], cohort: str, model: str) -> Metrics:
    valid = df[["Identifier", "Y"] + feats].dropna(subset=["Identifier", "Y"]).copy()
    y = valid["Y"].astype(int).values
    groups = valid["Identifier"].values
    pred = np.full(len(valid), np.nan)
    for tr_idx, te_idx in GroupKFold(n_splits=N_SPLITS).split(valid, y, groups):
        tr, te = valid.iloc[tr_idx], valid.iloc[te_idx]
        pipe = clf_pipe(tr, feats)
        pipe.fit(tr[feats], tr["Y"].astype(int).values)
        pred[te_idx] = pipe.predict_proba(te[feats])[:, 1]
    return metric(cohort, model, y, pred, groups, len(feats))


def time_split(df: pd.DataFrame, feats: list[str], cohort: str, model: str) -> Metrics:
    valid = df[["Identifier", "Y", "Session_Start"] + feats].dropna(subset=["Identifier", "Y", "Session_Start"]).copy()
    valid = valid.sort_values("Session_Start").reset_index(drop=True)
    cut = int(math.floor(len(valid) * 0.7))
    tr, te = valid.iloc[:cut].copy(), valid.iloc[cut:].copy()
    pipe = clf_pipe(tr, feats)
    pipe.fit(tr[feats], tr["Y"].astype(int).values)
    pred = pipe.predict_proba(te[feats])[:, 1]
    m = metric(cohort, model + "_time", te["Y"].astype(int).values, pred, te["Identifier"].values, len(feats))
    m.n = int(len(te))
    m.users = int(te["Identifier"].nunique())
    return m


def accounting(df: pd.DataFrame, feats: list[str], cohort: str) -> pd.DataFrame:
    valid = df[["Identifier", "Y", "Session_Start", "Cool_Reduction_Frac", "Delivered_Completion"] + feats].dropna(
        subset=["Identifier", "Y", "Session_Start", "Cool_Reduction_Frac"]
    ).copy()
    valid = valid.sort_values("Session_Start").reset_index(drop=True)
    cut = int(math.floor(len(valid) * 0.7))
    tr, te = valid.iloc[:cut].copy(), valid.iloc[cut:].copy()
    clf = clf_pipe(tr, feats)
    clf.fit(tr[feats], tr["Y"].astype(int).values)
    p_acc = 1 - clf.predict_proba(te[feats])[:, 1]
    tr_red = tr[tr["Y"].astype(int).eq(0)].copy()
    reg = reg_pipe(tr_red, feats)
    reg.fit(tr_red[feats], tr_red["Cool_Reduction_Frac"].astype(float).values)
    pred_red = np.clip(reg.predict(te[feats]), -1, 1)
    truth = te["Delivered_Completion"].mean()
    rows = []
    vals = {
        "acceptance_naive_observed_reduction": te["Cool_Reduction_Frac"].mean(),
        "behavior_aware_accept_only": np.mean(p_acc * te["Cool_Reduction_Frac"].astype(float).values),
        "behavior_aware_full_pre_event": np.mean(p_acc * pred_red),
        "observed_completion": truth,
    }
    for name, val in vals.items():
        rows.append({"cohort": cohort, "method": name, "mean": float(val), "pct_error": float(100 * (val - truth) / truth), "n": len(te), "users": te["Identifier"].nunique()})
    return pd.DataFrame(rows)


def run_cohort(df: pd.DataFrame, name: str, mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[mask].copy()
    fs = feature_sets(d)
    log(f"\n{name}: n={len(d):,}, users={d['Identifier'].nunique():,}, optout={d['Y'].mean():.4f}")
    metrics = []
    for model in ["weather_time", "plus_baseline_onset_setback", "no_history", "full", "history_only", "full_no_setback"]:
        m = groupkfold(d, fs[model], name, model)
        metrics.append(m)
        log(f"  GKF {model}: AUC={m.auc:.4f}, PR={m.pr_auc_optout:.4f}, Brier={m.brier:.4f}, ECE={m.ece_optout:.4f}")
    tm = time_split(d, fs["full"], name, "full")
    metrics.append(tm)
    log(f"  TIME full: AUC={tm.auc:.4f}, PR={tm.pr_auc_optout:.4f}, Brier={tm.brier:.4f}, ECE={tm.ece_optout:.4f}")
    acc = accounting(d, fs["full"], name)
    log(acc.to_string(index=False))
    return pd.DataFrame([asdict(m) for m in metrics]), acc


def main() -> None:
    df = load_base()
    log("=" * 78)
    log("Strict pre-event sensitivity")
    log("=" * 78)
    cohorts = {
        "baseline_onset_setback_0_6": df["Setback_Baseline_Onset"].between(0, 6),
        "all_cooling_no_setback_filter": pd.Series(True, index=df.index),
    }
    all_metrics = []
    all_acc = []
    for name, mask in cohorts.items():
        m, a = run_cohort(df, name, mask)
        all_metrics.append(m)
        all_acc.append(a)
    pd.concat(all_metrics, ignore_index=True).to_csv(OUT / "strict_metrics.csv", index=False)
    pd.concat(all_acc, ignore_index=True).to_csv(OUT / "strict_accounting.csv", index=False)
    (OUT / "strict_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
