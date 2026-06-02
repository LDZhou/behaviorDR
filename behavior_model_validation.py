"""
Remaining validation experiments for the behavior-aware DR reliability project.

Run from the ecobee project directory that contains dr_sessions.csv:

    python 13_remaining_validation_experiments.py

Outputs:
    remaining_experiments_out/
    remaining_experiments_out/figs/

What this script does:
1. User-level GroupKFold reliability-score validation
2. Reliability decile analysis
3. Calibration check
4. Out-of-time validation
5. Same-setback reliability heterogeneity, especially 3-4°F
6. Behavior-adjusted delivered-flexibility diagnostics by reliability decile

This script is deliberately conservative:
- It uses only event-start / pre-event / historical features for the acceptance model.
- It recomputes user history features chronologically to avoid future leakage.
- It evaluates prediction with user-level GroupKFold and out-of-time split.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGBM = False


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_PATH = Path("dr_sessions.csv")
OUT = Path("remaining_experiments_out")
FIG = OUT / "figs"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

SETBACK_BINS = [0, 1, 2, 3, 4, 6]
SETBACK_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-6"]

# Keep the feature groups explicit so they can be reported in the paper/slides.
WEATHER_TIME_CANDIDATES = [
    "Tout_onset", "RH_onset", "GHI_onset", "dew_onset", "CDH_during",
    "temperature_2m", "relative_humidity_2m", "shortwave_radiation", "dew_point_2m",
    "Duration_Min", "Hour_of_Day", "Is_Weekend", "Month", "Hour_Bin",
    "DR_Type", "province_state", "country",
]

SETBACK_CANDIDATES = [
    "Setback", "Setback_sq", "Setback_Bin",
]

BUILDING_BASELINE_CANDIDATES = [
    "floor_area_sqft", "building_age_yrs", "number_occupants", "has_heatpump",
    "has_electric", "number_cool_stages", "number_heat_stages", "building_type",
    "Baseline_Cool_Frac", "Avg_Baseline_Temp", "Setpoint_Cool_Start",
    "Avg_Baseline_Cool", "weather_is_fallback",
]

USER_HISTORY_CANDIDATES = [
    "Prev_OptOut_Recomputed", "Prior_OptOut_Rate_Recomputed",
    "OptOut_Streak_Recomputed", "Session_Seq_Recomputed",
]


@dataclass
class ModelResult:
    name: str
    n: int
    users: int
    auc: float
    pr_auc: float
    brier: float
    logloss: float
    ece: float
    cal_intercept: float
    cal_slope: float


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def log(msg: str, lines: List[str]) -> None:
    print(msg, flush=True)
    lines.append(str(msg))


def detect_outcome(df: pd.DataFrame) -> str:
    for col in ["OptOut_Immediate", "Opted_Out", "OptOut", "Y_immediate"]:
        if col in df.columns:
            return col
    raise ValueError("No opt-out outcome column found. Expected one of: OptOut_Immediate, Opted_Out, OptOut, Y_immediate")


def detect_setback(df: pd.DataFrame) -> str:
    for col in ["Setback_Amplitude_Mean", "Setback", "Setback_Amplitude"]:
        if col in df.columns:
            return col
    raise ValueError("No setback column found. Expected Setback_Amplitude_Mean or Setback")


def detect_time(df: pd.DataFrame) -> Optional[str]:
    for col in ["Session_Start", "session_start", "DR_Start", "date_time"]:
        if col in df.columns:
            return col
    return None


def safe_bool_to_num(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    if s.dtype == object:
        lower = s.astype(str).str.lower()
        if lower.isin(["true", "false", "nan", "none", "", "0", "1"]).mean() > 0.8:
            return lower.map({"true": 1, "false": 0, "1": 1, "0": 0}).astype(float)
    return pd.to_numeric(s, errors="coerce")


def recompute_history(df: pd.DataFrame, y_col: str, time_col: Optional[str]) -> pd.DataFrame:
    """Chronologically recompute history features using only prior events."""
    out = df.copy()
    if time_col is not None:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out = out.sort_values(["Identifier", time_col]).reset_index(drop=True)
    else:
        out = out.sort_values(["Identifier"]).reset_index(drop=True)

    y = out[y_col].astype(int)
    out["Session_Seq_Recomputed"] = out.groupby("Identifier").cumcount() + 1
    out["Prev_OptOut_Recomputed"] = out.groupby("Identifier")[y_col].shift(1)

    cum_oo = out.groupby("Identifier")[y_col].cumsum()
    n_seen = out.groupby("Identifier").cumcount() + 1
    prior_oo = cum_oo - y
    prior_n = n_seen - 1
    pop_rate = float(y.mean())
    out["Prior_OptOut_Rate_Recomputed"] = np.where(prior_n > 0, prior_oo / prior_n, pop_rate)

    def streak_prior(vals: pd.Series) -> pd.Series:
        vals = vals.astype(int).values
        res = np.zeros(len(vals), dtype=int)
        current = 0
        for j, v in enumerate(vals):
            res[j] = current
            if v == 1:
                current += 1
            else:
                current = 0
        return pd.Series(res, index=vals.index if hasattr(vals, "index") else None)

    # groupby/apply that preserves order
    streaks = []
    for _, g in out.groupby("Identifier", sort=False):
        current = 0
        vals = []
        for v in g[y_col].astype(int).values:
            vals.append(current)
            if v == 1:
                current += 1
            else:
                current = 0
        streaks.extend(vals)
    out["OptOut_Streak_Recomputed"] = streaks

    return out


def make_clean_sample(df: pd.DataFrame, lines: List[str]) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    y_col = detect_outcome(df)
    setback_col = detect_setback(df)
    time_col = detect_time(df)

    sdf = df.copy()

    # Restrict to cooling if possible.
    if "HvacMode" in sdf.columns:
        sdf = sdf[sdf["HvacMode"].astype(str).str.lower().eq("cool")].copy()

    # Basic filters consistent with earlier analysis.
    sdf[setback_col] = pd.to_numeric(sdf[setback_col], errors="coerce")
    sdf = sdf[sdf[setback_col].between(0, 6)].copy()
    if "N_DR_Rows" in sdf.columns:
        sdf = sdf[pd.to_numeric(sdf["N_DR_Rows"], errors="coerce").fillna(0) >= 3].copy()

    sdf[y_col] = pd.to_numeric(sdf[y_col], errors="coerce").fillna(0).astype(int)
    sdf["Setback"] = sdf[setback_col]
    sdf["Setback_sq"] = sdf["Setback"] ** 2
    sdf["Setback_Bin"] = pd.cut(
        sdf["Setback"], bins=SETBACK_BINS, labels=SETBACK_LABELS,
        include_lowest=True, right=True,
    ).astype(str)

    # Delivered-flexibility target if possible.
    if "Cool_Reduction_Frac" in sdf.columns:
        sdf["Cool_Reduction_Frac"] = pd.to_numeric(sdf["Cool_Reduction_Frac"], errors="coerce")
        sdf["Delivered_Flex_Observed"] = sdf["Cool_Reduction_Frac"] * (1 - sdf[y_col].astype(int))

    sdf = recompute_history(sdf, y_col, time_col)

    log(f"Loaded sample: {len(sdf):,} sessions, {sdf['Identifier'].nunique():,} users", lines)
    log(f"Outcome: {y_col}; Setback source: {setback_col}; Time column: {time_col}", lines)
    log(f"Opt-out rate: {sdf[y_col].mean():.2%}", lines)

    return sdf, y_col, setback_col, time_col


def available_features(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def split_feature_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in features:
        if c not in df.columns:
            continue
        if c in ["Setback_Bin", "Hour_Bin", "DR_Type", "province_state", "country", "building_type"]:
            cat_cols.append(c)
        elif df[c].dtype == object or str(df[c].dtype).startswith("category"):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return num_cols, cat_cols


def prepare_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_cols, cat_cols = split_feature_types(train, features)

    Xtr_parts = []
    Xte_parts = []

    if num_cols:
        tr_num = train[num_cols].copy()
        te_num = test[num_cols].copy()
        for c in num_cols:
            tr_num[c] = safe_bool_to_num(tr_num[c])
            te_num[c] = safe_bool_to_num(te_num[c])
            med = tr_num[c].median()
            if not np.isfinite(med):
                med = 0.0
            tr_num[c] = tr_num[c].fillna(med)
            te_num[c] = te_num[c].fillna(med)
        Xtr_parts.append(tr_num.astype(float))
        Xte_parts.append(te_num.astype(float))

    if cat_cols:
        tr_cat = train[cat_cols].astype(str).fillna("Missing")
        te_cat = test[cat_cols].astype(str).fillna("Missing")
        tr_d = pd.get_dummies(tr_cat, columns=cat_cols, dummy_na=False)
        te_d = pd.get_dummies(te_cat, columns=cat_cols, dummy_na=False)
        tr_d, te_d = tr_d.align(te_d, join="left", axis=1, fill_value=0)
        Xtr_parts.append(tr_d.astype(float))
        Xte_parts.append(te_d.astype(float))

    if not Xtr_parts:
        raise ValueError("No features available after preprocessing")

    Xtr = pd.concat(Xtr_parts, axis=1)
    Xte = pd.concat(Xte_parts, axis=1)
    return Xtr, Xte


def make_model():
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=80,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    ece = 0.0
    n = len(df)
    for _, g in df.groupby("bin"):
        ece += len(g) / n * abs(g["y"].mean() - g["p"].mean())
    return float(ece)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    eps = 1e-5
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def evaluate_predictions(name: str, y: np.ndarray, p: np.ndarray, users: pd.Series) -> ModelResult:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    pr = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan
    brier = brier_score_loss(y, p)
    ll = log_loss(y, p, labels=[0, 1])
    ece = expected_calibration_error(y, p)
    ci, cs = calibration_slope_intercept(y, p)
    return ModelResult(
        name=name, n=len(y), users=int(pd.Series(users).nunique()),
        auc=float(auc), pr_auc=float(pr), brier=float(brier), logloss=float(ll),
        ece=float(ece), cal_intercept=ci, cal_slope=cs,
    )


def decile_table(y: np.ndarray, p_optout: np.ndarray, prefix: str) -> pd.DataFrame:
    p_accept = 1 - p_optout
    df = pd.DataFrame({"y": y, "pred_optout": p_optout, "pred_accept": p_accept})
    # Decile 1 = lowest predicted reliability; Decile 10 = highest predicted reliability.
    df["reliability_decile"] = pd.qcut(df["pred_accept"].rank(method="first"), 10, labels=False) + 1
    tab = df.groupby("reliability_decile").agg(
        n=("y", "size"),
        pred_accept_mean=("pred_accept", "mean"),
        observed_accept_rate=("y", lambda x: 1 - np.mean(x)),
        observed_optout_rate=("y", "mean"),
        pred_optout_mean=("pred_optout", "mean"),
    ).reset_index()
    tab.to_csv(OUT / f"{prefix}_reliability_deciles.csv", index=False)
    return tab


def calibration_table(y: np.ndarray, p_optout: np.ndarray, prefix: str) -> pd.DataFrame:
    p_accept = 1 - p_optout
    df = pd.DataFrame({"y": y, "pred_accept": p_accept})
    df["calibration_bin"] = pd.qcut(df["pred_accept"].rank(method="first"), 10, labels=False) + 1
    tab = df.groupby("calibration_bin").agg(
        n=("y", "size"),
        pred_accept_mean=("pred_accept", "mean"),
        observed_accept_rate=("y", lambda x: 1 - np.mean(x)),
    ).reset_index()
    tab.to_csv(OUT / f"{prefix}_calibration_table.csv", index=False)
    return tab


def plot_decile(tab: pd.DataFrame, prefix: str, title: str) -> None:
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(tab["reliability_decile"], tab["observed_accept_rate"] * 100, marker="o", label="Observed acceptance")
    plt.plot(tab["reliability_decile"], tab["pred_accept_mean"] * 100, marker="s", label="Predicted acceptance")
    plt.xlabel("Reliability score decile")
    plt.ylabel("Acceptance rate (%)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"{prefix}_reliability_deciles.png", dpi=200)
    plt.close()


def plot_calibration(tab: pd.DataFrame, prefix: str, title: str) -> None:
    plt.figure(figsize=(5.2, 5.0))
    plt.plot([0, 100], [0, 100], linestyle="--", label="Perfect calibration")
    plt.plot(tab["pred_accept_mean"] * 100, tab["observed_accept_rate"] * 100, marker="o", label="Model")
    plt.xlabel("Predicted acceptance (%)")
    plt.ylabel("Observed acceptance (%)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"{prefix}_calibration.png", dpi=200)
    plt.close()


def fit_predict_groupkfold(
    df: pd.DataFrame,
    y_col: str,
    features: List[str],
    name: str,
    lines: List[str],
) -> Tuple[np.ndarray, ModelResult]:
    valid = df[["Identifier", y_col] + features].dropna(subset=["Identifier", y_col]).copy()
    # Require at least two observations per class.
    valid[y_col] = valid[y_col].astype(int)
    groups = valid["Identifier"].astype(str)
    y = valid[y_col].values

    oof = np.full(len(valid), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(valid, y, groups), start=1):
        train = valid.iloc[tr_idx].copy()
        test = valid.iloc[te_idx].copy()
        Xtr, Xte = prepare_train_test(train, test, features)
        model = make_model()
        model.fit(Xtr, train[y_col].astype(int).values)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(Xte)[:, 1]
        else:
            # HistGradientBoosting supports predict_proba, but keep safe.
            p = model.predict_proba(Xte)[:, 1]
        oof[te_idx] = p
        log(f"  {name} fold {fold}: train users={train['Identifier'].nunique():,}, test users={test['Identifier'].nunique():,}", lines)

    result = evaluate_predictions(name, y, oof, groups)
    pred_df = valid[["Identifier", y_col]].copy()
    pred_df["pred_optout"] = oof
    pred_df["pred_accept"] = 1 - oof
    pred_df.to_csv(OUT / f"{name}_groupkfold_predictions.csv", index=False)
    return oof, result


def fit_predict_time_split(
    df: pd.DataFrame,
    y_col: str,
    features: List[str],
    time_col: Optional[str],
    lines: List[str],
) -> Tuple[pd.DataFrame, ModelResult]:
    if time_col is None:
        raise ValueError("No time column available for out-of-time validation")

    valid = df[["Identifier", y_col, time_col] + features].dropna(subset=["Identifier", y_col, time_col]).copy()
    valid[time_col] = pd.to_datetime(valid[time_col], errors="coerce")
    valid = valid.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    valid[y_col] = valid[y_col].astype(int)

    cutoff = valid[time_col].quantile(0.70)
    train = valid[valid[time_col] <= cutoff].copy()
    test = valid[valid[time_col] > cutoff].copy()

    Xtr, Xte = prepare_train_test(train, test, features)
    model = make_model()
    model.fit(Xtr, train[y_col].values)
    p = model.predict_proba(Xte)[:, 1]

    result = evaluate_predictions("out_of_time", test[y_col].values, p, test["Identifier"])
    pred_df = test[["Identifier", y_col, time_col]].copy()
    pred_df["pred_optout"] = p
    pred_df["pred_accept"] = 1 - p
    pred_df.to_csv(OUT / "out_of_time_predictions.csv", index=False)

    log(f"Out-of-time split cutoff: {cutoff}", lines)
    log(f"  train n={len(train):,}, users={train['Identifier'].nunique():,}", lines)
    log(f"  test  n={len(test):,}, users={test['Identifier'].nunique():,}", lines)
    return pred_df, result


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------
def experiment_same_setback_heterogeneity(df: pd.DataFrame, y_col: str, lines: List[str]) -> None:
    """Directly show why a fixed 3-4°F rule is not enough."""
    sub = df[df["Setback_Bin"].eq("3-4")].copy()
    if len(sub) < 100:
        log("[Same setback heterogeneity] Too few 3-4°F sessions; skipping", lines)
        return

    # Simple risk groups known before current event.
    prior = sub["Prior_OptOut_Rate_Recomputed"].fillna(df[y_col].mean())
    prev = sub["Prev_OptOut_Recomputed"].fillna(0)

    conditions = [
        (prev.eq(0) & prior.le(0.10)),
        (prev.eq(0) & prior.gt(0.10) & prior.le(0.40)),
        (prev.eq(1) | prior.gt(0.40)),
    ]
    choices = ["low_risk", "medium_risk", "high_risk"]
    sub["History_Risk_Group"] = np.select(conditions, choices, default="medium_risk")

    agg_dict = {
        "n": (y_col, "size"),
        "users": ("Identifier", "nunique"),
        "optout_rate": (y_col, "mean"),
        "acceptance_rate": (y_col, lambda x: 1 - np.mean(x)),
        "prior_oo_rate_mean": ("Prior_OptOut_Rate_Recomputed", "mean"),
        "prev_oo_rate": ("Prev_OptOut_Recomputed", "mean"),
    }
    if "Cool_Reduction_Frac" in sub.columns:
        agg_dict["nominal_reduction"] = ("Cool_Reduction_Frac", "mean")
    if "Delivered_Flex_Observed" in sub.columns:
        agg_dict["delivered_flex"] = ("Delivered_Flex_Observed", "mean")

    tab = sub.groupby("History_Risk_Group").agg(**agg_dict).reset_index()
    order = ["low_risk", "medium_risk", "high_risk"]
    tab["order"] = tab["History_Risk_Group"].map({k: i for i, k in enumerate(order)})
    tab = tab.sort_values("order").drop(columns="order")
    tab.to_csv(OUT / "same_3_4_setback_by_history_risk.csv", index=False)

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(tab["History_Risk_Group"], tab["optout_rate"] * 100)
    plt.ylabel("Opt-out rate (%)")
    plt.xlabel("History risk group, only 3-4°F sessions")
    plt.title("Same setback, different user reliability")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig_same_3_4_setback_by_history_risk.png", dpi=200)
    plt.close()

    log("[Same setback heterogeneity] Saved same_3_4_setback_by_history_risk.csv", lines)


def experiment_delivered_by_decile(
    df: pd.DataFrame,
    y_col: str,
    pred_path: Path,
    lines: List[str],
) -> None:
    """Combine out-of-fold reliability score with observed nominal/delivered flex, if possible."""
    if not pred_path.exists() or "Cool_Reduction_Frac" not in df.columns:
        log("[Delivered by decile] Missing predictions or Cool_Reduction_Frac; skipping", lines)
        return

    pred = pd.read_csv(pred_path)
    # Need row alignment; groupkfold pred file currently lacks index. Reconstruct by merging is unsafe if duplicates.
    # Instead, this diagnostic is implemented in the main OOF dataframe later when available.
    log("[Delivered by decile] Use reliability_decile_behavior_accounting.csv generated from OOF dataframe", lines)


def behavior_accounting_by_reliability_decile(df_valid: pd.DataFrame, y_col: str, oof: np.ndarray, prefix: str) -> None:
    d = df_valid[["Identifier", y_col]].copy()
    d["pred_optout"] = oof
    d["pred_accept"] = 1 - oof

    # Attach optional columns by index alignment from valid dataframe.
    # Guard against duplicate column names: pandas returns a DataFrame if a name is duplicated.
    for c in ["Cool_Reduction_Frac", "Delivered_Flex_Observed", "Setback_Bin", "Setback"]:
        if c in df_valid.columns:
            col = df_valid.loc[:, c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            d[c] = col.to_numpy()

    d["reliability_decile"] = pd.qcut(d["pred_accept"].rank(method="first"), 10, labels=False) + 1
    agg = {
        "n": (y_col, "size"),
        "users": ("Identifier", "nunique"),
        "pred_accept_mean": ("pred_accept", "mean"),
        "observed_accept_rate": (y_col, lambda x: 1 - np.mean(x)),
        "observed_optout_rate": (y_col, "mean"),
    }
    if "Cool_Reduction_Frac" in d.columns:
        agg["nominal_reduction"] = ("Cool_Reduction_Frac", "mean")
    if "Delivered_Flex_Observed" in d.columns:
        agg["delivered_flex"] = ("Delivered_Flex_Observed", "mean")
    tab = d.groupby("reliability_decile").agg(**agg).reset_index()
    if "nominal_reduction" in tab.columns and "delivered_flex" in tab.columns:
        tab["behavior_discount"] = tab["delivered_flex"] / tab["nominal_reduction"].replace(0, np.nan)
    tab.to_csv(OUT / f"{prefix}_behavior_accounting_by_reliability_decile.csv", index=False)

    if "delivered_flex" in tab.columns:
        plt.figure(figsize=(7.0, 4.4))
        plt.plot(tab["reliability_decile"], tab["nominal_reduction"], marker="o", label="Nominal reduction")
        plt.plot(tab["reliability_decile"], tab["delivered_flex"], marker="s", label="Delivered flex")
        plt.xlabel("Reliability score decile")
        plt.ylabel("Mean value")
        plt.title("Behavior-adjusted value by reliability decile")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"{prefix}_behavior_accounting_by_reliability_decile.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    lines: List[str] = []
    log("=" * 78, lines)
    log("Remaining validation experiments for behavior-aware DR reliability", lines)
    log("=" * 78, lines)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find {DATA_PATH}. Run this script from your ecobee project directory.")

    raw = pd.read_csv(DATA_PATH)
    df, y_col, setback_col, time_col = make_clean_sample(raw, lines)

    # Full feature set for reliability score.
    weather_time = available_features(df, WEATHER_TIME_CANDIDATES)
    setback_feats = available_features(df, SETBACK_CANDIDATES)
    building_baseline = available_features(df, BUILDING_BASELINE_CANDIDATES)
    history_feats = available_features(df, USER_HISTORY_CANDIDATES)

    # Avoid duplicates while preserving order.
    full_features = []
    for c in weather_time + setback_feats + building_baseline + history_feats:
        if c not in full_features and c in df.columns:
            full_features.append(c)

    log("\nFeature groups:", lines)
    log(f"  weather/time: {weather_time}", lines)
    log(f"  setback: {setback_feats}", lines)
    log(f"  building/baseline: {building_baseline}", lines)
    log(f"  user history: {history_feats}", lines)

    # 1. GroupKFold reliability score validation.
    log("\n[1] User-level GroupKFold reliability-score validation", lines)
    valid_cols = ["Identifier", y_col] + full_features + [c for c in ["Cool_Reduction_Frac", "Delivered_Flex_Observed", "Setback_Bin", "Setback"] if c in df.columns]
    # Remove duplicate column names while preserving order. Duplicates can happen because Setback_Bin/Setback are also model features.
    valid_cols = list(dict.fromkeys(valid_cols))
    valid = df[valid_cols].dropna(subset=["Identifier", y_col]).copy()
    # fit_predict_groupkfold expects df with all features available; it internally handles missing feature values.
    oof, res_gkf = fit_predict_groupkfold(df, y_col, full_features, "full_model", lines)

    # Need the same valid dataframe used inside fit_predict_groupkfold for behavior-accounting alignment.
    valid_for_oof = df[["Identifier", y_col] + full_features + [c for c in ["Cool_Reduction_Frac", "Delivered_Flex_Observed", "Setback_Bin", "Setback"] if c in df.columns]].dropna(subset=["Identifier", y_col]).copy()
    y_valid = valid_for_oof[y_col].astype(int).values

    results = [res_gkf]
    metrics_df = pd.DataFrame([r.__dict__ for r in results])
    metrics_df.to_csv(OUT / "validation_metrics_summary.csv", index=False)
    log(f"  AUC={res_gkf.auc:.4f}, PR-AUC={res_gkf.pr_auc:.4f}, Brier={res_gkf.brier:.4f}, ECE={res_gkf.ece:.4f}", lines)
    log(f"  Calibration intercept={res_gkf.cal_intercept:.4f}, slope={res_gkf.cal_slope:.4f}", lines)

    dec = decile_table(y_valid, oof, "groupkfold")
    cal = calibration_table(y_valid, oof, "groupkfold")
    plot_decile(dec, "groupkfold", "Reliability score validation, user-level GroupKFold")
    plot_calibration(cal, "groupkfold", "Calibration, user-level GroupKFold")
    behavior_accounting_by_reliability_decile(valid_for_oof, y_col, oof, "groupkfold")

    # 2. Out-of-time validation.
    log("\n[2] Out-of-time validation", lines)
    if time_col is not None:
        try:
            pred_time, res_time = fit_predict_time_split(df, y_col, full_features, time_col, lines)
            pd.DataFrame([res_time.__dict__]).to_csv(OUT / "out_of_time_metrics.csv", index=False)
            log(f"  AUC={res_time.auc:.4f}, PR-AUC={res_time.pr_auc:.4f}, Brier={res_time.brier:.4f}, ECE={res_time.ece:.4f}", lines)
            dec_t = decile_table(pred_time[y_col].astype(int).values, pred_time["pred_optout"].values, "out_of_time")
            cal_t = calibration_table(pred_time[y_col].astype(int).values, pred_time["pred_optout"].values, "out_of_time")
            plot_decile(dec_t, "out_of_time", "Reliability score validation, out-of-time test")
            plot_calibration(cal_t, "out_of_time", "Calibration, out-of-time test")
        except Exception as e:
            log(f"  Out-of-time validation failed: {type(e).__name__}: {e}", lines)
    else:
        log("  No time column detected; skipping out-of-time validation", lines)

    # 3. Same-setback reliability heterogeneity.
    log("\n[3] Same-setback reliability heterogeneity", lines)
    experiment_same_setback_heterogeneity(df, y_col, lines)

    # 4. Save a clean result guide.
    log("\nRecommended files to inspect:", lines)
    for f in [
        "validation_metrics_summary.csv",
        "groupkfold_reliability_deciles.csv",
        "groupkfold_calibration_table.csv",
        "groupkfold_behavior_accounting_by_reliability_decile.csv",
        "out_of_time_metrics.csv",
        "out_of_time_reliability_deciles.csv",
        "same_3_4_setback_by_history_risk.csv",
    ]:
        log(f"  {OUT / f}", lines)
    log("\nRecommended figures:", lines)
    for f in [
        "groupkfold_reliability_deciles.png",
        "groupkfold_calibration.png",
        "groupkfold_behavior_accounting_by_reliability_decile.png",
        "out_of_time_reliability_deciles.png",
        "out_of_time_calibration.png",
        "fig_same_3_4_setback_by_history_risk.png",
    ]:
        log(f"  {FIG / f}", lines)

    (OUT / "remaining_experiments_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\nDone.", lines)


if __name__ == "__main__":
    main()
