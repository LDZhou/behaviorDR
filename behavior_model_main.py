#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_behavior_aware_dr_model.py

Behavior-aware DR reliability model for smart thermostat DR.

Core idea:
    Delivered flexibility = technical cooling reduction × user acceptance probability

This script builds the missing bridge between Paper1 empirical behavior results
and a full behavior-aware DR model:

1. Setback-bin composition diagnostics
2. Delivered flexibility by setback regime
3. Categorical adjusted opt-out model
4. Behavioral persistence / user-state diagnostics
5. Acceptance model with GroupKFold CV
6. Reduction model with GroupKFold CV
7. Behavior-adjusted policy benchmark

Input:
    dr_sessions.csv

Outputs:
    behavior_model_out/*.csv
    behavior_model_out/figs/*.png
    behavior_model_out/behavior_model_report.txt

Run:
    python 11_behavior_aware_dr_model.py

Notes:
    - Uses only user/event information available at or before event onset for ML policy models.
    - Does not use during-event leakage variables such as Comfort_Gap_Mean or Temp_Rise
      in policy prediction features, though they are reported in diagnostics.
    - Uses GroupKFold by Identifier for all predictive evaluation.
"""

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
INPUT = "dr_sessions.csv"
OUT = Path("behavior_model_out")
FIG = OUT / "figs"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

SETBACK_BINS = [0, 1, 2, 3, 4, 6]
SETBACK_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-6"]
REFERENCE_BIN = "2-3"

REPORT = []

def log(s=""):
    print(s, flush=True)
    REPORT.append(str(s))


def safe_to_csv(df, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def as_bool_int(s):
    if s is None:
        return None
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"]).astype(int)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================================================
# Load and prepare data
# ============================================================
def load_data():
    if not Path(INPUT).exists():
        raise FileNotFoundError(f"Cannot find {INPUT}. Run this script in the folder containing dr_sessions.csv")

    df = pd.read_csv(INPUT)
    log("=" * 78)
    log("Behavior-aware DR reliability model")
    log("=" * 78)
    log(f"Loaded {len(df):,} rows from {INPUT}")

    # Basic filters matching Paper1
    if "HvacMode" in df.columns:
        df = df[df["HvacMode"].astype(str).str.lower() == "cool"].copy()
    if "Setback_Amplitude_Mean" not in df.columns:
        raise ValueError("Missing Setback_Amplitude_Mean. This script requires session-level setback.")
    df = df[df["Setback_Amplitude_Mean"].between(0, 6)].copy()
    if "N_DR_Rows" in df.columns:
        df = df[df["N_DR_Rows"] >= 3].copy()

    # Time ordering
    if "Session_Start" in df.columns:
        df["Session_Start"] = pd.to_datetime(df["Session_Start"], errors="coerce")
        df = df.sort_values(["Identifier", "Session_Start"]).reset_index(drop=True)
    else:
        df = df.sort_values(["Identifier"]).reset_index(drop=True)

    # Main outcomes
    if "OptOut_Immediate" in df.columns:
        df["Y_immediate"] = as_bool_int(df["OptOut_Immediate"])
    elif "Opted_Out" in df.columns:
        df["Y_immediate"] = as_bool_int(df["Opted_Out"])
    else:
        raise ValueError("Missing OptOut_Immediate or Opted_Out outcome.")

    if "OptOut_Hold_Only" in df.columns:
        df["Y_hold_only"] = as_bool_int(df["OptOut_Hold_Only"])
    else:
        df["Y_hold_only"] = np.nan

    if "OptOut_StateChange" in df.columns:
        df["Y_state_change"] = as_bool_int(df["OptOut_StateChange"])
    else:
        df["Y_state_change"] = np.nan

    if "Opted_Out" in df.columns:
        df["Y_original_30min"] = as_bool_int(df["Opted_Out"])
    else:
        df["Y_original_30min"] = df["Y_immediate"]

    # Setback bin / regime
    df["Setback"] = df["Setback_Amplitude_Mean"].astype(float)
    df["Setback_sq"] = df["Setback"] ** 2
    df["Setback_Bin"] = pd.cut(
        df["Setback"],
        bins=SETBACK_BINS,
        labels=SETBACK_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)

    def regime_from_bin(b):
        if b in ["0-1", "1-2"]:
            return "weak_0_2"
        if b in ["2-3", "3-4"]:
            return "moderate_2_4"
        if b == "4-6":
            return "aggressive_4_6"
        return np.nan

    df["Setback_Regime"] = df["Setback_Bin"].map(regime_from_bin)

    # Core event/time features
    if "Month" not in df.columns and "Session_Start" in df.columns:
        df["Month"] = df["Session_Start"].dt.month
    if "Is_Weekend" not in df.columns and "Session_Start" in df.columns:
        df["Is_Weekend"] = df["Session_Start"].dt.dayofweek.isin([5, 6]).astype(int)
    if "Hour_of_Day" not in df.columns and "Session_Start" in df.columns:
        df["Hour_of_Day"] = df["Session_Start"].dt.hour
    if "Hour_Bin" not in df.columns and "Hour_of_Day" in df.columns:
        h = df["Hour_of_Day"]
        df["Hour_Bin"] = np.select(
            [h.between(12, 14), h.between(15, 18), h.between(19, 22)],
            ["early_afternoon", "peak_afternoon", "evening"],
            default="other",
        )

    # Cooling reduction / delivered flexibility
    if "Cool_Reduction_Frac" not in df.columns:
        if {"Baseline_Cool_Frac", "DR_Cool_Frac"}.issubset(df.columns):
            df["Cool_Reduction_Frac"] = df["Baseline_Cool_Frac"] - df["DR_Cool_Frac"]
        elif {"Avg_Baseline_Cool", "Avg_DR_Cool"}.issubset(df.columns):
            df["Cool_Reduction_Frac"] = (df["Avg_Baseline_Cool"] - df["Avg_DR_Cool"]) / 300.0
        else:
            df["Cool_Reduction_Frac"] = np.nan

    # Delivered flex observed under primary outcome
    df["Delivered_Flex_Observed"] = df["Cool_Reduction_Frac"] * (1 - df["Y_immediate"])

    # User history features based on primary outcome
    df["Session_Seq_Recalc"] = df.groupby("Identifier").cumcount() + 1
    df["Prev_OptOut"] = df.groupby("Identifier")["Y_immediate"].shift(1)
    cum_oo = df.groupby("Identifier")["Y_immediate"].cumsum()
    cumn = df.groupby("Identifier").cumcount() + 1
    df["Prior_OptOut_Rate"] = np.where(cumn > 1, (cum_oo - df["Y_immediate"]) / (cumn - 1), np.nan)
    df["Prior_OptOut_Rate_Filled"] = df["Prior_OptOut_Rate"].fillna(df["Y_immediate"].mean())

    # Streak of previous consecutive opt-outs
    streaks = []
    for _, g in df.groupby("Identifier", sort=False):
        st = 0
        for y in g["Y_immediate"].values:
            streaks.append(st)
            if y == 1:
                st += 1
            else:
                st = 0
    df["Prev_OptOut_Streak"] = streaks
    df["Prev_OptOut_Streak_Cap3"] = df["Prev_OptOut_Streak"].clip(upper=3)

    log(f"Filtered cooling sessions: {len(df):,}")
    log(f"Users: {df['Identifier'].nunique():,}")
    if "province_state" in df.columns:
        log(f"States/provinces: {df['province_state'].nunique():,}")
    log(f"Primary opt-out rate: {df['Y_immediate'].mean():.2%}")
    log(f"Mean setback: {df['Setback'].mean():.2f}°F")
    return df


# ============================================================
# 1. Setback-bin composition and delivered flexibility
# ============================================================
def diagnostics_by_setback_bin(df):
    log("\n[1] Setback-bin composition and delivered flexibility")

    cols_mean = [
        "Y_immediate", "Y_hold_only", "Y_state_change", "Y_original_30min",
        "Cool_Reduction_Frac", "Delivered_Flex_Observed",
        "Temp_Rise", "Comfort_Gap_Mean", "Comfort_Gap_Max",
        "Duration_Min", "Tout_onset", "CDH_during", "Baseline_Cool_Frac",
        "DR_Cool_Frac", "Avg_Baseline_Temp", "Setpoint_Cool_Start",
        "weather_is_fallback",
    ]
    cols_mean = [c for c in cols_mean if c in df.columns]

    agg = df.groupby("Setback_Bin", observed=True).agg(
        n=("Y_immediate", "count"),
        users=("Identifier", "nunique"),
        optout=("Y_immediate", "mean"),
        delivered_flex=("Delivered_Flex_Observed", "mean"),
        cool_reduction=("Cool_Reduction_Frac", "mean"),
    ).reset_index()

    # Add diagnostic means
    for c in cols_mean:
        if c not in ["Y_immediate", "Cool_Reduction_Frac", "Delivered_Flex_Observed"]:
            tmp = df.groupby("Setback_Bin", observed=True)[c].mean().reset_index(name=f"mean_{c}")
            agg = agg.merge(tmp, on="Setback_Bin", how="left")

    # Composition shares
    if "province_state" in df.columns:
        top_state = []
        top_state_share = []
        for b, g in df.groupby("Setback_Bin", observed=True):
            vc = g["province_state"].astype(str).value_counts(normalize=True)
            top_state.append(vc.index[0] if len(vc) else "")
            top_state_share.append(float(vc.iloc[0]) if len(vc) else np.nan)
        comp = pd.DataFrame({"Setback_Bin": list(df.groupby("Setback_Bin", observed=True).groups.keys()),
                             "top_state": top_state,
                             "top_state_share": top_state_share})
        agg = agg.merge(comp, on="Setback_Bin", how="left")

    if "DR_Type" in df.columns:
        cs_share = df.assign(is_cs=(df["DR_Type"] == "CS_DR").astype(int)).groupby("Setback_Bin", observed=True)["is_cs"].mean().reset_index(name="CS_DR_share")
        agg = agg.merge(cs_share, on="Setback_Bin", how="left")

    # Order rows
    agg["_ord"] = agg["Setback_Bin"].map({l: i for i, l in enumerate(SETBACK_LABELS)})
    agg = agg.sort_values("_ord").drop(columns="_ord")
    safe_to_csv(agg, OUT / "01_setback_bin_composition_delivered_flex.csv")
    log(agg[["Setback_Bin", "n", "optout", "cool_reduction", "delivered_flex"]].to_string(index=False))

    # Regime summary
    reg = df.dropna(subset=["Setback_Regime"]).groupby("Setback_Regime").agg(
        n=("Y_immediate", "count"),
        users=("Identifier", "nunique"),
        optout=("Y_immediate", "mean"),
        cool_reduction=("Cool_Reduction_Frac", "mean"),
        delivered_flex=("Delivered_Flex_Observed", "mean"),
        mean_setback=("Setback", "mean"),
    ).reset_index()
    safe_to_csv(reg, OUT / "01b_regime_summary.csv")

    # Plot: opt-out, nominal reduction, delivered reduction
    plot_df = agg.copy()
    x = np.arange(len(plot_df))
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.bar(x - 0.2, plot_df["optout"] * 100, width=0.4, label="Opt-out rate (%)")
    ax1.set_ylabel("Opt-out rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["Setback_Bin"])
    ax1.set_xlabel("Inferred setback amplitude (°F)")

    ax2 = ax1.twinx()
    ax2.plot(x, plot_df["cool_reduction"], marker="o", label="Nominal cooling reduction")
    ax2.plot(x, plot_df["delivered_flex"], marker="s", label="Delivered flexibility")
    ax2.set_ylabel("Cooling runtime fraction")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.set_title("Behavior-adjusted flexibility by setback bin")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_behavior_adjusted_flex_by_setback.png", dpi=180)
    plt.close(fig)

    return agg, reg


# ============================================================
# 2. Categorical adjusted opt-out model
# ============================================================
def categorical_adjusted_model(df):
    log("\n[2] Categorical adjusted opt-out model")
    out = []
    if not HAS_SM:
        log("  statsmodels not available; skipping adjusted categorical model")
        return pd.DataFrame()

    controls = []
    for c in ["Tout_onset", "CDH_during", "Duration_Min"]:
        if c in df.columns:
            controls.append(c)
    for c in ["Hour_Bin", "Month", "province_state", "DR_Type"]:
        if c in df.columns:
            controls.append(f"C({c})")
    if "Is_Weekend" in df.columns:
        controls.append("Is_Weekend")

    model_df = df.dropna(subset=["Y_immediate", "Setback_Bin", "Setback"] + [c for c in ["Tout_onset", "Duration_Min"] if c in df.columns]).copy()
    model_df = model_df[model_df["Setback_Bin"].isin(SETBACK_LABELS)].copy()
    model_df["Setback_Bin"] = pd.Categorical(model_df["Setback_Bin"], categories=SETBACK_LABELS)

    # Use treatment coding with 2-3 as reference via explicit relevel not trivial in formula.
    # Create dummies manually, reference 2-3.
    for b in SETBACK_LABELS:
        if b != REFERENCE_BIN:
            model_df[f"SB_{b.replace('-', '_')}"] = (model_df["Setback_Bin"].astype(str) == b).astype(int)
    sb_terms = " + ".join([f"SB_{b.replace('-', '_')}" for b in SETBACK_LABELS if b != REFERENCE_BIN])
    fml = "Y_immediate ~ " + sb_terms
    if controls:
        fml += " + " + " + ".join(controls)

    try:
        m = smf.glm(fml, data=model_df, family=__import__('statsmodels').api.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": model_df["Identifier"]}
        )
        log("  Adjusted categorical GLM succeeded")
        # Adjusted predictions: set each bin for all rows, average predicted probability
        pred_rows = []
        for b in SETBACK_LABELS:
            tmp = model_df.copy()
            for bb in SETBACK_LABELS:
                if bb != REFERENCE_BIN:
                    tmp[f"SB_{bb.replace('-', '_')}"] = int(bb == b)
            p = m.predict(tmp).mean()
            raw = model_df.loc[model_df["Setback_Bin"].astype(str) == b, "Y_immediate"].mean()
            n = int((model_df["Setback_Bin"].astype(str) == b).sum())
            pred_rows.append({"Setback_Bin": b, "n": n, "raw_optout": raw, "adjusted_optout": p})
        pred = pd.DataFrame(pred_rows)
        safe_to_csv(pred, OUT / "02_adjusted_categorical_optout.csv")

        coef = pd.DataFrame({
            "term": m.params.index,
            "coef": m.params.values,
            "se": m.bse.values,
            "pvalue": m.pvalues.values,
        })
        safe_to_csv(coef, OUT / "02b_adjusted_categorical_coefficients.csv")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(pred))
        ax.plot(x, pred["raw_optout"] * 100, marker="o", label="Raw")
        ax.plot(x, pred["adjusted_optout"] * 100, marker="s", label="Adjusted")
        ax.set_xticks(x)
        ax.set_xticklabels(pred["Setback_Bin"])
        ax.set_ylabel("Opt-out rate (%)")
        ax.set_xlabel("Setback bin (°F)")
        ax.set_title("Raw vs adjusted opt-out by setback bin")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "fig2_adjusted_categorical_optout.png", dpi=180)
        plt.close(fig)
        return pred
    except Exception as e:
        log(f"  Adjusted categorical GLM failed: {type(e).__name__}: {e}")
        return pd.DataFrame()


# ============================================================
# 3. Persistence / behavioral state diagnostics
# ============================================================
def persistence_diagnostics(df):
    log("\n[3] Persistence / behavioral state diagnostics")
    d = df.dropna(subset=["Prev_OptOut", "Y_immediate"]).copy()
    if len(d) == 0:
        log("  No repeated-user rows for persistence diagnostics")
        return

    # Transition matrix
    trans = d.groupby("Prev_OptOut").agg(
        n=("Y_immediate", "count"),
        current_optout=("Y_immediate", "mean"),
    ).reset_index()
    trans["Prev_Status"] = trans["Prev_OptOut"].map({0.0: "previous_stay", 1.0: "previous_optout"})
    safe_to_csv(trans, OUT / "03_persistence_transition.csv")
    log(trans[["Prev_Status", "n", "current_optout"]].to_string(index=False))

    # Streak analysis
    streak = d.copy()
    streak["Streak_Group"] = streak["Prev_OptOut_Streak"].clip(upper=3).astype(int).astype(str)
    streak.loc[streak["Streak_Group"] == "3", "Streak_Group"] = "3+"
    streak_tbl = streak.groupby("Streak_Group").agg(
        n=("Y_immediate", "count"),
        current_optout=("Y_immediate", "mean"),
    ).reset_index()
    order = {"0": 0, "1": 1, "2": 2, "3+": 3}
    streak_tbl["_ord"] = streak_tbl["Streak_Group"].map(order)
    streak_tbl = streak_tbl.sort_values("_ord").drop(columns="_ord")
    safe_to_csv(streak_tbl, OUT / "03b_persistence_streak.csv")

    # Prior rate groups
    hist = df.copy()
    hist["Risk_State"] = pd.cut(
        hist["Prior_OptOut_Rate_Filled"],
        bins=[-0.001, 0.10, 0.40, 1.0],
        labels=["low_prior_risk", "medium_prior_risk", "high_prior_risk"],
    ).astype(str)
    hist.loc[hist["Prev_OptOut"] == 1, "Risk_State"] = "recent_optout"
    risk_tbl = hist.groupby("Risk_State").agg(
        n=("Y_immediate", "count"),
        users=("Identifier", "nunique"),
        optout=("Y_immediate", "mean"),
        delivered_flex=("Delivered_Flex_Observed", "mean"),
    ).reset_index()
    safe_to_csv(risk_tbl, OUT / "03c_behavior_risk_states.csv")

    # Model Prev vs Prior with simple GLM fallback
    if HAS_SM:
        controls = []
        for c in ["Setback", "Setback_sq", "Tout_onset", "Duration_Min"]:
            if c in d.columns:
                controls.append(c)
        for c in ["Hour_Bin", "Month", "province_state", "DR_Type"]:
            if c in d.columns:
                controls.append(f"C({c})")
        if "Is_Weekend" in d.columns:
            controls.append("Is_Weekend")
        d2 = d.dropna(subset=["Y_immediate", "Prev_OptOut", "Prior_OptOut_Rate_Filled", "Setback", "Setback_sq"]).copy()
        fmls = {
            "prev_only": "Y_immediate ~ Prev_OptOut",
            "prev_plus_prior": "Y_immediate ~ Prev_OptOut + Prior_OptOut_Rate_Filled",
            "prev_prior_event_controls": "Y_immediate ~ Prev_OptOut + Prior_OptOut_Rate_Filled + " + " + ".join(controls),
        }
        rows = []
        import statsmodels.api as sm
        for name, fml in fmls.items():
            try:
                m = smf.glm(fml, data=d2, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": d2["Identifier"]}
                )
                for term in ["Prev_OptOut", "Prior_OptOut_Rate_Filled", "Setback", "Setback_sq"]:
                    if term in m.params.index:
                        rows.append({
                            "model": name,
                            "term": term,
                            "coef": float(m.params[term]),
                            "odds_ratio": float(np.exp(m.params[term])),
                            "pvalue": float(m.pvalues[term]),
                            "n": int(m.nobs),
                        })
                log(f"  {name}: succeeded")
            except Exception as e:
                log(f"  {name}: failed {type(e).__name__}: {e}")
        if rows:
            safe_to_csv(pd.DataFrame(rows), OUT / "03d_persistence_models.csv")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(trans["Prev_Status"], trans["current_optout"] * 100)
    ax[0].set_ylabel("Current opt-out rate (%)")
    ax[0].set_title("Transition by previous event")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(streak_tbl["Streak_Group"], streak_tbl["current_optout"] * 100)
    ax[1].set_ylabel("Current opt-out rate (%)")
    ax[1].set_xlabel("Previous opt-out streak")
    ax[1].set_title("Behavioral persistence by streak")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_behavioral_persistence.png", dpi=180)
    plt.close(fig)


# ============================================================
# ML utilities
# ============================================================
def get_feature_sets(df):
    # Only pre/onset or program-known variables. Avoid leakage from during DR outcomes.
    weather = [c for c in ["Tout_onset", "RH_onset", "GHI_onset", "dew_onset", "CDH_during"] if c in df.columns]
    time = [c for c in ["Duration_Min", "Hour_of_Day", "Is_Weekend", "Month"] if c in df.columns]
    cat_time = [c for c in ["Hour_Bin", "DR_Type", "province_state", "country"] if c in df.columns]
    setback = ["Setback"]
    cat_setback = ["Setback_Bin", "Setback_Regime"]
    building = [c for c in [
        "floor_area_sqft", "building_age_yrs", "number_occupants",
        "number_cool_stages", "number_heat_stages", "number_remote_sensors",
        "has_heatpump", "has_electric", "allow_comp_with_aux",
        "weather_is_fallback",
    ] if c in df.columns]
    baseline = [c for c in [
        "Baseline_Cool_Frac", "Avg_Baseline_Temp", "Setpoint_Cool_Start",
        "Avg_Baseline_Cool", "Avg_Baseline_Hum",
    ] if c in df.columns]
    history = [c for c in [
        "Session_Seq_Recalc", "Prev_OptOut", "Prior_OptOut_Rate_Filled",
        "Prev_OptOut_Streak", "Prev_OptOut_Streak_Cap3",
    ] if c in df.columns]

    feature_sets = {
        "M0_weather_time": weather + time + cat_time,
        "M1_plus_setback": weather + time + cat_time + setback + cat_setback,
        "M2_plus_building_baseline": weather + time + cat_time + setback + cat_setback + building + baseline,
        "M3_full_plus_history": weather + time + cat_time + setback + cat_setback + building + baseline + history,
    }
    # Remove duplicate while preserving order
    out = {}
    for k, vals in feature_sets.items():
        seen = set()
        out[k] = [v for v in vals if not (v in seen or seen.add(v))]
    return out


def build_preprocessor(df, features):
    cat_cols = [c for c in features if (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
    num_cols = [c for c in features if c not in cat_cols]

    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, num_cols, cat_cols


def make_classifier():
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    # fallback
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        random_state=RANDOM_STATE,
    )


def make_regressor():
    if HAS_LGB:
        return lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        random_state=RANDOM_STATE,
    )


def evaluate_classifier(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
        "mean_pred": float(np.mean(p)),
        "mean_y": float(np.mean(y)),
    }
    return out


def calibration_table(y, p, n_bins=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], q=n_bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(
        n=("y", "count"),
        pred_mean=("p", "mean"),
        obs_rate=("y", "mean"),
    ).reset_index().astype({"bin": str})


# ============================================================
# 4. Acceptance model with GroupKFold
# ============================================================
def train_acceptance_models(df):
    log("\n[4] Acceptance model: GroupKFold ablations")
    feature_sets = get_feature_sets(df)
    target = "Y_immediate"
    groups = df["Identifier"].astype(str).values

    rows = []
    oof_store = {}

    for name, features in feature_sets.items():
        model_df = df.dropna(subset=[target]).copy()
        # Need at least features existing
        features = [f for f in features if f in model_df.columns]
        X = model_df[features]
        y = model_df[target].astype(int).values
        g = model_df["Identifier"].astype(str).values

        oof = np.zeros(len(model_df))
        fold_rows = []
        gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(g))))
        for fold, (tr, te) in enumerate(gkf.split(X, y, g), 1):
            pre, _, _ = build_preprocessor(model_df, features)
            clf = make_classifier()
            pipe = Pipeline(steps=[("pre", pre), ("model", clf)])
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            met = evaluate_classifier(y[te], p)
            met.update({"model": name, "fold": fold, "n_test": len(te)})
            fold_rows.append(met)
        full_met = evaluate_classifier(y, oof)
        full_met.update({"model": name, "fold": "OOF", "n_test": len(y), "n_features": len(features)})
        rows.append(full_met)
        for r in fold_rows:
            r["n_features"] = len(features)
        rows.extend(fold_rows)
        oof_store[name] = (model_df.index.values, y, oof)
        log(f"  {name:28s}: AUC={full_met['auc']:.4f}, PR-AUC={full_met['pr_auc']:.4f}, Brier={full_met['brier']:.4f}")

        cal = calibration_table(y, oof)
        safe_to_csv(cal, OUT / f"04_calibration_{name}.csv")

    metrics = pd.DataFrame(rows)
    safe_to_csv(metrics, OUT / "04_acceptance_model_metrics.csv")

    # Plot OOF metrics
    oof_metrics = metrics[metrics["fold"] == "OOF"].copy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(oof_metrics["model"], oof_metrics["auc"])
    ax[0].set_title("Acceptance model OOF AUC")
    ax[0].set_ylim(max(0.5, oof_metrics["auc"].min() - 0.03), min(1.0, oof_metrics["auc"].max() + 0.03))
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].bar(oof_metrics["model"], oof_metrics["brier"])
    ax[1].set_title("Acceptance model OOF Brier")
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_acceptance_ablation_metrics.png", dpi=180)
    plt.close(fig)

    # Fit final full model for feature importance and policy use
    full_name = "M3_full_plus_history"
    features = feature_sets[full_name]
    model_df = df.dropna(subset=[target]).copy()
    X = model_df[features]
    y = model_df[target].astype(int).values
    pre, _, _ = build_preprocessor(model_df, features)
    clf = make_classifier()
    pipe = Pipeline(steps=[("pre", pre), ("model", clf)])
    pipe.fit(X, y)

    # Feature importance for LightGBM if available
    try:
        feat_names = pipe.named_steps["pre"].get_feature_names_out()
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
            imp = imp.sort_values("importance", ascending=False)
            safe_to_csv(imp, OUT / "04b_acceptance_feature_importance.csv")
            log("  Saved acceptance feature importance")
    except Exception as e:
        log(f"  Feature importance skipped: {e}")

    # Save full model predictions in-sample for behavior score diagnostics
    df["Pred_OptOut_FullModel_in_sample"] = pipe.predict_proba(df[features])[:, 1]
    df["Behavioral_Reliability_Score"] = 1 - df["Pred_OptOut_FullModel_in_sample"]
    rel = df.copy()
    rel["Reliability_Decile"] = pd.qcut(rel["Behavioral_Reliability_Score"], 10, duplicates="drop")
    rel_tbl = rel.groupby("Reliability_Decile", observed=True).agg(
        n=("Y_immediate", "count"),
        pred_accept=("Behavioral_Reliability_Score", "mean"),
        observed_accept=("Y_immediate", lambda x: 1 - x.mean()),
        delivered_flex=("Delivered_Flex_Observed", "mean"),
    ).reset_index().astype({"Reliability_Decile": str})
    safe_to_csv(rel_tbl, OUT / "04c_reliability_score_deciles.csv")

    return pipe, feature_sets[full_name], metrics, df


# ============================================================
# 5. Reduction model with GroupKFold
# ============================================================
def train_reduction_model(df, features):
    log("\n[5] Technical cooling reduction model")
    target = "Cool_Reduction_Frac"
    if target not in df.columns or df[target].notna().sum() < 100:
        log("  Missing or insufficient Cool_Reduction_Frac; skipping reduction model")
        return None, pd.DataFrame()

    # Train on non-opt-out sessions first, because opted-out sessions contaminate technical response.
    model_df = df[(df["Y_immediate"] == 0) & df[target].notna()].copy()
    if len(model_df) < 500:
        model_df = df[df[target].notna()].copy()
        log("  Warning: few non-opt-out sessions; using all sessions for reduction model")
    else:
        log(f"  Training on non-opt-out delivered sessions: {len(model_df):,}")

    features = [f for f in features if f in model_df.columns]
    X = model_df[features]
    y = model_df[target].astype(float).values
    g = model_df["Identifier"].astype(str).values
    oof = np.zeros(len(model_df))

    rows = []
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(g))))
    for fold, (tr, te) in enumerate(gkf.split(X, y, g), 1):
        pre, _, _ = build_preprocessor(model_df, features)
        reg = make_regressor()
        pipe = Pipeline(steps=[("pre", pre), ("model", reg)])
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        oof[te] = pred
        rows.append({
            "fold": fold,
            "n_test": len(te),
            "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
            "mae": float(mean_absolute_error(y[te], pred)),
            "r2": float(r2_score(y[te], pred)),
        })
    rows.append({
        "fold": "OOF",
        "n_test": len(y),
        "rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "mae": float(mean_absolute_error(y, oof)),
        "r2": float(r2_score(y, oof)),
    })
    metrics = pd.DataFrame(rows)
    safe_to_csv(metrics, OUT / "05_reduction_model_metrics.csv")
    oof_row = metrics[metrics["fold"] == "OOF"].iloc[0]
    log(f"  OOF RMSE={oof_row['rmse']:.4f}, MAE={oof_row['mae']:.4f}, R2={oof_row['r2']:.4f}")

    # Fit final model on reduction training set
    pre, _, _ = build_preprocessor(model_df, features)
    reg = make_regressor()
    pipe = Pipeline(steps=[("pre", pre), ("model", reg)])
    pipe.fit(X, y)

    return pipe, metrics


# ============================================================
# 6. Behavior-adjusted policy benchmark
# ============================================================
def policy_benchmark(df, acceptance_model, reduction_model, features):
    log("\n[6] Behavior-adjusted policy benchmark")
    if acceptance_model is None or reduction_model is None:
        log("  Missing acceptance or reduction model; skipping policy benchmark")
        return pd.DataFrame()

    # Candidate representative setbacks by bin
    arms = {
        "0-1": 0.5,
        "1-2": 1.5,
        "2-3": 2.5,
        "3-4": 3.5,
        "4-6": 5.0,
    }
    eval_df = df.copy()
    features = [f for f in features if f in eval_df.columns]

    pred_records = []
    for arm_label, arm_val in arms.items():
        tmp = eval_df.copy()
        tmp["Setback"] = arm_val
        tmp["Setback_Amplitude_Mean"] = arm_val
        tmp["Setback_sq"] = arm_val ** 2
        tmp["Setback_Bin"] = arm_label
        if arm_label in ["0-1", "1-2"]:
            tmp["Setback_Regime"] = "weak_0_2"
        elif arm_label in ["2-3", "3-4"]:
            tmp["Setback_Regime"] = "moderate_2_4"
        else:
            tmp["Setback_Regime"] = "aggressive_4_6"

        p_optout = acceptance_model.predict_proba(tmp[features])[:, 1]
        accept = 1 - p_optout
        red = reduction_model.predict(tmp[features])
        red = np.clip(red, -1, 1)
        delivered = red * accept
        pred_records.append(pd.DataFrame({
            "Identifier": tmp["Identifier"].values,
            "Session_Index": np.arange(len(tmp)),
            "arm": arm_label,
            "arm_setback": arm_val,
            "pred_optout": p_optout,
            "pred_accept": accept,
            "pred_reduction": red,
            "pred_delivered": delivered,
        }))
    pred_long = pd.concat(pred_records, ignore_index=True)
    safe_to_csv(pred_long.groupby("arm").agg(
        n=("pred_delivered", "count"),
        pred_optout=("pred_optout", "mean"),
        pred_accept=("pred_accept", "mean"),
        pred_reduction=("pred_reduction", "mean"),
        pred_delivered=("pred_delivered", "mean"),
    ).reset_index(), OUT / "06_predicted_arm_values.csv")

    # Choose best arm per session
    wide = pred_long.pivot(index="Session_Index", columns="arm", values="pred_delivered")
    best_arm = wide.idxmax(axis=1)
    eval_df["Policy_Best_Arm"] = best_arm.values

    # Define simple rule policy based on behavioral state
    eval_df["Rule_Behavior_Aware_Arm"] = "3-4"
    high_risk = (eval_df["Prev_OptOut"].fillna(0) == 1) | (eval_df["Prior_OptOut_Rate_Filled"] >= 0.4)
    eval_df.loc[high_risk, "Rule_Behavior_Aware_Arm"] = "2-3"
    low_risk = (eval_df["Prior_OptOut_Rate_Filled"] < 0.1) & (eval_df["Prev_OptOut"].fillna(0) == 0)
    # only keep 3-4 for low risk, do not force aggressive in simple rule
    eval_df.loc[low_risk, "Rule_Behavior_Aware_Arm"] = "3-4"

    # Evaluate predicted value of policies
    def policy_value(arm_series, name):
        idx = pd.DataFrame({"Session_Index": np.arange(len(eval_df)), "arm": arm_series.values})
        val = idx.merge(pred_long, on=["Session_Index", "arm"], how="left")
        return {
            "policy": name,
            "pred_optout": val["pred_optout"].mean(),
            "pred_accept": val["pred_accept"].mean(),
            "pred_reduction": val["pred_reduction"].mean(),
            "pred_delivered": val["pred_delivered"].mean(),
        }

    policies = []
    for arm in arms:
        policies.append(policy_value(pd.Series([arm] * len(eval_df)), f"uniform_{arm}"))
    policies.append(policy_value(eval_df["Rule_Behavior_Aware_Arm"], "rule_behavior_aware"))
    policies.append(policy_value(eval_df["Policy_Best_Arm"], "model_targeted_argmax"))
    policies_df = pd.DataFrame(policies).sort_values("pred_delivered", ascending=False)
    safe_to_csv(policies_df, OUT / "06_policy_benchmark_predicted_values.csv")

    # Distribution of selected arms
    selected = pd.DataFrame({
        "policy": ["rule_behavior_aware"] * len(eval_df) + ["model_targeted_argmax"] * len(eval_df),
        "arm": pd.concat([eval_df["Rule_Behavior_Aware_Arm"], eval_df["Policy_Best_Arm"]], ignore_index=True),
    })
    dist = selected.groupby(["policy", "arm"]).size().reset_index(name="n")
    dist["share"] = dist["n"] / dist.groupby("policy")["n"].transform("sum")
    safe_to_csv(dist, OUT / "06b_policy_selected_arm_distribution.csv")

    log(policies_df.to_string(index=False))

    # Plot policy values
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = policies_df.sort_values("pred_delivered")
    ax.barh(plot_df["policy"], plot_df["pred_delivered"])
    ax.set_xlabel("Predicted behavior-adjusted delivered flexibility")
    ax.set_title("Policy benchmark, model-based predicted values")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_policy_benchmark.png", dpi=180)
    plt.close(fig)

    return policies_df


# ============================================================
# 7. Low-setback anomaly check
# ============================================================
def low_setback_anomaly_check(df):
    log("\n[7] Low-setback anomaly check: 0-1°F vs 2-3°F")
    sub = df[df["Setback_Bin"].isin(["0-1", "2-3"])].copy()
    if len(sub) == 0:
        return
    vars_to_compare = [
        "Y_immediate", "Cool_Reduction_Frac", "Delivered_Flex_Observed",
        "Duration_Min", "Tout_onset", "CDH_during", "Comfort_Gap_Mean",
        "Temp_Rise", "Baseline_Cool_Frac", "DR_Cool_Frac",
        "Avg_Baseline_Temp", "Setpoint_Cool_Start", "weather_is_fallback",
        "N_DR_Rows", "Prior_OptOut_Rate_Filled",
    ]
    vars_to_compare = [v for v in vars_to_compare if v in sub.columns]
    rows = []
    for v in vars_to_compare:
        g = sub.groupby("Setback_Bin")[v].mean()
        rows.append({
            "variable": v,
            "mean_0_1": g.get("0-1", np.nan),
            "mean_2_3": g.get("2-3", np.nan),
            "diff_0_1_minus_2_3": g.get("0-1", np.nan) - g.get("2-3", np.nan),
        })
    out = pd.DataFrame(rows)
    safe_to_csv(out, OUT / "07_low_setback_vs_reference_diagnostics.csv")
    log(out.head(12).to_string(index=False))


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    df = load_data()
    diagnostics_by_setback_bin(df)
    categorical_adjusted_model(df)
    persistence_diagnostics(df)
    acceptance_model, acc_features, acc_metrics, df_scored = train_acceptance_models(df)
    reduction_model, red_metrics = train_reduction_model(df_scored, acc_features)
    policy_benchmark(df_scored, acceptance_model, reduction_model, acc_features)
    low_setback_anomaly_check(df_scored)

    # Save scored data subset with key outputs
    key_cols = [
        "Identifier", "Session_Start", "Setback", "Setback_Bin", "Setback_Regime",
        "Y_immediate", "Cool_Reduction_Frac", "Delivered_Flex_Observed",
        "Prev_OptOut", "Prior_OptOut_Rate_Filled", "Prev_OptOut_Streak",
        "Pred_OptOut_FullModel_in_sample", "Behavioral_Reliability_Score",
    ]
    key_cols = [c for c in key_cols if c in df_scored.columns]
    safe_to_csv(df_scored[key_cols], OUT / "99_session_behavior_scores.csv")

    log("\n" + "=" * 78)
    log(f"Finished in {(time.time() - t0) / 60:.1f} minutes")
    log(f"Outputs written to: {OUT.resolve()}")
    log("=" * 78)

    with open(OUT / "behavior_model_report.txt", "w") as f:
        f.write("\n".join(REPORT))


if __name__ == "__main__":
    main()
