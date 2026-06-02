"""
Paper 1 robustness and completion analyses.

V2: adds GLM Binomial fallbacks for singular-matrix Logit fits and FE-level diagnostics.

Input:  dr_sessions.csv
Output: paper1_out/robustness/*

What this script adds:
  1) Opt-out definition sensitivity: immediate / hold-only / state-change / original
  2) Main U-shape logit under each outcome definition
  3) Leave-one-state-out robustness
  4) Common-support / trimming robustness
  5) Event-type split robustness
  6) User fixed-effect LPM via within-user demeaning
  7) Comfort mechanism check: add Comfort_Gap_Mean / Comfort_Gap_Max
  8) Persistence robustness: Prev_OptOut vs user prior opt-out rate
  9) Publication-ready CSV tables + simple figures

Run from the same directory as dr_sessions.csv:
    python 09_paper1_robustness.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT = Path("paper1_out")
ROB = OUT / "robustness"
FIG = ROB / "figs"
ROB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

LOG: List[str] = []

def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def as_int_bool(s: pd.Series) -> pd.Series:
    """Convert bool/0/1/string-ish column to 0/1 with NaN preserved."""
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return s.astype(str).str.lower().map({
        "true": 1, "false": 0, "1": 1, "0": 0,
        "yes": 1, "no": 0, "nan": np.nan, "none": np.nan, "": np.nan,
    }).astype(float)


def prep_data(path: str = "dr_sessions.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic filters matching paper1_analysis.py
    if "HvacMode" in df.columns:
        df = df[df["HvacMode"].astype(str).str.lower().eq("cool")].copy()
    df["Setback"] = safe_num(df["Setback_Amplitude_Mean"])
    df = df[df["Setback"].between(0, 6)].copy()
    if "N_DR_Rows" in df.columns:
        df = df[safe_num(df["N_DR_Rows"]) >= 3].copy()

    # Dates / ordering
    if "Session_Start" in df.columns:
        df["Session_Start"] = pd.to_datetime(df["Session_Start"], errors="coerce")
        df = df.sort_values(["Identifier", "Session_Start"]).copy()
    else:
        df = df.sort_values(["Identifier"]).copy()

    # Core variables
    df["Setback_sq"] = df["Setback"] ** 2
    df["Precool"] = safe_num(df["Precool_Depth"]).fillna(0) if "Precool_Depth" in df.columns else 0.0
    df["Tout_onset"] = safe_num(df["Tout_onset"])
    df["CDH_during"] = safe_num(df["CDH_during"])
    df["Duration_Min"] = safe_num(df["Duration_Min"])

    if "Is_Weekend" in df.columns:
        df["Weekend"] = as_int_bool(df["Is_Weekend"]).fillna(0)
    else:
        df["Weekend"] = 0

    if "has_heatpump" in df.columns:
        df["HP"] = as_int_bool(df["has_heatpump"]).fillna(0)
    else:
        df["HP"] = 0

    if "floor_area_sqft" in df.columns:
        df["floor_1k"] = safe_num(df["floor_area_sqft"]) / 1000.0
    else:
        df["floor_1k"] = np.nan

    if "building_age_yrs" in df.columns:
        df["age_dec"] = safe_num(df["building_age_yrs"]) / 10.0
    else:
        df["age_dec"] = np.nan

    if "number_occupants" in df.columns:
        occ = safe_num(df["number_occupants"])
        df["log_occ"] = np.log1p(occ.fillna(occ.median()))
    else:
        df["log_occ"] = 0.0

    # Categorical controls with safe defaults
    for c, default in [("province_state", "UNK"), ("Hour_Bin", "other"), ("Month", "UNK"), ("DR_Type", "UNK")]:
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default).astype(str)

    # Outcome definitions available in your 02_sessions.py output
    outcome_candidates = {
        "immediate": "OptOut_Immediate",
        "hold_only": "OptOut_Hold_Only",
        "state_change": "OptOut_StateChange",
        "original_30min": "OptOut_Original",
        "opted_out_alias": "Opted_Out",
    }
    for name, col in outcome_candidates.items():
        if col in df.columns:
            df[f"Y_{name}"] = as_int_bool(df[col])

    # Preferred primary if available
    if "Y_immediate" not in df.columns and "Y_opted_out_alias" in df.columns:
        df["Y_immediate"] = df["Y_opted_out_alias"]

    # History variables for each outcome will be generated later per outcome.
    return df


def model_formula(y_col: str, include_comfort: Optional[str] = None, include_state_fe: bool = True) -> str:
    terms = [
        "Setback", "Setback_sq", "Precool", "Tout_onset", "CDH_during",
        "Duration_Min", "Weekend", "HP", "floor_1k", "age_dec", "log_occ",
        "C(Hour_Bin)", "C(Month)",
    ]
    if include_state_fe:
        terms.append("C(province_state)")
    if include_comfort is not None:
        terms.append(include_comfort)
    return f"{y_col} ~ " + " + ".join(terms)


def model_vars(include_comfort: Optional[str] = None) -> List[str]:
    cols = [
        "Setback", "Setback_sq", "Precool", "Tout_onset", "CDH_during",
        "Duration_Min", "Weekend", "HP", "floor_1k", "age_dec", "log_occ",
        "Hour_Bin", "Month", "province_state", "Identifier",
    ]
    if include_comfort is not None:
        cols.append(include_comfort)
    return cols


def extract_result(result, y_col: str, model_name: str, n_users: int, extra: Dict = None) -> Dict:
    extra = extra or {}
    params = getattr(result, "params", pd.Series(dtype=float))
    if not hasattr(params, "get"):
        params = pd.Series(params)
    pvalues = getattr(result, "pvalues", pd.Series(dtype=float))
    if not hasattr(pvalues, "get"):
        pvalues = pd.Series(pvalues, index=params.index if hasattr(params, "index") else None)
    out = {
        "model": model_name,
        "outcome": y_col,
        "n": int(result.nobs) if hasattr(result, "nobs") else np.nan,
        "users": n_users,
        "coef_setback": params.get("Setback", np.nan),
        "p_setback": pvalues.get("Setback", np.nan),
        "coef_setback_sq": params.get("Setback_sq", np.nan),
        "p_setback_sq": pvalues.get("Setback_sq", np.nan),
    }
    b1, b2 = out["coef_setback"], out["coef_setback_sq"]
    if pd.notna(b1) and pd.notna(b2) and b2 != 0:
        out["implied_optimum_F"] = -b1 / (2 * b2)
    else:
        out["implied_optimum_F"] = np.nan
    out.update(extra)
    return out


def _drop_degenerate_categories(d: pd.DataFrame, y_col: str, cat_cols: List[str], min_per_level: int = 20) -> pd.DataFrame:
    """Remove categorical levels that are too sparse or have no outcome variation.
    This prevents fixed-effect logit singularities / quasi-complete separation.
    Used only for robustness fits, and reported through estimator_method.
    """
    out = d.copy()
    for c in cat_cols:
        if c not in out.columns:
            continue
        keep_levels = []
        for lev, g in out.groupby(c, observed=True):
            n = len(g)
            nunique = g[y_col].nunique(dropna=True)
            # Keep sparse levels only if they are not degenerate; otherwise they are separation bait.
            if n >= min_per_level and nunique >= 2:
                keep_levels.append(lev)
        if keep_levels and len(keep_levels) < out[c].nunique():
            out = out[out[c].isin(keep_levels)].copy()
    return out


def fit_logit(df: pd.DataFrame, y_col: str, name: str, include_comfort: Optional[str] = None,
              include_state_fe: bool = True, min_n: int = 200) -> Optional[Dict]:
    needed = [y_col] + model_vars(include_comfort)
    needed = [c for c in needed if c in df.columns]
    d0 = df[needed].dropna().copy()
    if len(d0) < min_n or d0[y_col].nunique() < 2:
        log(f"  SKIP {name}: insufficient data n={len(d0)}, classes={d0[y_col].nunique() if len(d0) else 0}")
        return None

    # Formula with fixed effects as requested.
    fml = model_formula(y_col, include_comfort=include_comfort, include_state_fe=include_state_fe)

    # Try 1: original clustered Logit on full sample.
    try:
        res = smf.logit(fml, data=d0).fit(disp=0, maxiter=300, method="lbfgs",
                                           cov_type="cluster", cov_kwds={"groups": d0["Identifier"]})
        return extract_result(res, y_col, name, d0["Identifier"].nunique(),
                              {"estimator_method": "Logit_cluster_full", "n_before_fe_drop": len(d0)})
    except Exception as e1:
        # Try 2: GLM Binomial is numerically more forgiving and uses a pseudo-inverse more often.
        try:
            res = smf.glm(fml, data=d0, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": d0["Identifier"]}, maxiter=300
            )
            out = extract_result(res, y_col, name, d0["Identifier"].nunique(),
                                 {"estimator_method": "GLM_Binomial_cluster_full",
                                  "n_before_fe_drop": len(d0),
                                  "fallback_from": f"{type(e1).__name__}: {e1}"})
            log(f"  NOTE {name}: Logit failed, used GLM Binomial clustered fallback")
            return out
        except Exception as e2:
            # Try 3: drop degenerate FE levels, then clustered GLM. This is a robustness diagnostic, not the main model.
            cat_cols = ["Hour_Bin", "Month"] + (["province_state"] if include_state_fe else [])
            d = _drop_degenerate_categories(d0, y_col, cat_cols, min_per_level=20)
            if len(d) < min_n or d[y_col].nunique() < 2:
                log(f"  FAIL {name}: after dropping degenerate FE levels n={len(d)}; original errors: {type(e1).__name__}, {type(e2).__name__}")
                return None
            try:
                res = smf.glm(fml, data=d, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": d["Identifier"]}, maxiter=300
                )
                out = extract_result(res, y_col, name, d["Identifier"].nunique(),
                                     {"estimator_method": "GLM_Binomial_cluster_drop_degenerate_FE",
                                      "n_before_fe_drop": len(d0),
                                      "n_after_fe_drop": len(d),
                                      "fallback_from": f"{type(e1).__name__}: {e1} | {type(e2).__name__}: {e2}"})
                log(f"  NOTE {name}: used GLM fallback after dropping degenerate FE levels: {len(d0)} -> {len(d)} rows")
                return out
            except Exception as e3:
                # Try 4: no state FE, but keep temporal controls. Useful to diagnose whether state FE are causing singularity.
                try:
                    fml2 = model_formula(y_col, include_comfort=include_comfort, include_state_fe=False)
                    res = smf.glm(fml2, data=d0, family=sm.families.Binomial()).fit(
                        cov_type="cluster", cov_kwds={"groups": d0["Identifier"]}, maxiter=300
                    )
                    out = extract_result(res, y_col, name, d0["Identifier"].nunique(),
                                         {"estimator_method": "GLM_Binomial_cluster_no_stateFE_DIAGNOSTIC",
                                          "n_before_fe_drop": len(d0),
                                          "fallback_from": f"{type(e1).__name__}: {e1} | {type(e2).__name__}: {e2} | {type(e3).__name__}: {e3}"})
                    log(f"  NOTE {name}: used no-state-FE diagnostic fallback")
                    return out
                except Exception as e4:
                    log(f"  FAIL {name}: {type(e1).__name__}: {e1}; GLM: {type(e2).__name__}: {e2}; drop-FE: {type(e3).__name__}: {e3}; no-stateFE: {type(e4).__name__}: {e4}")
                    return None


def fit_logit_simple(df: pd.DataFrame, y_col: str, name: str, extra_terms: List[str],
                     base_terms: List[str], min_n: int = 200) -> Optional[Dict]:
    terms = base_terms + extra_terms
    cats = [t.replace("C(", "").replace(")", "") for t in terms if t.startswith("C(")]
    raw_terms = [t for t in terms if not t.startswith("C(")]
    needed = [y_col, "Identifier"] + raw_terms + cats
    needed = [c for c in needed if c in df.columns]
    d = df[needed].dropna().copy()
    if len(d) < min_n or d[y_col].nunique() < 2:
        log(f"  SKIP {name}: insufficient data n={len(d)}")
        return None
    fml = f"{y_col} ~ " + " + ".join(terms)
    try:
        res = smf.logit(fml, data=d).fit(disp=0, maxiter=200,
                                          cov_type="cluster", cov_kwds={"groups": d["Identifier"]})
        return extract_result(res, y_col, name, d["Identifier"].nunique())
    except Exception as e:
        log(f"  FAIL {name}: {type(e).__name__}: {e}")
        return None


def within_user_lpm(df: pd.DataFrame, y_col: str, name: str) -> Optional[Dict]:
    """User fixed-effect LPM using within-user demeaning; avoids creating 5k user dummies."""
    base = [
        "Setback", "Setback_sq", "Precool", "Tout_onset", "CDH_during",
        "Duration_Min", "Weekend", "HP", "floor_1k", "age_dec", "log_occ",
        "Identifier", y_col, "Hour_Bin", "Month", "DR_Type",
    ]
    d = df[[c for c in base if c in df.columns]].dropna().copy()
    # Need at least 2 sessions per user and within-user outcome variation helps but is not strictly required for LPM
    counts = d["Identifier"].value_counts()
    d = d[d["Identifier"].isin(counts[counts >= 2].index)].copy()
    if len(d) < 200 or d[y_col].nunique() < 2:
        log(f"  SKIP {name}: insufficient FE sample n={len(d)}")
        return None

    x_num = ["Setback", "Setback_sq", "Precool", "Tout_onset", "CDH_during",
             "Duration_Min", "Weekend", "HP", "floor_1k", "age_dec", "log_occ"]
    X_parts = [d[x_num].astype(float)]
    # Low-cardinality dummies. Drop first to avoid full collinearity before demeaning.
    for cat in ["Hour_Bin", "Month", "DR_Type"]:
        if cat in d.columns:
            X_parts.append(pd.get_dummies(d[cat].astype(str), prefix=cat, drop_first=True, dtype=float))
    X = pd.concat(X_parts, axis=1)
    y = d[y_col].astype(float)

    # Within transform by user
    groups = d["Identifier"]
    X_dm = X - X.groupby(groups).transform("mean")
    y_dm = y - y.groupby(groups).transform("mean")

    # Drop columns with no within variation
    keep = X_dm.var(axis=0) > 1e-12
    X_dm = X_dm.loc[:, keep]
    if "Setback" not in X_dm.columns or "Setback_sq" not in X_dm.columns:
        log(f"  SKIP {name}: no within-user variation in setback terms")
        return None

    try:
        res = sm.OLS(y_dm.values, X_dm.astype(float).values).fit(
            cov_type="cluster", cov_kwds={"groups": groups.values}
        )
        params = pd.Series(res.params, index=X_dm.columns)
        pvals = pd.Series(res.pvalues, index=X_dm.columns)
        b1, b2 = params.get("Setback", np.nan), params.get("Setback_sq", np.nan)
        return {
            "model": name,
            "outcome": y_col,
            "n": len(d),
            "users": d["Identifier"].nunique(),
            "coef_setback": b1,
            "p_setback": pvals.get("Setback", np.nan),
            "coef_setback_sq": b2,
            "p_setback_sq": pvals.get("Setback_sq", np.nan),
            "implied_optimum_F": -b1 / (2 * b2) if pd.notna(b1) and pd.notna(b2) and b2 != 0 else np.nan,
        }
    except Exception as e:
        log(f"  FAIL {name}: {type(e).__name__}: {e}")
        return None


def persistence_models(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(["Identifier", "Session_Start"] if "Session_Start" in d.columns else ["Identifier"])
    d["Prev_OptOut"] = d.groupby("Identifier")[y_col].shift(1)
    d["CumOO"] = d.groupby("Identifier")[y_col].cumsum()
    d["Seq"] = d.groupby("Identifier").cumcount() + 1
    d["Prior_OptOut_Rate"] = np.where(d["Seq"] > 1, (d["CumOO"] - d[y_col]) / (d["Seq"] - 1), np.nan)

    base_terms = ["Setback", "Setback_sq", "Tout_onset", "CDH_during", "Duration_Min",
                  "Weekend", "C(Hour_Bin)", "C(Month)", "C(province_state)"]
    rows = []
    specs = {
        "lag_only": ["Prev_OptOut"],
        "lag_plus_event_controls": ["Prev_OptOut"] + base_terms,
        "lag_plus_prior_rate_plus_event_controls": ["Prev_OptOut", "Prior_OptOut_Rate"] + base_terms,
    }
    for name, terms in specs.items():
        cats = [t.replace("C(", "").replace(")", "") for t in terms if t.startswith("C(")]
        raw = [t for t in terms if not t.startswith("C(")]
        needed = [y_col, "Identifier"] + raw + cats
        dd = d[[c for c in needed if c in d.columns]].dropna().copy()
        if len(dd) < 200 or dd[y_col].nunique() < 2:
            continue
        fml = f"{y_col} ~ " + " + ".join(terms)
        try:
            res = smf.logit(fml, data=dd).fit(disp=0, maxiter=200,
                                               cov_type="cluster", cov_kwds={"groups": dd["Identifier"]})
            coef = res.params.get("Prev_OptOut", np.nan)
            rows.append({
                "outcome": y_col,
                "model": name,
                "n": len(dd),
                "users": dd["Identifier"].nunique(),
                "coef_prev_optout": coef,
                "OR_prev_optout": np.exp(coef) if pd.notna(coef) else np.nan,
                "p_prev_optout": res.pvalues.get("Prev_OptOut", np.nan),
                "coef_prior_rate": res.params.get("Prior_OptOut_Rate", np.nan),
                "p_prior_rate": res.pvalues.get("Prior_OptOut_Rate", np.nan),
            })
        except Exception as e:
            log(f"  FAIL persistence {name}: {type(e).__name__}: {e}")

    # Descriptive transition rates
    trans = d[[y_col, "Prev_OptOut"]].dropna()
    if len(trans):
        rows.append({
            "outcome": y_col,
            "model": "descriptive_transition",
            "n": len(trans),
            "users": np.nan,
            "P_OO_given_prev_OO": trans.loc[trans["Prev_OptOut"] == 1, y_col].mean(),
            "P_OO_given_prev_stay": trans.loc[trans["Prev_OptOut"] == 0, y_col].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    log("=" * 72)
    log("Paper 1 robustness analyses")
    log("=" * 72)

    df = prep_data("dr_sessions.csv")
    log(f"Loaded filtered cooling sessions: {len(df):,}")
    log(f"Users: {df['Identifier'].nunique():,}; states: {df['province_state'].nunique():,}")

    outcome_cols = [c for c in ["Y_immediate", "Y_hold_only", "Y_state_change", "Y_original_30min"] if c in df.columns]
    if not outcome_cols:
        raise RuntimeError("No opt-out outcome columns found. Expected OptOut_Immediate / OptOut_Hold_Only / OptOut_StateChange / OptOut_Original.")
    log("Outcomes found: " + ", ".join(outcome_cols))

    # ------------------------------------------------------------------
    # 0. Outcome definition rates + raw bins
    # ------------------------------------------------------------------
    rates = []
    for y in outcome_cols:
        rates.append({"outcome": y, "n": df[y].notna().sum(), "rate": df[y].mean()})
    pd.DataFrame(rates).to_csv(ROB / "00_outcome_rates.csv", index=False)

    # Diagnostics for sparse / separated fixed-effect cells.
    diag_rows = []
    for y in outcome_cols:
        for cat in ["province_state", "Hour_Bin", "Month", "DR_Type"]:
            if cat in df.columns:
                g = df.dropna(subset=[y]).groupby(cat, observed=True)[y].agg(["count", "mean", "sum"]).reset_index()
                g["outcome"] = y
                g["category"] = cat
                g["all_zero_or_one"] = (g["mean"].eq(0) | g["mean"].eq(1))
                diag_rows.append(g.rename(columns={cat: "level"}))
    if diag_rows:
        pd.concat(diag_rows, ignore_index=True).to_csv(ROB / "00b_outcome_by_FE_level_diagnostics.csv", index=False)

    raw_bins = []
    for y in outcome_cols:
        tmp = df.dropna(subset=[y, "Setback"]).copy()
        tmp["SB_bin"] = pd.cut(tmp["Setback"], bins=[0, 1, 2, 3, 4, 6], include_lowest=True)
        g = tmp.groupby("SB_bin", observed=True).agg(
            n=(y, "count"), opt_out=(y, "mean"),
            temp_rise=("Temp_Rise", "mean") if "Temp_Rise" in tmp.columns else (y, "mean"),
            cool_red=("Cool_Reduction_Frac", "mean") if "Cool_Reduction_Frac" in tmp.columns else (y, "mean"),
            comfort_gap=("Comfort_Gap_Mean", "mean") if "Comfort_Gap_Mean" in tmp.columns else (y, "mean"),
        ).reset_index()
        g["outcome"] = y
        raw_bins.append(g)
    pd.concat(raw_bins, ignore_index=True).to_csv(ROB / "01_raw_setback_bins_by_outcome.csv", index=False)

    # ------------------------------------------------------------------
    # 1. Opt-out definition sensitivity
    # ------------------------------------------------------------------
    log("\n[1] Outcome-definition sensitivity")
    rows = []
    for y in outcome_cols:
        r = fit_logit(df, y, f"main_logit_{y}")
        if r is not None:
            rows.append(r)
            log(f"  {y}: b1={r['coef_setback']:+.4f}, b2={r['coef_setback_sq']:+.4f}, opt={r['implied_optimum_F']:.2f}, p2={r['p_setback_sq']:.2e}")
    sens = pd.DataFrame(rows)
    sens.to_csv(ROB / "02_outcome_definition_sensitivity.csv", index=False)

    primary = "Y_immediate" if "Y_immediate" in outcome_cols else outcome_cols[0]

    # ------------------------------------------------------------------
    # 2. Leave-one-state-out
    # ------------------------------------------------------------------
    log("\n[2] Leave-one-state-out")
    state_counts = df["province_state"].value_counts()
    big_states = list(state_counts[state_counts >= 500].index)
    loo_rows = []
    for st in big_states:
        sub = df[df["province_state"] != st].copy()
        r = fit_logit(sub, primary, f"drop_{st}")
        if r is not None:
            r["dropped_state"] = st
            r["dropped_state_n"] = int(state_counts.loc[st])
            loo_rows.append(r)
            log(f"  drop {st:>3s}: opt={r['implied_optimum_F']:.2f}, p2={r['p_setback_sq']:.2e}")
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(ROB / "03_leave_one_state_out.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Common-support / trimming robustness
    # ------------------------------------------------------------------
    log("\n[3] Common support / trimming")
    restrictions = {
        "all_0_6": df["Setback"].between(0, 6),
        "trim_0p5_5p5": df["Setback"].between(0.5, 5.5),
        "core_1_5": df["Setback"].between(1, 5),
        "exclude_0_1": ~df["Setback"].between(0, 1, inclusive="both"),
        "exclude_4_6": ~df["Setback"].between(4, 6, inclusive="both"),
        "middle_1_4p5": df["Setback"].between(1, 4.5),
    }
    trim_rows = []
    for name, mask in restrictions.items():
        sub = df[mask].copy()
        r = fit_logit(sub, primary, f"support_{name}")
        if r is not None:
            r["restriction"] = name
            trim_rows.append(r)
            log(f"  {name:>16s}: n={r['n']}, opt={r['implied_optimum_F']:.2f}, p2={r['p_setback_sq']:.2e}")
    pd.DataFrame(trim_rows).to_csv(ROB / "04_common_support_trimming.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Event type split
    # ------------------------------------------------------------------
    log("\n[4] Event type split")
    et_rows = []
    if "DR_Type" in df.columns:
        for et, sub in df.groupby("DR_Type"):
            r = fit_logit(sub.copy(), primary, f"event_type_{et}", include_state_fe=True)
            if r is not None:
                r["event_type"] = et
                et_rows.append(r)
                log(f"  {et:>12s}: n={r['n']}, opt={r['implied_optimum_F']:.2f}, p2={r['p_setback_sq']:.2e}")
    pd.DataFrame(et_rows).to_csv(ROB / "05_event_type_split.csv", index=False)

    # ------------------------------------------------------------------
    # 5. User fixed-effect LPM
    # ------------------------------------------------------------------
    log("\n[5] User fixed-effect LPM")
    fe_rows = []
    for y in outcome_cols:
        r = within_user_lpm(df, y, f"user_FE_LPM_{y}")
        if r is not None:
            fe_rows.append(r)
            log(f"  {y}: n={r['n']}, users={r['users']}, opt={r['implied_optimum_F']:.2f}, p2={r['p_setback_sq']:.2e}")
    pd.DataFrame(fe_rows).to_csv(ROB / "06_user_fixed_effect_lpm.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Comfort mechanism: add comfort gap variables
    # ------------------------------------------------------------------
    log("\n[6] Comfort mechanism checks")
    comfort_rows = []
    for comfort in ["Comfort_Gap_Mean", "Comfort_Gap_Max", "Temp_Rise"]:
        if comfort in df.columns:
            df[comfort] = safe_num(df[comfort])
            r = fit_logit(df, primary, f"main_plus_{comfort}", include_comfort=comfort)
            if r is not None:
                r["comfort_added"] = comfort
                comfort_rows.append(r)
                log(f"  + {comfort:>18s}: opt={r['implied_optimum_F']:.2f}, p2={r['p_setback_sq']:.2e}")
    pd.DataFrame(comfort_rows).to_csv(ROB / "07_comfort_mechanism.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Persistence robustness
    # ------------------------------------------------------------------
    log("\n[7] Persistence robustness")
    pers_all = []
    for y in outcome_cols:
        p = persistence_models(df, y)
        if len(p):
            pers_all.append(p)
            log(f"  {y}: persistence rows={len(p)}")
    if pers_all:
        pd.concat(pers_all, ignore_index=True).to_csv(ROB / "08_persistence_robustness.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Simple figures
    # ------------------------------------------------------------------
    log("\n[8] Figures")
    try:
        # Raw outcome sensitivity U-shape
        bins = pd.read_csv(ROB / "01_raw_setback_bins_by_outcome.csv")
        plt.figure(figsize=(7.2, 4.6))
        for y, sub in bins.groupby("outcome"):
            x = sub["SB_bin"].astype(str)
            plt.plot(x, sub["opt_out"] * 100, marker="o", label=y)
        plt.xlabel("Setback amplitude bin (°F)")
        plt.ylabel("Opt-out rate (%)")
        plt.title("Setback dose-response under alternative opt-out definitions")
        plt.grid(axis="y", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / "fig_robust_outcome_definitions.png", dpi=200)
        plt.close()

        # Leave-one-state-out optimum
        if len(loo):
            plotdf = loo.sort_values("implied_optimum_F")
            plt.figure(figsize=(7.2, 4.6))
            plt.scatter(plotdf["dropped_state"], plotdf["implied_optimum_F"])
            plt.axhline(sens.loc[sens["outcome"] == primary, "implied_optimum_F"].iloc[0], linestyle="--", linewidth=1)
            plt.ylabel("Implied optimum setback (°F)")
            plt.xlabel("Excluded state")
            plt.title("Leave-one-state-out robustness")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG / "fig_leave_one_state_out.png", dpi=200)
            plt.close()

        # User FE vs pooled
        if len(fe_rows):
            fe = pd.DataFrame(fe_rows)
            merged = pd.concat([
                sens.assign(kind="pooled_logit"),
                fe.assign(kind="user_FE_LPM"),
            ], ignore_index=True)
            plt.figure(figsize=(7.2, 4.6))
            for kind, sub in merged.groupby("kind"):
                plt.scatter(sub["outcome"], sub["implied_optimum_F"], label=kind)
            plt.axhspan(2, 4, alpha=0.1)
            plt.ylabel("Implied optimum setback (°F)")
            plt.xlabel("Outcome definition")
            plt.xticks(rotation=20, ha="right")
            plt.title("Pooled vs within-user implied optima")
            plt.grid(axis="y", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG / "fig_pooled_vs_user_fe.png", dpi=200)
            plt.close()
    except Exception as e:
        log(f"  Figure generation failed: {type(e).__name__}: {e}")

    # Final concise report
    log("\nDone. Outputs written to paper1_out/robustness/")
    (ROB / "robustness_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
