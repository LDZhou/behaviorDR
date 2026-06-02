"""
Patch diagnostics for behavior-aware DR model.
Fixes:
  1) adjusted categorical opt-out model using statsmodels.api
  2) persistence model with event controls, avoiding previous weights-length bug
  3) additional delivered-flexibility and policy sanity diagnostics

Run in the project directory containing dr_sessions.csv:
    python 12_behavior_model_diagnostics_fix.py

Outputs:
    behavior_model_out/02_adjusted_categorical_optout_fixed.csv
    behavior_model_out/03d_persistence_models_fixed.csv
    behavior_model_out/08_nominal_vs_delivered_by_bin.csv
    behavior_model_out/figs/fig2_adjusted_categorical_optout_fixed.png
"""
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = Path("behavior_model_out")
FIG = OUT / "figs"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

print("="*78)
print("Behavior model diagnostics patch")
print("="*78)

# -----------------------------
# Load and standardize
# -----------------------------
df = pd.read_csv("dr_sessions.csv")
print(f"Loaded {len(df):,} sessions")

# Cooling subset and common restrictions
if "HvacMode" in df.columns:
    df = df[df["HvacMode"].astype(str).str.lower().eq("cool")].copy()

if "Setback_Amplitude_Mean" not in df.columns:
    raise ValueError("Missing Setback_Amplitude_Mean")

# Outcome preference
if "OptOut_Immediate" in df.columns:
    df["Y"] = df["OptOut_Immediate"].astype(int)
elif "Opted_Out" in df.columns:
    df["Y"] = df["Opted_Out"].astype(int)
else:
    raise ValueError("No opt-out outcome found")

# Core filters
if "N_DR_Rows" in df.columns:
    df = df[df["N_DR_Rows"] >= 3].copy()
df = df[df["Setback_Amplitude_Mean"].between(0, 6)].copy()

# Setback bins
bin_edges = [0, 1, 2, 3, 4, 6]
bin_labels = ["0-1", "1-2", "2-3", "3-4", "4-6"]
df["Setback_Bin"] = pd.cut(
    df["Setback_Amplitude_Mean"], bins=bin_edges, labels=bin_labels,
    include_lowest=True, right=True
)
df = df[df["Setback_Bin"].notna()].copy()
df["Setback_Bin"] = df["Setback_Bin"].astype(str)
df["Setback"] = df["Setback_Amplitude_Mean"].astype(float)
df["Setback_sq"] = df["Setback"] ** 2

# Controls
for col in ["Duration_Min", "Tout_onset", "CDH_during", "Hour_of_Day", "Month", "Is_Weekend"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "DR_Type" not in df.columns:
    df["DR_Type"] = "Unknown"
if "province_state" not in df.columns:
    df["province_state"] = "Unknown"

# Drop tiny states for FE stability in diagnostic GLM
state_counts = df["province_state"].value_counts()
valid_states = state_counts[state_counts >= 100].index
df_fe = df[df["province_state"].isin(valid_states)].copy()
print(f"Working cooling sample: {len(df):,}; FE sample states>=100: {len(df_fe):,}, states={df_fe['province_state'].nunique()}")

# -----------------------------
# 1) Adjusted categorical model
# -----------------------------
print("\n[1] Adjusted categorical opt-out model")
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    need = ["Y", "Setback_Bin", "Duration_Min", "Tout_onset", "CDH_during", "Hour_of_Day", "Month", "province_state", "DR_Type"]
    mdf = df_fe.dropna(subset=[c for c in need if c in df_fe.columns]).copy()
    # Use 2-3 as reference by ordering category labels
    mdf["Setback_Bin"] = pd.Categorical(mdf["Setback_Bin"], categories=bin_labels, ordered=False)
    fml = (
        'Y ~ C(Setback_Bin, Treatment(reference="2-3")) + '
        'Duration_Min + Tout_onset + CDH_during + Hour_of_Day + C(Month) + '
        'C(DR_Type) + C(province_state)'
    )
    model = smf.glm(fml, data=mdf, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["Identifier"]} if "Identifier" in mdf.columns else None
    )

    # Predict adjusted opt-out for each bin using empirical covariate distribution.
    rows = []
    for b in bin_labels:
        tmp = mdf.copy()
        tmp["Setback_Bin"] = b
        pred = model.predict(tmp)
        rows.append({
            "Setback_Bin": b,
            "adjusted_optout": float(np.mean(pred)),
            "raw_optout": float(mdf.loc[mdf["Setback_Bin"].astype(str)==b, "Y"].mean()),
            "n": int((mdf["Setback_Bin"].astype(str)==b).sum())
        })
    adj = pd.DataFrame(rows)
    adj.to_csv(OUT / "02_adjusted_categorical_optout_fixed.csv", index=False)
    print(adj.to_string(index=False))

    # Coefficient table for bin contrasts
    coef_rows = []
    for term in model.params.index:
        if "Setback_Bin" in term:
            coef_rows.append({
                "term": term,
                "coef": model.params[term],
                "odds_ratio": float(np.exp(model.params[term])),
                "pvalue": model.pvalues[term]
            })
    pd.DataFrame(coef_rows).to_csv(OUT / "02b_adjusted_categorical_coefficients_fixed.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.6, 4.2))
        x = np.arange(len(adj))
        plt.plot(x, adj["raw_optout"]*100, marker="o", label="Raw")
        plt.plot(x, adj["adjusted_optout"]*100, marker="o", label="Adjusted")
        plt.xticks(x, adj["Setback_Bin"])
        plt.ylabel("Opt-out rate (%)")
        plt.xlabel("Setback bin (°F)")
        plt.title("Adjusted opt-out by setback bin")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / "fig2_adjusted_categorical_optout_fixed.png", dpi=180)
        plt.close()
    except Exception as e:
        print(f"  plot skipped: {e}")
except Exception as e:
    print(f"  adjusted categorical model failed: {type(e).__name__}: {e}")

# -----------------------------
# 2) Persistence models fixed
# -----------------------------
print("\n[2] Persistence models with prior rate and controls")
try:
    import statsmodels.api as sm

    pdf = df.sort_values(["Identifier", "Session_Start"] if "Session_Start" in df.columns else ["Identifier"]).copy()
    pdf["Prev_OptOut"] = pdf.groupby("Identifier")["Y"].shift(1)
    pdf["Cum_OptOut_Before"] = pdf.groupby("Identifier")["Y"].cumsum() - pdf["Y"]
    pdf["Event_Index"] = pdf.groupby("Identifier").cumcount() + 1
    pdf["Prior_OptOut_Rate"] = np.where(pdf["Event_Index"] > 1, pdf["Cum_OptOut_Before"] / (pdf["Event_Index"] - 1), np.nan)
    pdf = pdf[pdf["Prev_OptOut"].notna()].copy()
    pdf["Prior_OptOut_Rate_Filled"] = pdf["Prior_OptOut_Rate"].fillna(pdf["Y"].mean())

    # Build design matrix manually to avoid formula/weight bugs.
    base_cols = ["Prev_OptOut"]
    prior_cols = ["Prev_OptOut", "Prior_OptOut_Rate_Filled"]
    control_cols = ["Prev_OptOut", "Prior_OptOut_Rate_Filled", "Setback", "Setback_sq"]
    for c in ["Duration_Min", "Tout_onset", "CDH_during", "Hour_of_Day", "Is_Weekend"]:
        if c in pdf.columns:
            control_cols.append(c)
    if "Month" in pdf.columns:
        month_dum = pd.get_dummies(pdf["Month"].astype("Int64").astype(str), prefix="Month", drop_first=True)
    else:
        month_dum = pd.DataFrame(index=pdf.index)
    if "DR_Type" in pdf.columns:
        dr_dum = pd.get_dummies(pdf["DR_Type"].astype(str), prefix="DR", drop_first=True)
    else:
        dr_dum = pd.DataFrame(index=pdf.index)

    specs = {
        "prev_only": (base_cols, None),
        "prev_plus_prior": (prior_cols, None),
        "prev_prior_event_controls": (control_cols, [month_dum, dr_dum]),
    }
    out_rows = []
    for name, (cols, extra_dfs) in specs.items():
        mdf = pdf[["Y", "Identifier"] + cols].copy()
        X_parts = [mdf[cols].apply(pd.to_numeric, errors="coerce")]
        if extra_dfs is not None:
            for edf in extra_dfs:
                X_parts.append(edf.loc[mdf.index])
        X = pd.concat(X_parts, axis=1)
        tmp = pd.concat([mdf[["Y", "Identifier"]], X], axis=1).dropna()
        y = tmp["Y"].astype(float)
        X = tmp.drop(columns=["Y", "Identifier"]).astype(float)
        # Remove constant or all-zero columns
        keep = X.columns[X.std(axis=0) > 1e-12]
        X = X[keep]
        X = sm.add_constant(X, has_constant="add")
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": tmp["Identifier"]}
        )
        for term in ["Prev_OptOut", "Prior_OptOut_Rate_Filled", "Setback", "Setback_sq"]:
            if term in res.params.index:
                out_rows.append({
                    "model": name,
                    "term": term,
                    "coef": res.params[term],
                    "odds_ratio": float(np.exp(res.params[term])),
                    "pvalue": res.pvalues[term],
                    "n": int(len(tmp)),
                    "users": int(tmp["Identifier"].nunique())
                })
        print(f"  {name}: succeeded, n={len(tmp):,}, users={tmp['Identifier'].nunique():,}")
    pd.DataFrame(out_rows).to_csv(OUT / "03d_persistence_models_fixed.csv", index=False)
    print(pd.DataFrame(out_rows).to_string(index=False))
except Exception as e:
    print(f"  persistence fixed models failed: {type(e).__name__}: {e}")

# -----------------------------
# 3) Nominal vs delivered by bin
# -----------------------------
print("\n[3] Nominal vs delivered flexibility by bin")
if "Cool_Reduction_Frac" in df.columns:
    df["Delivered_Flex_Observed"] = df["Cool_Reduction_Frac"] * (1 - df["Y"])
    summ = df.groupby("Setback_Bin", observed=True).agg(
        n=("Y", "count"),
        optout=("Y", "mean"),
        nominal_reduction=("Cool_Reduction_Frac", "mean"),
        delivered_flex=("Delivered_Flex_Observed", "mean"),
        acceptance=("Y", lambda s: 1 - s.mean()),
        mean_setback=("Setback", "mean")
    ).reset_index()
    summ["behavior_discount"] = summ["delivered_flex"] / summ["nominal_reduction"]
    summ.to_csv(OUT / "08_nominal_vs_delivered_by_bin.csv", index=False)
    print(summ.to_string(index=False))
else:
    print("  Cool_Reduction_Frac missing")

print("\nDone. Outputs written to", OUT.resolve())
