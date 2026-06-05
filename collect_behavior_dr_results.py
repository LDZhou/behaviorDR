#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_behavior_dr_results.py

Collects finished behaviorDR experiment outputs into a compact bundle for paper writing
and figure generation.

Expected input folder:
    behavior_model_out/

Expected files are produced by behavior_model_main.py, such as:
    01_setback_bin_composition_delivered_flex.csv
    02_adjusted_categorical_optout.csv
    03_persistence_transition.csv
    04_acceptance_model_metrics.csv
    04c_reliability_score_deciles.csv
    05_reduction_model_metrics.csv
    06_policy_benchmark_predicted_values.csv
    99_session_behavior_scores.csv
    behavior_model_report.txt
    figs/*.png

Run:
    python collect_behavior_dr_results.py

Optional:
    python collect_behavior_dr_results.py --out behavior_model_out
    python collect_behavior_dr_results.py --paper-title "Behavior-Aware Delivered Flexibility in Residential Thermostat Demand Response"

Outputs:
    paper_result_summary.md
    paper_result_summary.json
    paper_tables/
    paper_figs_existing/
    behaviorDR_paper_results_bundle.zip

Send the zip or paper_result_summary.md to ChatGPT.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            return None
    return None


def safe_float(x: Any, digits: int = 4) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return round(float(x), digits)
    except Exception:
        return None


def pct(x: Any, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{100 * float(x):.{digits}f}%"
    except Exception:
        return "NA"


def num(x: Any, digits: int = 3) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"


def md_table(df: Optional[pd.DataFrame], max_rows: int = 20, floatfmt: str = ".4f") -> str:
    if df is None or len(df) == 0:
        return "_Missing or empty._"
    d = df.copy().head(max_rows)
    # Avoid giant object reprs
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    return d.to_markdown(index=False)


def copy_if_exists(src: Path, dst_dir: Path) -> Optional[Path]:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return dst
    return None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def best_oof_row(metrics: Optional[pd.DataFrame], model_col: str = "model") -> Optional[pd.Series]:
    if metrics is None or len(metrics) == 0:
        return None
    if "fold" in metrics.columns:
        oof = metrics[metrics["fold"].astype(str).str.upper() == "OOF"].copy()
        if len(oof):
            # Prefer full model if present
            if model_col in oof.columns and (oof[model_col].astype(str) == "M3_full_plus_history").any():
                return oof[oof[model_col].astype(str) == "M3_full_plus_history"].iloc[0]
            # Else choose highest auc if present
            if "auc" in oof.columns:
                return oof.sort_values("auc", ascending=False).iloc[0]
            return oof.iloc[0]
    return metrics.iloc[0]


# -----------------------------
# Core collection
# -----------------------------
def collect_results(out_dir: Path, paper_title: str) -> Dict[str, Any]:
    figs_dir = out_dir / "figs"

    files = {
        "setback": out_dir / "01_setback_bin_composition_delivered_flex.csv",
        "regime": out_dir / "01b_regime_summary.csv",
        "adjusted_optout": out_dir / "02_adjusted_categorical_optout.csv",
        "adjusted_coef": out_dir / "02b_adjusted_categorical_coefficients.csv",
        "persistence_transition": out_dir / "03_persistence_transition.csv",
        "persistence_streak": out_dir / "03b_persistence_streak.csv",
        "risk_states": out_dir / "03c_behavior_risk_states.csv",
        "persistence_models": out_dir / "03d_persistence_models.csv",
        "acceptance_metrics": out_dir / "04_acceptance_model_metrics.csv",
        "acceptance_importance": out_dir / "04b_acceptance_feature_importance.csv",
        "reliability_deciles": out_dir / "04c_reliability_score_deciles.csv",
        "reduction_metrics": out_dir / "05_reduction_model_metrics.csv",
        "arm_values": out_dir / "06_predicted_arm_values.csv",
        "policy_values": out_dir / "06_policy_benchmark_predicted_values.csv",
        "policy_arm_dist": out_dir / "06b_policy_selected_arm_distribution.csv",
        "low_setback_check": out_dir / "07_low_setback_vs_reference_diagnostics.csv",
        "session_scores": out_dir / "99_session_behavior_scores.csv",
    }

    dfs = {k: read_csv_if_exists(v) for k, v in files.items()}

    summary: Dict[str, Any] = {
        "paper_title": paper_title,
        "input_output_folder": str(out_dir.resolve()),
        "available_files": {k: v.exists() for k, v in files.items()},
        "key_numbers": {},
        "tables": {},
        "figure_files_existing": [],
        "warnings": [],
    }

    # Basic dataset numbers from 99_session_behavior_scores.csv if available
    session = dfs["session_scores"]
    if session is not None:
        kn = summary["key_numbers"]
        kn["n_sessions_scored"] = int(len(session))
        if "Identifier" in session.columns:
            kn["n_users_scored"] = int(session["Identifier"].nunique())
        if "Y_immediate" in session.columns:
            kn["overall_optout_rate"] = safe_float(session["Y_immediate"].mean(), 4)
            kn["overall_acceptance_rate"] = safe_float(1 - session["Y_immediate"].mean(), 4)
        if "Setback" in session.columns:
            kn["mean_setback"] = safe_float(session["Setback"].mean(), 3)
        if "Delivered_Flex_Observed" in session.columns:
            kn["mean_observed_delivered_flex"] = safe_float(session["Delivered_Flex_Observed"].mean(), 4)
        if "Cool_Reduction_Frac" in session.columns:
            kn["mean_cooling_reduction"] = safe_float(session["Cool_Reduction_Frac"].mean(), 4)

    # Setback results
    setback = dfs["setback"]
    if setback is not None:
        cols = [c for c in ["Setback_Bin", "n", "users", "optout", "cool_reduction", "delivered_flex",
                            "mean_Tout_onset", "mean_Duration_Min", "top_state", "top_state_share", "CS_DR_share"]
                if c in setback.columns]
        summary["tables"]["setback_core"] = setback[cols].to_dict(orient="records")

        # Best delivered flex bin
        if {"Setback_Bin", "delivered_flex"}.issubset(setback.columns):
            tmp = setback.dropna(subset=["delivered_flex"])
            if len(tmp):
                r = tmp.sort_values("delivered_flex", ascending=False).iloc[0]
                summary["key_numbers"]["best_delivered_flex_bin"] = str(r["Setback_Bin"])
                summary["key_numbers"]["best_delivered_flex_value"] = safe_float(r["delivered_flex"], 4)

        if {"Setback_Bin", "optout"}.issubset(setback.columns):
            tmp = setback.dropna(subset=["optout"])
            if len(tmp):
                rmin = tmp.sort_values("optout", ascending=True).iloc[0]
                rmax = tmp.sort_values("optout", ascending=False).iloc[0]
                summary["key_numbers"]["lowest_raw_optout_bin"] = str(rmin["Setback_Bin"])
                summary["key_numbers"]["lowest_raw_optout"] = safe_float(rmin["optout"], 4)
                summary["key_numbers"]["highest_raw_optout_bin"] = str(rmax["Setback_Bin"])
                summary["key_numbers"]["highest_raw_optout"] = safe_float(rmax["optout"], 4)

    # Adjusted opt-out
    adjusted = dfs["adjusted_optout"]
    if adjusted is not None:
        summary["tables"]["adjusted_optout"] = adjusted.to_dict(orient="records")
        if {"Setback_Bin", "adjusted_optout"}.issubset(adjusted.columns):
            tmp = adjusted.dropna(subset=["adjusted_optout"])
            if len(tmp):
                rmin = tmp.sort_values("adjusted_optout", ascending=True).iloc[0]
                rmax = tmp.sort_values("adjusted_optout", ascending=False).iloc[0]
                summary["key_numbers"]["lowest_adjusted_optout_bin"] = str(rmin["Setback_Bin"])
                summary["key_numbers"]["lowest_adjusted_optout"] = safe_float(rmin["adjusted_optout"], 4)
                summary["key_numbers"]["highest_adjusted_optout_bin"] = str(rmax["Setback_Bin"])
                summary["key_numbers"]["highest_adjusted_optout"] = safe_float(rmax["adjusted_optout"], 4)

    # Persistence
    trans = dfs["persistence_transition"]
    if trans is not None:
        summary["tables"]["persistence_transition"] = trans.to_dict(orient="records")
        if {"Prev_Status", "current_optout"}.issubset(trans.columns):
            for _, r in trans.iterrows():
                summary["key_numbers"][f"current_optout_after_{r['Prev_Status']}"] = safe_float(r["current_optout"], 4)

    streak = dfs["persistence_streak"]
    if streak is not None:
        summary["tables"]["persistence_streak"] = streak.to_dict(orient="records")
        if {"Streak_Group", "current_optout"}.issubset(streak.columns):
            tmp = streak.dropna(subset=["current_optout"])
            if len(tmp):
                rmin = tmp.sort_values("current_optout", ascending=True).iloc[0]
                rmax = tmp.sort_values("current_optout", ascending=False).iloc[0]
                summary["key_numbers"]["lowest_streak_optout_group"] = str(rmin["Streak_Group"])
                summary["key_numbers"]["lowest_streak_optout"] = safe_float(rmin["current_optout"], 4)
                summary["key_numbers"]["highest_streak_optout_group"] = str(rmax["Streak_Group"])
                summary["key_numbers"]["highest_streak_optout"] = safe_float(rmax["current_optout"], 4)

    risk = dfs["risk_states"]
    if risk is not None:
        summary["tables"]["risk_states"] = risk.to_dict(orient="records")
        if {"Risk_State", "delivered_flex"}.issubset(risk.columns):
            tmp = risk.dropna(subset=["delivered_flex"])
            if len(tmp):
                rmin = tmp.sort_values("delivered_flex", ascending=True).iloc[0]
                rmax = tmp.sort_values("delivered_flex", ascending=False).iloc[0]
                summary["key_numbers"]["lowest_risk_state_by_delivered_flex"] = str(rmin["Risk_State"])
                summary["key_numbers"]["lowest_risk_state_delivered_flex"] = safe_float(rmin["delivered_flex"], 4)
                summary["key_numbers"]["highest_risk_state_by_delivered_flex"] = str(rmax["Risk_State"])
                summary["key_numbers"]["highest_risk_state_delivered_flex"] = safe_float(rmax["delivered_flex"], 4)
                if rmin["delivered_flex"] and not pd.isna(rmin["delivered_flex"]) and rmin["delivered_flex"] != 0:
                    summary["key_numbers"]["risk_state_delivered_flex_ratio_high_over_low"] = safe_float(
                        rmax["delivered_flex"] / rmin["delivered_flex"], 3
                    )

    # Acceptance metrics
    acc = dfs["acceptance_metrics"]
    if acc is not None:
        oof = acc[acc["fold"].astype(str).str.upper() == "OOF"].copy() if "fold" in acc.columns else acc.copy()
        summary["tables"]["acceptance_oof_metrics"] = oof.to_dict(orient="records")
        row = best_oof_row(acc)
        if row is not None:
            for c in ["model", "auc", "pr_auc", "brier", "logloss", "mean_pred", "mean_y", "n_features"]:
                if c in row.index:
                    val = row[c]
                    summary["key_numbers"][f"acceptance_full_{c}"] = safe_float(val, 4) if c != "model" else str(val)

        # Marginal AUC gain table
        if {"model", "auc", "fold"}.issubset(acc.columns):
            o = acc[acc["fold"].astype(str).str.upper() == "OOF"].copy()
            order = ["M0_weather_time", "M1_plus_setback", "M2_plus_building_baseline", "M3_full_plus_history"]
            o["_order"] = o["model"].map({m: i for i, m in enumerate(order)})
            o = o.sort_values("_order")
            gains = []
            prev_auc = None
            for _, r in o.iterrows():
                auc = float(r["auc"])
                gains.append({
                    "model": r["model"],
                    "auc": auc,
                    "delta_auc_from_previous": None if prev_auc is None else auc - prev_auc,
                    "n_features": int(r["n_features"]) if "n_features" in r and not pd.isna(r["n_features"]) else None
                })
                prev_auc = auc
            summary["tables"]["acceptance_ablation_gains"] = gains

    # Reliability deciles
    dec = dfs["reliability_deciles"]
    if dec is not None:
        summary["tables"]["reliability_deciles"] = dec.to_dict(orient="records")
        if {"pred_accept", "observed_accept", "delivered_flex"}.issubset(dec.columns):
            # Spread from highest to lowest predicted reliability decile
            tmp = dec.dropna(subset=["pred_accept", "observed_accept"]).copy()
            if len(tmp) >= 2:
                low = tmp.sort_values("pred_accept").iloc[0]
                high = tmp.sort_values("pred_accept").iloc[-1]
                summary["key_numbers"]["lowest_decile_pred_accept"] = safe_float(low["pred_accept"], 4)
                summary["key_numbers"]["lowest_decile_observed_accept"] = safe_float(low["observed_accept"], 4)
                summary["key_numbers"]["highest_decile_pred_accept"] = safe_float(high["pred_accept"], 4)
                summary["key_numbers"]["highest_decile_observed_accept"] = safe_float(high["observed_accept"], 4)
                if "delivered_flex" in tmp.columns:
                    summary["key_numbers"]["lowest_decile_delivered_flex"] = safe_float(low["delivered_flex"], 4)
                    summary["key_numbers"]["highest_decile_delivered_flex"] = safe_float(high["delivered_flex"], 4)

    # Reduction metrics
    red = dfs["reduction_metrics"]
    if red is not None:
        summary["tables"]["reduction_metrics"] = red.to_dict(orient="records")
        if "fold" in red.columns:
            o = red[red["fold"].astype(str).str.upper() == "OOF"]
            if len(o):
                r = o.iloc[0]
                for c in ["rmse", "mae", "r2"]:
                    if c in r.index:
                        summary["key_numbers"][f"reduction_oof_{c}"] = safe_float(r[c], 4)

    # Policy values
    pol = dfs["policy_values"]
    if pol is not None:
        summary["tables"]["policy_values"] = pol.to_dict(orient="records")
        if {"policy", "pred_delivered"}.issubset(pol.columns):
            tmp = pol.dropna(subset=["pred_delivered"])
            if len(tmp):
                best = tmp.sort_values("pred_delivered", ascending=False).iloc[0]
                base_candidates = tmp[tmp["policy"].astype(str).str.startswith("uniform_")]
                if len(base_candidates):
                    # choose best uniform baseline
                    base = base_candidates.sort_values("pred_delivered", ascending=False).iloc[0]
                    gain = best["pred_delivered"] - base["pred_delivered"]
                    gain_pct = gain / base["pred_delivered"] if base["pred_delivered"] != 0 else np.nan
                    summary["key_numbers"]["best_policy"] = str(best["policy"])
                    summary["key_numbers"]["best_policy_pred_delivered"] = safe_float(best["pred_delivered"], 4)
                    summary["key_numbers"]["best_uniform_policy"] = str(base["policy"])
                    summary["key_numbers"]["best_uniform_pred_delivered"] = safe_float(base["pred_delivered"], 4)
                    summary["key_numbers"]["best_policy_gain_vs_best_uniform_abs"] = safe_float(gain, 4)
                    summary["key_numbers"]["best_policy_gain_vs_best_uniform_pct"] = safe_float(gain_pct, 4)

    # Low-setback anomaly check
    low = dfs["low_setback_check"]
    if low is not None:
        summary["tables"]["low_setback_check"] = low.to_dict(orient="records")

    # Existing figures
    if figs_dir.exists():
        for p in sorted(figs_dir.glob("*")):
            if p.suffix.lower() in [".png", ".pdf", ".jpg", ".jpeg", ".svg"]:
                summary["figure_files_existing"].append(str(p))

    # Warnings / sanity
    required = ["setback", "acceptance_metrics", "reliability_deciles", "session_scores"]
    for k in required:
        if dfs[k] is None:
            summary["warnings"].append(f"Missing recommended output: {files[k].name}")

    if "acceptance_full_auc" in summary["key_numbers"]:
        if summary["key_numbers"]["acceptance_full_auc"] < 0.75:
            summary["warnings"].append("Acceptance model AUC is below 0.75; be careful with strong predictability claims.")

    return summary


def write_outputs(out_dir: Path, summary: Dict[str, Any]) -> None:
    bundle_dir = out_dir / "paper_bundle"
    tables_dir = bundle_dir / "paper_tables"
    figs_existing_dir = bundle_dir / "paper_figs_existing"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_existing_dir.mkdir(parents=True, exist_ok=True)

    # Copy all csv tables to bundle
    for p in sorted(out_dir.glob("*.csv")):
        shutil.copy2(p, tables_dir / p.name)

    # Copy report if exists
    report = out_dir / "behavior_model_report.txt"
    if report.exists():
        shutil.copy2(report, bundle_dir / report.name)

    # Copy existing figures
    figs_dir = out_dir / "figs"
    if figs_dir.exists():
        for p in sorted(figs_dir.glob("*")):
            if p.suffix.lower() in [".png", ".pdf", ".jpg", ".jpeg", ".svg"]:
                shutil.copy2(p, figs_existing_dir / p.name)

    # Write JSON
    json_path = bundle_dir / "paper_result_summary.json"
    write_json(json_path, summary)

    # Write markdown summary
    md = make_markdown(out_dir, summary)
    md_path = bundle_dir / "paper_result_summary.md"
    md_path.write_text(md, encoding="utf-8")

    # Also write summary at root out_dir for quick upload/paste
    (out_dir / "paper_result_summary.md").write_text(md, encoding="utf-8")
    write_json(out_dir / "paper_result_summary.json", summary)

    # Zip bundle
    zip_path = out_dir / "behaviorDR_paper_results_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in bundle_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(bundle_dir))

    print("\n[DONE] Result bundle created.")
    print(f"Markdown summary: {out_dir / 'paper_result_summary.md'}")
    print(f"JSON summary:     {out_dir / 'paper_result_summary.json'}")
    print(f"ZIP bundle:       {zip_path}")
    print("\nSend me either:")
    print("  1) behaviorDR_paper_results_bundle.zip")
    print("or")
    print("  2) paper_result_summary.md + relevant CSVs/figs")


def make_markdown(out_dir: Path, summary: Dict[str, Any]) -> str:
    kn = summary["key_numbers"]

    def kv_line(key: str, label: str, formatter=None) -> str:
        if key not in kn or kn[key] is None:
            return f"- **{label}:** NA"
        v = kn[key]
        if formatter == "pct":
            return f"- **{label}:** {pct(v)}"
        if formatter == "num":
            return f"- **{label}:** {num(v)}"
        return f"- **{label}:** {v}"

    lines: List[str] = []

    lines.append(f"# Paper Result Summary: {summary['paper_title']}\n")
    lines.append(f"Output folder: `{summary['input_output_folder']}`\n")

    lines.append("## 1. Key Numbers\n")
    lines.append(kv_line("n_sessions_scored", "Scored sessions"))
    lines.append(kv_line("n_users_scored", "Scored users"))
    lines.append(kv_line("overall_optout_rate", "Overall opt-out rate", "pct"))
    lines.append(kv_line("overall_acceptance_rate", "Overall acceptance rate", "pct"))
    lines.append(kv_line("mean_setback", "Mean setback", "num"))
    lines.append(kv_line("mean_cooling_reduction", "Mean cooling reduction", "num"))
    lines.append(kv_line("mean_observed_delivered_flex", "Mean observed delivered flexibility", "num"))
    lines.append("")

    lines.append("## 2. Setback and Delivered Flexibility\n")
    lines.append(kv_line("best_delivered_flex_bin", "Best delivered-flex setback bin"))
    lines.append(kv_line("best_delivered_flex_value", "Best delivered-flex value", "num"))
    lines.append(kv_line("lowest_raw_optout_bin", "Lowest raw opt-out bin"))
    lines.append(kv_line("lowest_raw_optout", "Lowest raw opt-out", "pct"))
    lines.append(kv_line("highest_raw_optout_bin", "Highest raw opt-out bin"))
    lines.append(kv_line("highest_raw_optout", "Highest raw opt-out", "pct"))
    lines.append(kv_line("lowest_adjusted_optout_bin", "Lowest adjusted opt-out bin"))
    lines.append(kv_line("lowest_adjusted_optout", "Lowest adjusted opt-out", "pct"))
    lines.append(kv_line("highest_adjusted_optout_bin", "Highest adjusted opt-out bin"))
    lines.append(kv_line("highest_adjusted_optout", "Highest adjusted opt-out", "pct"))
    lines.append("")

    if "setback_core" in summary["tables"]:
        lines.append("### Setback core table\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["setback_core"])))
        lines.append("")
    if "adjusted_optout" in summary["tables"]:
        lines.append("### Adjusted opt-out by setback\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["adjusted_optout"])))
        lines.append("")

    lines.append("## 3. Behavioral Persistence\n")
    for key, label in [
        ("current_optout_after_previous_stay", "Current opt-out after previous stay"),
        ("current_optout_after_previous_optout", "Current opt-out after previous opt-out"),
        ("lowest_streak_optout_group", "Lowest streak opt-out group"),
        ("lowest_streak_optout", "Lowest streak opt-out"),
        ("highest_streak_optout_group", "Highest streak opt-out group"),
        ("highest_streak_optout", "Highest streak opt-out"),
    ]:
        formatter = "pct" if "optout" in key and not key.endswith("group") else None
        lines.append(kv_line(key, label, formatter))
    lines.append("")
    if "persistence_transition" in summary["tables"]:
        lines.append("### Previous-event transition\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["persistence_transition"])))
        lines.append("")
    if "persistence_streak" in summary["tables"]:
        lines.append("### Streak table\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["persistence_streak"])))
        lines.append("")
    if "risk_states" in summary["tables"]:
        lines.append("### Risk states\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["risk_states"])))
        lines.append("")

    lines.append("## 4. Acceptance Model\n")
    lines.append(kv_line("acceptance_full_model", "Full model"))
    lines.append(kv_line("acceptance_full_auc", "Full model AUC", "num"))
    lines.append(kv_line("acceptance_full_pr_auc", "Full model PR-AUC", "num"))
    lines.append(kv_line("acceptance_full_brier", "Full model Brier", "num"))
    lines.append(kv_line("acceptance_full_logloss", "Full model log loss", "num"))
    lines.append(kv_line("acceptance_full_mean_pred", "Mean predicted opt-out", "pct"))
    lines.append(kv_line("acceptance_full_mean_y", "Mean observed opt-out", "pct"))
    lines.append("")
    if "acceptance_ablation_gains" in summary["tables"]:
        lines.append("### Acceptance ablation gains\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["acceptance_ablation_gains"])))
        lines.append("")
    if "acceptance_oof_metrics" in summary["tables"]:
        lines.append("### Acceptance OOF metrics\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["acceptance_oof_metrics"])))
        lines.append("")

    lines.append("## 5. Reliability Deciles\n")
    lines.append(kv_line("lowest_decile_pred_accept", "Lowest decile predicted acceptance", "pct"))
    lines.append(kv_line("lowest_decile_observed_accept", "Lowest decile observed acceptance", "pct"))
    lines.append(kv_line("highest_decile_pred_accept", "Highest decile predicted acceptance", "pct"))
    lines.append(kv_line("highest_decile_observed_accept", "Highest decile observed acceptance", "pct"))
    lines.append(kv_line("lowest_decile_delivered_flex", "Lowest decile delivered flexibility", "num"))
    lines.append(kv_line("highest_decile_delivered_flex", "Highest decile delivered flexibility", "num"))
    lines.append("")
    if "reliability_deciles" in summary["tables"]:
        lines.append("### Reliability decile table\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["reliability_deciles"])))
        lines.append("")

    lines.append("## 6. Technical Reduction Model\n")
    lines.append(kv_line("reduction_oof_rmse", "Reduction OOF RMSE", "num"))
    lines.append(kv_line("reduction_oof_mae", "Reduction OOF MAE", "num"))
    lines.append(kv_line("reduction_oof_r2", "Reduction OOF R2", "num"))
    lines.append("")
    if "reduction_metrics" in summary["tables"]:
        lines.append("### Reduction metrics\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["reduction_metrics"])))
        lines.append("")

    lines.append("## 7. Policy Benchmark\n")
    lines.append(kv_line("best_policy", "Best policy"))
    lines.append(kv_line("best_policy_pred_delivered", "Best policy predicted delivered flexibility", "num"))
    lines.append(kv_line("best_uniform_policy", "Best uniform policy"))
    lines.append(kv_line("best_uniform_pred_delivered", "Best uniform predicted delivered flexibility", "num"))
    lines.append(kv_line("best_policy_gain_vs_best_uniform_abs", "Best policy gain vs best uniform, absolute", "num"))
    lines.append(kv_line("best_policy_gain_vs_best_uniform_pct", "Best policy gain vs best uniform, percent", "pct"))
    lines.append("")
    if "policy_values" in summary["tables"]:
        lines.append("### Policy values\n")
        lines.append(md_table(pd.DataFrame(summary["tables"]["policy_values"])))
        lines.append("")

    lines.append("## 8. Low-Setback Anomaly / Robustness Check\n")
    if "low_setback_check" in summary["tables"]:
        lines.append(md_table(pd.DataFrame(summary["tables"]["low_setback_check"]), max_rows=50))
        lines.append("")
    else:
        lines.append("_Missing or not generated._\n")

    lines.append("## 9. Existing Figure Files\n")
    if summary["figure_files_existing"]:
        for p in summary["figure_files_existing"]:
            lines.append(f"- `{p}`")
    else:
        lines.append("_No figure files found._")
    lines.append("")

    lines.append("## 10. Warnings\n")
    if summary["warnings"]:
        for w in summary["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## 11. Suggested Paper Figures to Generate Next\n")
    lines.extend([
        "1. Framework figure: control action → technical reduction × acceptance probability → delivered flexibility.",
        "2. Setback figure: raw/adjusted opt-out plus nominal cooling reduction and delivered flexibility by setback bin.",
        "3. Behavioral persistence figure: previous opt-out / streak / two-step history transition.",
        "4. Acceptance model figure: ablation AUC + Brier or calibration.",
        "5. Reliability decile figure: predicted vs observed acceptance and delivered flexibility by reliability decile.",
        "6. Same-setback operational figure: delivered flexibility by behavioral risk group within the 3--4°F setback bin.",
        "7. Policy benchmark figure: uniform policies vs rule behavior-aware vs model-targeted policy, clearly labeled as model-based simulation if applicable.",
    ])
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="behavior_model_out",
                        help="Output directory from behavior_model_main.py")
    parser.add_argument("--paper-title", type=str,
                        default="Behavior-Aware Delivered Flexibility in Residential Thermostat Demand Response")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.exists():
        raise FileNotFoundError(
            f"Cannot find output folder: {out_dir}\n"
            f"Run this script in the project folder or pass --out /path/to/behavior_model_out"
        )

    summary = collect_results(out_dir, args.paper_title)
    write_outputs(out_dir, summary)


if __name__ == "__main__":
    main()
