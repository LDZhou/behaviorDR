#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack_narrative_results.py
=============================================================================
只为「Behavior-Aware Delivered Flexibility」这篇叙事主干打包所需结果文件。

- 只收这篇故事真正用到的表 / 图（按论文章节分组）。
- 故意【不收】已废弃的 policy benchmark（06_* / 06b_*）和 CATE（paper2_out）。
- 终端会打印每个章节的 [FOUND]/[MISSING]，缺文件时提示该跑哪个脚本。
- 会另外生成 dr_sessions_preview.json（列名 + 行数 + opt-out 率 + setback 描述），
  用来对齐样本量分歧（叙事 26,293 / 5,764  vs  report 26,462 / 5,812），
  但【不】把完整的 dr_sessions.csv 打进包。

用法（在 ecobee 项目根目录下，即三个 *_out 文件夹的上级目录运行）：
    python pack_narrative_results.py
    python pack_narrative_results.py --root /nfs/hpc/share/zhouling/research/ecobee
    python pack_narrative_results.py --include-optional   # 连可选诊断文件一起打

产出：
    narrative_results_bundle.zip
把这个 zip 发回来即可。
=============================================================================
"""

import argparse
import json
import platform
import zipfile
from datetime import datetime
from pathlib import Path

# 单文件大小上限（MB）。预测明细 / 匹配对文件可能偏大，超过就跳过并在报告里标注。
MAX_FILE_MB = 50

# 三个生成脚本 -> 输出目录，用于「整个文件夹缺失」时的提示
DIR_SOURCE = {
    "behavior_model_out":        "behavior_model_main.py (+ behavior_model_diagnostics.py 补丁)",
    "remaining_experiments_out": "behavior_model_validation.py",
    "persistence_sanity_out":    "behavior_model_persistence_checks.py",
}

# -----------------------------------------------------------------------------
# 文件清单：section -> [(相对路径, 级别)]
#   级别 core    = 叙事正文/主图直接引用，缺了那条 claim 就没数
#   级别 support = 审稿人会追问的稳健性/校验，强烈建议有
#   级别 optional= 锦上添花/明细，--include-optional 才打包
# -----------------------------------------------------------------------------
MANIFEST = {
    "00_reports (运行日志，核对数字用)": [
        ("behavior_model_out/behavior_model_report.txt", "core"),
        ("remaining_experiments_out/remaining_experiments_report.txt", "support"),
        ("persistence_sanity_out/README_results_interpretation.txt", "support"),
    ],

    "Sec3_Data (样本/分箱构成)": [
        ("behavior_model_out/01_setback_bin_composition_delivered_flex.csv", "core"),
        ("behavior_model_out/01b_regime_summary.csv", "core"),
        ("behavior_model_out/08_nominal_vs_delivered_by_bin.csv", "core"),  # reduction×accept 不独立的证据
        ("behavior_model_out/07_low_setback_vs_reference_diagnostics.csv", "support"),  # 0-1°F 异常诊断
    ],

    "Sec4.1_a 非线性 acceptance vs setback (adjusted, Fig.2)": [
        ("behavior_model_out/02_adjusted_categorical_optout_fixed.csv", "core"),
        ("behavior_model_out/02b_adjusted_categorical_coefficients_fixed.csv", "core"),
    ],

    "Sec4.1_b 持续性 / 两步历史 (7.8% -> 74.2%)": [
        ("behavior_model_out/03e_three_event_transition.csv", "core"),       # 两步 transition cube
        ("behavior_model_out/03_persistence_transition.csv", "support"),
        ("behavior_model_out/03b_persistence_streak.csv", "support"),
        ("behavior_model_out/03c_behavior_risk_states.csv", "support"),
        ("behavior_model_out/03d_persistence_models_fixed.csv", "support"),
    ],

    "Sec4.1_c 特征消融 (AUC 0.681/0.752/0.760/0.862)": [
        ("behavior_model_out/04_acceptance_model_metrics.csv", "core"),
        ("behavior_model_out/04b_acceptance_feature_importance.csv", "optional"),  # split-count importance，仅参考勿当主证据
    ],

    "Sec4.2 可判别 + 标定 (AUC~0.86, ECE<0.02, slope~1)  [命门]": [
        # 主脚本里的 OOF 校准（可信）
        ("behavior_model_out/04_calibration_M3_full_plus_history.csv", "core"),
        ("behavior_model_out/04_calibration_M2_plus_building_baseline.csv", "optional"),
        ("behavior_model_out/04_calibration_M1_plus_setback.csv", "optional"),
        ("behavior_model_out/04_calibration_M0_weather_time.csv", "optional"),
        # ECE / calibration slope 真正的来源（不在之前的 bundle 里）
        ("remaining_experiments_out/validation_metrics_summary.csv", "core"),
        ("remaining_experiments_out/groupkfold_reliability_deciles.csv", "core"),
        ("remaining_experiments_out/groupkfold_calibration_table.csv", "core"),
        ("persistence_sanity_out/groupkfold_metrics.csv", "support"),
        ("persistence_sanity_out/groupkfold_folds.csv", "support"),
        ("persistence_sanity_out/groupkfold_reliability_deciles.csv", "support"),
        ("persistence_sanity_out/groupkfold_calibration_table.csv", "support"),
        # 事前可用性：未来事件
        ("remaining_experiments_out/out_of_time_metrics.csv", "core"),
        ("remaining_experiments_out/out_of_time_reliability_deciles.csv", "support"),
        ("remaining_experiments_out/out_of_time_calibration_table.csv", "support"),
        ("persistence_sanity_out/out_of_time_metrics.csv", "support"),
        # 注意：主脚本的 04c_reliability_score_deciles.csv 是 in-sample 打分，故意不收，避免误用
    ],

    "Sec4.3 同 setback 不同 delivered (low 0.213 vs high 0.101 @3-4F)  [headline]": [
        ("remaining_experiments_out/same_3_4_setback_by_history_risk.csv", "core"),
        ("persistence_sanity_out/same_3_4_setback_heterogeneity.csv", "core"),
        ("remaining_experiments_out/groupkfold_behavior_accounting_by_reliability_decile.csv", "support"),
        ("persistence_sanity_out/behavior_accounting_by_reliability_decile.csv", "support"),
    ],

    "Sec4.1 持续性的因果稳健性 (审稿人必问，非因果声明)": [
        ("persistence_sanity_out/first_optout_event_study_raw.csv", "support"),
        ("persistence_sanity_out/first_optout_event_study_model.csv", "support"),
        ("persistence_sanity_out/matched_first_optout_summary.csv", "support"),
        ("persistence_sanity_out/user_fe_lag_lpm.csv", "support"),
        ("persistence_sanity_out/lead_placebo_test.csv", "support"),
        ("persistence_sanity_out/matched_first_optout_pairs.csv", "optional"),
    ],

    "Reduction 模型 (Delivered=Reduction×Accept 的 Reduction 半边)": [
        ("behavior_model_out/05_reduction_model_metrics.csv", "core"),
    ],

    "Figures (现成图，新框架图 Fig.1 仍需另画)": [
        ("behavior_model_out/figs/fig1_behavior_adjusted_flex_by_setback.png", "core"),
        ("behavior_model_out/figs/fig2_adjusted_categorical_optout_fixed.png", "core"),   # Fig.2
        ("behavior_model_out/figs/fig3_behavioral_persistence.png", "core"),
        ("remaining_experiments_out/figs/groupkfold_reliability_deciles.png", "core"),    # Fig.3 左
        ("remaining_experiments_out/figs/groupkfold_calibration.png", "core"),            # Fig.3 右
        ("remaining_experiments_out/figs/fig_same_3_4_setback_by_history_risk.png", "core"),  # Fig.4
        ("persistence_sanity_out/figs/fig_same_3_4_setback_reliability.png", "support"),
        ("persistence_sanity_out/figs/groupkfold_calibration.png", "support"),
        ("persistence_sanity_out/figs/fig_first_optout_event_study.png", "support"),
        ("persistence_sanity_out/figs/fig_matched_first_optout.png", "support"),
        ("remaining_experiments_out/figs/out_of_time_reliability_deciles.png", "optional"),
        ("remaining_experiments_out/figs/out_of_time_calibration.png", "optional"),
    ],

    # 预测明细：若想让我亲自重算 ECE/校准曲线时有用，默认仅 optional
    "Optional 预测明细 (可让我复算校准)": [
        ("remaining_experiments_out/full_model_groupkfold_predictions.csv", "optional"),
        ("remaining_experiments_out/out_of_time_predictions.csv", "optional"),
    ],
}

ZIP_NAME = "narrative_results_bundle.zip"


def human(nbytes):
    for unit in ["B", "KB", "MB"]:
        if nbytes < 1024 or unit == "MB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes/1024**(['B','KB','MB'].index(unit)):.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} MB"


def make_session_preview(root: Path):
    """生成 dr_sessions 轻量预览，用来对齐样本量分歧；找不到 pandas 或文件就跳过。"""
    csv = root / "dr_sessions.csv"
    if not csv.exists():
        return None, "dr_sessions.csv 未找到（预览跳过，不影响打包）"
    try:
        import pandas as pd
    except Exception:
        return None, "pandas 不可用（预览跳过）"

    summary = {"file": "dr_sessions.csv"}
    try:
        head = pd.read_csv(csv, nrows=5)
        summary["columns"] = list(head.columns)
        # 复刻主分析的过滤口径，看过滤后 N 到底是多少
        usecols = [c for c in ["Identifier", "HvacMode", "Setback_Amplitude_Mean",
                               "N_DR_Rows", "OptOut_Immediate", "Opted_Out"]
                   if c in head.columns]
        df = pd.read_csv(csv, usecols=usecols) if usecols else pd.read_csv(csv)
        summary["raw_n_rows"] = int(len(df))
        if "Identifier" in df.columns:
            summary["raw_n_users"] = int(df["Identifier"].nunique())

        f = df
        steps = {"raw": len(f)}
        if "HvacMode" in f.columns:
            f = f[f["HvacMode"].astype(str).str.lower().eq("cool")]
            steps["after_cool"] = len(f)
        if "Setback_Amplitude_Mean" in f.columns:
            sb = pd.to_numeric(f["Setback_Amplitude_Mean"], errors="coerce")
            f = f[sb.between(0, 6)]
            steps["after_setback_0_6"] = len(f)
        if "N_DR_Rows" in f.columns:
            f = f[pd.to_numeric(f["N_DR_Rows"], errors="coerce") >= 3]
            steps["after_NDR_ge3"] = len(f)
        summary["filter_funnel"] = steps
        summary["filtered_n_rows"] = int(len(f))
        if "Identifier" in f.columns:
            summary["filtered_n_users"] = int(f["Identifier"].nunique())
        for y in ["OptOut_Immediate", "Opted_Out"]:
            if y in f.columns:
                summary[f"{y}_mean_filtered"] = float(pd.to_numeric(f[y], errors="coerce").mean())
    except Exception as e:
        summary["preview_error"] = repr(e)

    out = root / "_dr_sessions_preview.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="项目根目录（三个 *_out 的上级），默认当前目录")
    ap.add_argument("--include-optional", action="store_true", help="把 optional 级别文件也打包")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print("=" * 78)
    print(f"项目根目录: {root}")
    print(f"打包 optional 文件: {args.include_optional}")
    print("=" * 78)

    # 整个输出文件夹是否存在
    for d, src in DIR_SOURCE.items():
        status = "OK " if (root / d).is_dir() else "缺失"
        if status == "缺失":
            print(f"  [{status}] {d}/   <- 需要先跑: {src}")
        else:
            print(f"  [{status}] {d}/")
    print()

    preview_path, preview_note = make_session_preview(root)
    if preview_note:
        print(f"  样本量预览: {preview_note}")
    else:
        print(f"  样本量预览已生成: {preview_path.name}")
    print()

    found, missing_core, missing_support, skipped_big = [], [], [], []
    manifest_records = []

    zip_path = root / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for section, items in MANIFEST.items():
            print(f"[{section}]")
            for rel, level in items:
                if level == "optional" and not args.include_optional:
                    continue
                p = root / rel
                if not p.exists():
                    tag = "MISSING"
                    print(f"   [{tag}] {rel}" + ("   <-- 关键，缺它没数" if level == "core" else ""))
                    (missing_core if level == "core" else missing_support if level == "support" else []).append(rel)
                    continue
                size_mb = p.stat().st_size / 1024**2
                if size_mb > MAX_FILE_MB:
                    print(f"   [SKIP>{MAX_FILE_MB}MB] {rel} ({size_mb:.1f} MB)")
                    skipped_big.append(rel)
                    continue
                zf.write(p, rel)
                found.append(rel)
                manifest_records.append({"path": rel, "section": section, "level": level,
                                         "size": human(p.stat().st_size)})
                print(f"   [FOUND] {rel} ({human(p.stat().st_size)})")
            print()

        if preview_path and preview_path.exists():
            zf.write(preview_path, "_dr_sessions_preview.json")
            found.append("_dr_sessions_preview.json")

        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "root": str(root),
            "python": platform.python_version(),
            "include_optional": args.include_optional,
            "n_found": len(found),
            "missing_core": missing_core,
            "missing_support": missing_support,
            "skipped_too_big": skipped_big,
            "files": manifest_records,
            "note": "为 Behavior-Aware Delivered Flexibility 叙事打包；已剔除 policy/CATE 废弃内容。",
        }
        zf.writestr("_pack_manifest.json", json.dumps(meta, indent=2, ensure_ascii=False))

    # ---- 终端汇总 ----
    print("=" * 78)
    print(f"打包完成: {zip_path}  ({human(zip_path.stat().st_size)})  共 {len(found)} 个文件")
    print("=" * 78)
    if missing_core:
        print("\n!!! CORE 缺失（直接支撑正文 claim，强烈建议补跑后重打包）:")
        for m in missing_core:
            print(f"    - {m}")
    if missing_support:
        print("\n--- SUPPORT 缺失（审稿稳健性，建议补）:")
        for m in missing_support:
            print(f"    - {m}")
    if skipped_big:
        print(f"\n(过大被跳过，>{MAX_FILE_MB}MB):")
        for m in skipped_big:
            print(f"    - {m}")
    if not (missing_core or missing_support):
        print("\n全部 core+support 文件齐全。")
    print(f"\n把 {ZIP_NAME} 发回来即可。")


if __name__ == "__main__":
    main()
