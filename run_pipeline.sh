#!/usr/bin/env bash
#
# run_pipeline.sh
# -----------------------------------------------------------------------------
# 一条命令跑完三个分析分支，产出所有结果。
# 前提：dr_sessions.csv 已存在（由 prep_*.py 离线生成，不在本 pipeline 内）。
#
# 三个分支彼此独立、只读 dr_sessions.csv；分支【内部】有"分析->图"的先后序。
# 任一分支/步骤失败不影响其他（逐步判错，最后汇总）。
#
# 用法：
#     bash run_pipeline.sh            # 跑全部三个分支
#     bash run_pipeline.sh setback    # 只跑分支A
#     bash run_pipeline.sh cate       # 只跑分支B
#     bash run_pipeline.sh behavior   # 只跑分支C
# -----------------------------------------------------------------------------
set -uo pipefail

PY="${PYTHON:-python}"
TARGET="${1:-all}"
FAILED=()

run() {  # run <脚本名>；缺文件或报错只记录、不中断整条 pipeline
  if [ ! -f "$1" ]; then echo "  [跳过] 找不到 $1"; return; fi
  echo ""
  echo ">>> 运行 $1"
  if ! "$PY" "$1"; then
    echo "  [失败] $1"
    FAILED+=("$1")
  fi
}

if [ ! -f dr_sessions.csv ]; then
  echo "错误：当前目录没有 dr_sessions.csv。"
  echo "      请先用 prep_*.py 离线生成（数据提取不在本 pipeline 内）。"
  exit 1
fi

echo "=============================================================="
echo " DR 分析 pipeline  (输入: dr_sessions.csv)   目标: $TARGET"
echo "=============================================================="

# ---- 分支A：setback剂量响应 & opt-out行为 ----
if [ "$TARGET" = "all" ] || [ "$TARGET" = "setback" ]; then
  echo ""; echo "########## 分支A: setback响应 ##########"
  run exploratory_scan.py             # 广扫（独立）
  run setback_response_analysis.py    # 必须在 figures 之前（产 f1_*/f3_*/f4_*.csv）
  run setback_response_robustness.py  # 独立稳健性
  run setback_response_figures.py     # 读 analysis 产物
  run setback_response_statemap.py    # 读 analysis 的 f3_state_agg.csv
fi

# ---- 分支B：CATE & 最优setback策略 ----
if [ "$TARGET" = "all" ] || [ "$TARGET" = "cate" ]; then
  echo ""; echo "########## 分支B: CATE策略 ##########"
  run cate_estimation.py          # 必须最先（产 cate_results.npz / cate_model.pkl）
  run cate_policy_evaluation.py   # 读 cate_results.npz
  run cate_policy_figures.py      # 读 estimation + policy 产物
fi

# ---- 分支C：行为感知DR模型 ----
if [ "$TARGET" = "all" ] || [ "$TARGET" = "behavior" ]; then
  echo ""; echo "########## 分支C: 行为感知模型 ##########"
  run behavior_model_main.py               # 核心模型
  run behavior_model_diagnostics.py        # 诊断（独立读 dr_sessions.csv）
  run behavior_model_validation.py         # 验证实验
  run behavior_model_persistence_checks.py # 持续性/因果sanity
fi

echo ""
echo "=============================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo " 全部完成，无失败步骤。"
else
  echo " 完成，但以下步骤失败（其余结果不受影响）："
  for f in "${FAILED[@]}"; do echo "   - $f"; done
fi
echo " 产出目录："
echo "   分支A -> paper1_out/   (含 robustness/) + setback图 + slide2_map.png"
echo "   分支B -> paper2_out/   (+ figs/)"
echo "   分支C -> behavior_model_out/  remaining_experiments_out/  persistence_sanity_out/"
echo "=============================================================="
