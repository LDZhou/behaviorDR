#!/usr/bin/env python3
"""Lightweight sensitivity check for the N_DR_Rows session-length threshold.

This reconstructs contiguous DR-labeled blocks from dr_data.parquet without the
full feature pipeline, including blocks with 1 or 2 records that prep_05 skipped.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET = Path("dr_data.parquet")
OUT = Path("strict_pre_event_out")
OUT.mkdir(exist_ok=True)
CHUNK = 500
POST_WIN = pd.Timedelta(minutes=30)
USER_OVERRIDE = {"hold", "auto"}
DR_EXACT = {"CS DRsb", "CS DRpc", "CS DRPC", "drsb", "drpc", "DR Event sb", "DR Event pc", "DR Eventsb", "DR Eventpc"}
DR_PREFIX = ("DR CTS",)


def is_dr_vec(cal: pd.Series) -> pd.Series:
    cal = cal.fillna("").astype(str).str.strip()
    out = cal.isin(DR_EXACT)
    for p in DR_PREFIX:
        out = out | cal.str.startswith(p)
    return out


def process_user(uid, udf: pd.DataFrame) -> list[dict]:
    udf = udf.sort_values("date_time").reset_index(drop=True)
    udf["is_dr"] = is_dr_vec(udf["CalendarEvent"])
    if not udf["is_dr"].any():
        return []
    udf["block"] = (udf["is_dr"] != udf["is_dr"].shift()).cumsum()
    rows = []
    for _, bdf in udf[udf["is_dr"]].groupby("block"):
        s_start = bdf["date_time"].min()
        s_end = bdf["date_time"].max()
        post = udf[(udf["date_time"] > s_end) & (udf["date_time"] <= s_end + POST_WIN)]
        first_post = post.iloc[0]["CalendarEvent"] if len(post) else ""
        first_post = first_post if isinstance(first_post, str) else ""
        modes = bdf["HvacMode"].replace("", np.nan).dropna() if "HvacMode" in bdf.columns else pd.Series(dtype=str)
        hvac_mode = modes.mode().iloc[0] if len(modes) else ""
        rows.append(
            {
                "Identifier": uid,
                "Session_Start": s_start,
                "N_DR_Rows": int(len(bdf)),
                "Duration_Min": float((s_end - s_start).total_seconds() / 60.0),
                "HvacMode": hvac_mode,
                "Setback_Amplitude_Mean": float(pd.to_numeric(bdf["Setback_Amplitude"], errors="coerce").mean())
                if "Setback_Amplitude" in bdf.columns
                else np.nan,
                "OptOut_Immediate": bool(first_post in USER_OVERRIDE),
            }
        )
    return rows


def main() -> None:
    t0 = time.time()
    pf = pq.ParquetFile(PARQUET)
    rows = []
    cols = ["date_time", "Identifier", "CalendarEvent", "HvacMode", "Setback_Amplitude"]
    carry = pd.DataFrame(columns=cols)
    for ci in range(pf.num_row_groups):
        df = pf.read_row_group(ci, columns=cols).to_pandas()
        if len(carry):
            df = pd.concat([carry, df], ignore_index=True)
            carry = pd.DataFrame(columns=cols)
        df = df.sort_values(["Identifier", "date_time"]).reset_index(drop=True)
        last_uid = df["Identifier"].iloc[-1]
        complete = df[df["Identifier"] != last_uid]
        carry = df[df["Identifier"] == last_uid].copy()
        for uid, udf in complete.groupby("Identifier", sort=False):
            rows.extend(process_user(uid, udf))
        del df, complete
        gc.collect()
        if (ci + 1) % 10 == 0 or ci == pf.num_row_groups - 1:
            print(f"row_group {ci+1}/{pf.num_row_groups}: cumulative sessions={len(rows):,}, elapsed={time.time()-t0:.0f}s", flush=True)
    if len(carry):
        for uid, udf in carry.groupby("Identifier", sort=False):
            rows.extend(process_user(uid, udf))
    sess = pd.DataFrame(rows).sort_values(["Identifier", "Session_Start"])
    sess.to_csv(OUT / "n_dr_rows_all_blocks.csv", index=False)

    summaries = []
    for label, mask in {
        "all_dr_blocks": pd.Series(True, index=sess.index),
        "cooling_blocks": sess["HvacMode"].astype(str).str.lower().eq("cool"),
        "cooling_mean_setback_0_6": sess["HvacMode"].astype(str).str.lower().eq("cool")
        & sess["Setback_Amplitude_Mean"].between(0, 6),
    }.items():
        sub0 = sess[mask].copy()
        for thr in [1, 2, 3, 4, 6, 12]:
            sub = sub0[sub0["N_DR_Rows"] >= thr]
            summaries.append(
                {
                    "cohort": label,
                    "threshold": thr,
                    "sessions": int(len(sub)),
                    "users": int(sub["Identifier"].nunique()),
                    "optout": float(sub["OptOut_Immediate"].mean()) if len(sub) else np.nan,
                    "mean_duration_min": float(sub["Duration_Min"].mean()) if len(sub) else np.nan,
                    "mean_n_rows": float(sub["N_DR_Rows"].mean()) if len(sub) else np.nan,
                }
            )
    summ = pd.DataFrame(summaries)
    summ.to_csv(OUT / "n_dr_rows_threshold_summary.csv", index=False)
    print(summ.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
