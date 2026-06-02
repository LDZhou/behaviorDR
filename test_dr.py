import pandas as pd
import numpy as np

df = pd.read_csv("dr_sessions.csv")

# 选择 outcome
# 优先用 OptOut_Immediate；如果没有，就换成你当前主 outcome
if "OptOut_Immediate" in df.columns:
    y_col = "OptOut_Immediate"
elif "Y_immediate" in df.columns:
    y_col = "Y_immediate"
elif "OptOut" in df.columns:
    y_col = "OptOut"
else:
    raise ValueError("Cannot find opt-out outcome column.")

# 时间列
if "Session_Start" in df.columns:
    time_col = "Session_Start"
elif "Start" in df.columns:
    time_col = "Start"
else:
    raise ValueError("Cannot find session start time column.")

df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

# 只保留 cooling，如果你的文件里有 HvacMode
if "HvacMode" in df.columns:
    m = df["HvacMode"].astype(str).str.lower().str.contains("cool", na=False)
    if m.sum() > 1000:
        df = df.loc[m].copy()

# 二值化
df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
df[y_col] = (df[y_col] > 0).astype(int)

# 按用户和时间排序
df = df.dropna(subset=["Identifier", time_col]).copy()
df = df.sort_values(["Identifier", time_col])

# 构造上一场、当前、下一场
g = df.groupby("Identifier", sort=False)
df["PrevOptOut"] = g[y_col].shift(1)
df["CurrOptOut"] = df[y_col]
df["NextOptOut"] = g[y_col].shift(-1)

# 只看有 prev 和 next 的中间事件
mid = df.dropna(subset=["PrevOptOut", "NextOptOut"]).copy()
mid["PrevOptOut"] = mid["PrevOptOut"].astype(int)
mid["NextOptOut"] = mid["NextOptOut"].astype(int)

# 0 -> 1 -> 0
case_010 = mid[(mid["PrevOptOut"] == 0) & (mid["CurrOptOut"] == 1)]
p_010 = (case_010["NextOptOut"] == 0).mean()
n_010 = len(case_010)

# 1 -> 0 -> 1
case_101 = mid[(mid["PrevOptOut"] == 1) & (mid["CurrOptOut"] == 0)]
p_101 = (case_101["NextOptOut"] == 1).mean()
n_101 = len(case_101)

print("Outcome column:", y_col)
print()
print("P(NextStay | PrevStay, CurrentOptOut)")
print(f"0 -> 1 -> 0 probability = {p_010:.4f} ({p_010*100:.2f}%), n = {n_010:,}")
print()
print("P(NextOptOut | PrevOptOut, CurrentStay)")
print(f"1 -> 0 -> 1 probability = {p_101:.4f} ({p_101*100:.2f}%), n = {n_101:,}")

# 顺便输出完整三连 transition table
tab = (
    mid.groupby(["PrevOptOut", "CurrOptOut", "NextOptOut"])
    .size()
    .reset_index(name="n")
)

tab["pattern"] = (
    tab["PrevOptOut"].astype(str)
    + " -> "
    + tab["CurrOptOut"].astype(str)
    + " -> "
    + tab["NextOptOut"].astype(str)
)

# 条件在 Prev,Curr 下的概率
tab["denom_prev_curr"] = tab.groupby(["PrevOptOut", "CurrOptOut"])["n"].transform("sum")
tab["prob_next_given_prev_curr"] = tab["n"] / tab["denom_prev_curr"]

print()
print("Full 3-event transition table:")
print(tab[["pattern", "n", "prob_next_given_prev_curr"]].to_string(index=False))