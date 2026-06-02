import pandas as pd
from pathlib import Path

OUT = Path("behavior_model_out")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("dr_sessions.csv")

# 基本保持你主分析口径
if "HvacMode" in df.columns:
    df = df[df["HvacMode"].astype(str).str.lower().eq("cool")].copy()

if "N_DR_Rows" in df.columns:
    df = df[pd.to_numeric(df["N_DR_Rows"], errors="coerce") >= 3].copy()

if "Setback_Amplitude_Mean" in df.columns:
    sb = pd.to_numeric(df["Setback_Amplitude_Mean"], errors="coerce")
    df = df[sb.between(0, 6)].copy()

# outcome
if "OptOut_Immediate" in df.columns:
    y_col = "OptOut_Immediate"
elif "Opted_Out" in df.columns:
    y_col = "Opted_Out"
else:
    raise ValueError("No opt-out outcome found.")

df[y_col] = pd.to_numeric(df[y_col], errors="coerce").astype("Int64")

# time ordering
if "Session_Start" in df.columns:
    df["Session_Start"] = pd.to_datetime(df["Session_Start"], errors="coerce")
    df = df.sort_values(["Identifier", "Session_Start"]).copy()
else:
    df = df.sort_values(["Identifier"]).copy()

df["Prev"] = df.groupby("Identifier")[y_col].shift(1)
df["Current"] = df[y_col]
df["Next"] = df.groupby("Identifier")[y_col].shift(-1)

d = df.dropna(subset=["Prev", "Current", "Next"]).copy()
for c in ["Prev", "Current", "Next"]:
    d[c] = d[c].astype(int)

tbl = (
    d.groupby(["Prev", "Current", "Next"])
    .size()
    .reset_index(name="n")
)

den = (
    d.groupby(["Prev", "Current"])
    .size()
    .reset_index(name="denom_prev_current")
)

tbl = tbl.merge(den, on=["Prev", "Current"], how="left")
tbl["prob_next_given_prev_current"] = tbl["n"] / tbl["denom_prev_current"]

tbl["pattern"] = (
    tbl["Prev"].astype(str)
    + " -> "
    + tbl["Current"].astype(str)
    + " -> "
    + tbl["Next"].astype(str)
)

tbl = tbl[
    ["pattern", "Prev", "Current", "Next", "n", "denom_prev_current", "prob_next_given_prev_current"]
].sort_values(["Prev", "Current", "Next"])

tbl.to_csv(OUT / "03e_three_event_transition.csv", index=False)

# Two specific interpretations
def get_prob(prev, cur, nxt):
    row = tbl[(tbl["Prev"] == prev) & (tbl["Current"] == cur) & (tbl["Next"] == nxt)]
    if row.empty:
        return None
    r = row.iloc[0]
    return r["prob_next_given_prev_current"], int(r["n"]), int(r["denom_prev_current"])

p_010 = get_prob(0, 1, 0)
p_101 = get_prob(1, 0, 1)

print(tbl.to_string(index=False))
print()
if p_010:
    print(f"P(NextStay | PrevStay, CurrentOptOut) = {p_010[0]:.4f}, n = {p_010[1]}, denom = {p_010[2]}")
if p_101:
    print(f"P(NextOptOut | PrevOptOut, CurrentStay) = {p_101[0]:.4f}, n = {p_101[1]}, denom = {p_101[2]}")

print()
print(f"Saved: {OUT / '03e_three_event_transition.csv'}")