"""
Broad scan to find paper-worthy findings.
10 sections, each independent. Final ranked summary at end.

Input:  dr_sessions.csv
Output: console report + findings.csv
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print('03_broad_scan.py started\n')

sdf = pd.read_csv('dr_sessions.csv')
sdf['OptOut'] = sdf['Opted_Out'].astype(int)
print(f'Loaded {len(sdf):,} sessions, {sdf["Identifier"].nunique():,} users')
print(f'Overall opt-out: {sdf["OptOut"].mean():.2%}')
print(f'Cooling sessions: {(sdf["HvacMode"]=="cool").sum():,}\n')

# Restrict to cooling for thermal analyses (most events are summer cooling)
cool = sdf[sdf['HvacMode']=='cool'].copy()
print(f'Restricting to cool mode for thermal analyses: {len(cool):,} sessions\n')

findings = []
def add(section, finding, mag, potential, note=''):
    findings.append({'section':section,'finding':finding,'magnitude':mag,
                     'potential':potential,'note':note})
    print(f'  >>> {potential:6s} | {finding}  [mag={mag}]\n      {note}\n')

def bar(rate, width=40):
    return '█' * int(rate * width)

def sect(n, title):
    print('\n' + '='*70)
    print(f'F{n}. {title}')
    print('='*70)

# ============================================================
# F1. Outdoor temperature dose-response
# ============================================================
sect(1, 'Outdoor temperature dose-response on opt-out & temp rise')
sub = cool.dropna(subset=['Tout_onset']).copy()
print(f'  N with Tout: {len(sub):,}')
sub['Tout_bin'] = pd.cut(sub['Tout_onset'], bins=[40,70,80,85,90,95,110])
g = sub.groupby('Tout_bin', observed=True).agg(
    n=('OptOut','count'),
    opt_out=('OptOut','mean'),
    temp_rise=('Temp_Rise','mean'),
    cool_red_frac=('Cool_Reduction_Frac','mean'),
    comfort_gap=('Comfort_Gap_Mean','mean'),
).reset_index()
print(g.to_string(index=False))
print()
for _,r in g.iterrows():
    print(f'  {str(r["Tout_bin"]):15s} n={r["n"]:>5}  OptOut={r["opt_out"]:.1%} {bar(r["opt_out"])}')

if len(g) > 1:
    spread = g['opt_out'].max() - g['opt_out'].min()
    rises  = g['temp_rise'].dropna().values
    rise_spread = rises.max() - rises.min() if len(rises)>1 else 0
    rho, p_rho = stats.spearmanr(sub['Tout_onset'], sub['OptOut'])
    monotonic = all(g['opt_out'].diff().dropna() >= -0.005)
    pot = 'HIGH' if spread > 0.10 and monotonic else ('MED' if spread > 0.05 else 'LOW')
    add('F1', 'Tout → OptOut dose-response',
        f'spread={spread:.1%}, ρ={rho:.3f} (p={p_rho:.1e})',
        pot,
        f'Temp_Rise spread {rise_spread:.2f}°F across Tout bins; '
        f'monotonic={monotonic}. {"Strong confounder — must control" if pot=="HIGH" else "Weak"}')

# ============================================================
# F2. Setback amplitude dose-response (utility design knob)
# ============================================================
sect(2, 'Setback amplitude dose-response')
sub = cool.dropna(subset=['Setback_Amplitude_Mean']).copy()
sub = sub[sub['Setback_Amplitude_Mean'].between(-1, 12)]
sub['SB_bin'] = pd.cut(sub['Setback_Amplitude_Mean'], bins=[-1,0,1,2,3,4,6,12])
g = sub.groupby('SB_bin', observed=True).agg(
    n=('OptOut','count'),
    opt_out=('OptOut','mean'),
    temp_rise=('Temp_Rise','mean'),
    cool_red_frac=('Cool_Reduction_Frac','mean'),
).reset_index()
print(g.to_string(index=False))
print()
for _,r in g.iterrows():
    print(f'  {str(r["SB_bin"]):12s} n={r["n"]:>5}  OptOut={r["opt_out"]:.1%} {bar(r["opt_out"])}  '
          f'TempRise={r["temp_rise"]:.2f}°F')

if len(g) > 2 and sub['Setback_Amplitude_Mean'].notna().sum() > 100:
    spread = g['opt_out'].max() - g['opt_out'].min()
    # detect elbow: largest jump between adjacent bins
    diffs = g['opt_out'].diff().dropna().values
    elbow_idx = np.argmax(diffs) if len(diffs) > 0 else -1
    elbow_size = diffs.max() if len(diffs) > 0 else 0
    elbow_at = str(g.iloc[elbow_idx+1]['SB_bin']) if elbow_idx >= 0 else 'N/A'
    rho, p = stats.spearmanr(sub['Setback_Amplitude_Mean'], sub['OptOut'])
    pot = 'HIGH' if elbow_size > 0.05 else ('MED' if spread > 0.05 else 'LOW')
    add('F2', 'Setback amplitude → OptOut dose-response',
        f'spread={spread:.1%}, max-jump={elbow_size:.1%} at {elbow_at}, ρ={rho:.3f}',
        pot,
        f'{"Elbow detected — actionable utility insight" if elbow_size > 0.05 else "Roughly linear or weak"}')

# ============================================================
# F3. Building physics: which features predict temp_rise?
# ============================================================
sect(3, 'Building characteristics → Temp_Rise (delivered events only)')
delivered = cool[~cool['OptOut'].astype(bool)].copy()
feats = ['Setback_Amplitude_Mean','Tout_onset','floor_area_sqft',
         'building_age_yrs','number_occupants','Duration_Min']
delivered['has_heatpump_int'] = delivered['has_heatpump'].astype(float)
feats.append('has_heatpump_int')
df_reg = delivered[feats + ['Temp_Rise']].dropna()
print(f'  N for regression: {len(df_reg):,}')

if len(df_reg) > 200:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    X = df_reg[feats].values
    y = df_reg['Temp_Rise'].values
    Xs = StandardScaler().fit_transform(X)
    lr = LinearRegression().fit(Xs, y)
    r2 = lr.score(Xs, y)
    coefs = sorted(zip(feats, lr.coef_), key=lambda x: abs(x[1]), reverse=True)
    print(f'  R² = {r2:.4f}  (linear, standardized)\n')
    print(f'  Standardized coefficients (|β| ranked):')
    for f,c in coefs:
        print(f'    {f:30s}  β={c:+.3f}')

    top = coefs[0]
    pot = 'HIGH' if r2 > 0.15 and abs(top[1]) > 0.2 else ('MED' if r2 > 0.05 else 'LOW')
    add('F3', 'Building physics → Temp_Rise',
        f'R²={r2:.3f}, top={top[0]} (β={top[1]:+.2f})',
        pot,
        f'{"Building/program features explain meaningful variance" if pot!="LOW" else "Weak predictors"}')

# ============================================================
# F4. Precool effect
# ============================================================
sect(4, 'Precool depth → Temp_Rise & OptOut')
sub = cool.dropna(subset=['Precool_Depth']).copy()
sub = sub[sub['Precool_Depth'].between(-10, 2)]
print(f'  N with Precool_Depth: {len(sub):,}')
print(f'  Precool_Depth distribution: median={sub["Precool_Depth"].median():.2f}°F, '
      f'pct precool (<-0.5°F): {(sub["Precool_Depth"] < -0.5).mean():.1%}')
sub['PC_bin'] = pd.cut(sub['Precool_Depth'], bins=[-10,-3,-2,-1,-0.5,0.5,2])
g = sub.groupby('PC_bin', observed=True).agg(
    n=('OptOut','count'),
    opt_out=('OptOut','mean'),
    temp_rise=('Temp_Rise','mean'),
).reset_index()
print(g.to_string(index=False))

if len(g) > 1:
    rho_t, p_t = stats.spearmanr(sub['Precool_Depth'], sub['Temp_Rise'])
    rho_o, p_o = stats.spearmanr(sub['Precool_Depth'], sub['OptOut'])
    has_precool = sub[sub['Precool_Depth'] < -1]
    no_precool  = sub[sub['Precool_Depth'].between(-0.5, 0.5)]
    if len(has_precool) > 50 and len(no_precool) > 50:
        d_oo = no_precool['OptOut'].mean() - has_precool['OptOut'].mean()
        d_tr = no_precool['Temp_Rise'].mean() - has_precool['Temp_Rise'].mean()
        print(f'\n  Precool (<-1°F) vs No-precool: ΔOptOut={d_oo:+.1%}, ΔTempRise={d_tr:+.2f}°F')
        pot = 'HIGH' if abs(d_oo) > 0.05 or abs(d_tr) > 0.5 else 'LOW'
        add('F4', 'Precool effect',
            f'ΔOptOut={d_oo:+.1%}, ΔTempRise={d_tr:+.2f}°F',
            pot,
            f'{"Precool meaningfully reduces stress" if pot=="HIGH" else "Precool effect is small"}')

# ============================================================
# F5. State / climate heterogeneity
# ============================================================
sect(5, 'State-level heterogeneity')
g = cool.groupby('province_state').agg(
    n=('OptOut','count'),
    opt_out=('OptOut','mean'),
    temp_rise=('Temp_Rise','mean'),
    Tout=('Tout_onset','mean'),
    setback=('Setback_Amplitude_Mean','mean'),
).reset_index()
g = g[g['n'] >= 100].sort_values('opt_out', ascending=False)
print(g.to_string(index=False))

if len(g) > 3:
    spread_oo = g['opt_out'].max() - g['opt_out'].min()
    spread_tr = g['temp_rise'].max() - g['temp_rise'].min()
    pot = 'HIGH' if spread_oo > 0.15 else ('MED' if spread_oo > 0.07 else 'LOW')
    add('F5', 'State heterogeneity in opt-out',
        f'OptOut spread={spread_oo:.1%}, TempRise spread={spread_tr:.2f}°F',
        pot,
        f'Top: {g.iloc[0]["province_state"]} ({g.iloc[0]["opt_out"]:.1%}), '
        f'Bottom: {g.iloc[-1]["province_state"]} ({g.iloc[-1]["opt_out"]:.1%})')

# ============================================================
# F6. Heatpump vs gas furnace
# ============================================================
sect(6, 'Heatpump vs gas furnace (cooling mode)')
hp = cool[cool['has_heatpump']==True]
gas = cool[cool['has_heatpump']==False]
print(f'  Heatpump:  n={len(hp):,}, OptOut={hp["OptOut"].mean():.2%}, '
      f'TempRise={hp["Temp_Rise"].mean():.2f}°F')
print(f'  Non-HP:    n={len(gas):,}, OptOut={gas["OptOut"].mean():.2%}, '
      f'TempRise={gas["Temp_Rise"].mean():.2f}°F')
if len(hp) > 50 and len(gas) > 50:
    chi2, p_chi = stats.chi2_contingency(pd.crosstab(cool['has_heatpump'], cool['OptOut']))[:2]
    d_oo = hp['OptOut'].mean() - gas['OptOut'].mean()
    pot = 'MED' if abs(d_oo) > 0.03 else 'LOW'
    add('F6', 'HP vs non-HP cooling DR response',
        f'ΔOptOut={d_oo:+.1%}, χ²-p={p_chi:.1e}',
        pot, f'HP user pool is {len(hp)/(len(hp)+len(gas)):.1%}')

# ============================================================
# F7. Time-of-day / day-of-week / month
# ============================================================
sect(7, 'Temporal patterns')
print('  By hour-of-day:')
for h, sub in cool.groupby('Hour_of_Day'):
    if len(sub) >= 50:
        print(f'    Hour {h:>2}: n={len(sub):>5}  OptOut={sub["OptOut"].mean():.1%} {bar(sub["OptOut"].mean())}')
print('\n  By weekday vs weekend:')
for k, label in [(0,'Weekday'),(1,'Weekend')]:
    sub = cool[cool['Is_Weekend']==k]
    print(f'    {label}: n={len(sub):>5}  OptOut={sub["OptOut"].mean():.1%}')
print('\n  By month:')
for m, sub in cool.groupby('Month'):
    if len(sub) >= 50:
        print(f'    Month {m:>2}: n={len(sub):>5}  OptOut={sub["OptOut"].mean():.1%}')

hr_g = cool.groupby('Hour_of_Day')['OptOut'].mean()
hr_spread = hr_g.max() - hr_g.min()
mo_g = cool.groupby('Month')['OptOut'].mean()
mo_spread = mo_g.max() - mo_g.min()
pot = 'MED' if max(hr_spread, mo_spread) > 0.05 else 'LOW'
add('F7', 'Temporal opt-out patterns',
    f'hour spread={hr_spread:.1%}, month spread={mo_spread:.1%}',
    pot, 'Useful for utility event scheduling')

# ============================================================
# F8. ICC (within vs between user variance)
# ============================================================
sect(8, 'ICC: between- vs within-user variance')
multi = cool[cool.groupby('Identifier')['OptOut'].transform('count') >= 3]
um = multi.groupby('Identifier')['OptOut'].mean()
var_b = um.var()
var_w = (multi['OptOut'] - multi.groupby('Identifier')['OptOut'].transform('mean')).var()
icc_oo = var_b / (var_b + var_w) if (var_b+var_w) > 0 else np.nan

um_tr = multi.dropna(subset=['Temp_Rise']).groupby('Identifier')['Temp_Rise'].mean()
m_tr = multi.dropna(subset=['Temp_Rise'])
v_b_tr = um_tr.var()
v_w_tr = (m_tr['Temp_Rise'] - m_tr.groupby('Identifier')['Temp_Rise'].transform('mean')).var()
icc_tr = v_b_tr / (v_b_tr + v_w_tr) if (v_b_tr+v_w_tr) > 0 else np.nan

print(f'  N users with ≥3 sessions: {multi["Identifier"].nunique():,}')
print(f'  ICC(OptOut)    = {icc_oo:.3f}')
print(f'  ICC(Temp_Rise) = {icc_tr:.3f}')

pot = 'HIGH' if icc_oo > 0.4 else ('MED' if icc_oo > 0.2 else 'LOW')
add('F8', 'User-level ICC of opt-out',
    f'ICC_OO={icc_oo:.2f}, ICC_TR={icc_tr:.2f}',
    pot,
    f'{"User identity dominates — paper hook for targeting framing" if icc_oo > 0.4 else "Within-user variation is large"}')

# ============================================================
# F9. Persistence (Markov)
# ============================================================
sect(9, 'Opt-out persistence (P(OO_t | OO_{t-1}))')
multi = cool.sort_values(['Identifier','Session_Start'])
multi['Prev_OO'] = multi.groupby('Identifier')['OptOut'].shift(1)
v = multi.dropna(subset=['Prev_OO'])
p_after_oo  = v[v['Prev_OO']==1]['OptOut'].mean()
p_after_st  = v[v['Prev_OO']==0]['OptOut'].mean()
print(f'  P(OO | prev=OO)   = {p_after_oo:.2%}  (n={(v["Prev_OO"]==1).sum():,})')
print(f'  P(OO | prev=Stay) = {p_after_st:.2%}  (n={(v["Prev_OO"]==0).sum():,})')
print(f'  Persistence gap   = {p_after_oo - p_after_st:+.1%}')
pot = 'HIGH' if p_after_oo - p_after_st > 0.3 else ('MED' if p_after_oo - p_after_st > 0.1 else 'LOW')
add('F9', 'Opt-out persistence',
    f'gap={p_after_oo-p_after_st:.1%}',
    pot,
    f'{"Strong stickiness — single OO predicts churn" if pot=="HIGH" else "Moderate persistence"}')

# ============================================================
# F10. Setback × Tout interaction (heat wave + aggressive setback)
# ============================================================
sect(10, 'Setback × Tout interaction')
sub = cool.dropna(subset=['Setback_Amplitude_Mean','Tout_onset']).copy()
sub = sub[sub['Setback_Amplitude_Mean'].between(0,8)]
sub['Tout_lo'] = sub['Tout_onset'] < 85
sub['SB_lo']   = sub['Setback_Amplitude_Mean'] < 2
g = sub.groupby(['Tout_lo','SB_lo'])['OptOut'].agg(['count','mean']).reset_index()
g['Tout'] = g['Tout_lo'].map({True:'<85F', False:'≥85F'})
g['Setback'] = g['SB_lo'].map({True:'<2F', False:'≥2F'})
print(g[['Tout','Setback','count','mean']].to_string(index=False))

# Interaction = (high_high - high_low) - (low_high - low_low)
try:
    hh = g[(~g['Tout_lo']) & (~g['SB_lo'])]['mean'].iloc[0]
    hl = g[(~g['Tout_lo']) & ( g['SB_lo'])]['mean'].iloc[0]
    lh = g[( g['Tout_lo']) & (~g['SB_lo'])]['mean'].iloc[0]
    ll = g[( g['Tout_lo']) & ( g['SB_lo'])]['mean'].iloc[0]
    interaction = (hh - hl) - (lh - ll)
    print(f'\n  Interaction (extra OptOut from aggressive setback on hot days): {interaction:+.1%}')
    pot = 'HIGH' if abs(interaction) > 0.05 else 'LOW'
    add('F10', 'Setback × Tout interaction',
        f'extra OO on hot+aggressive: {interaction:+.1%}',
        pot,
        f'{"Important nonlinearity for utility scheduling" if pot=="HIGH" else "Mostly additive"}')
except Exception as e:
    print(f'  could not compute interaction: {e}')

# ============================================================
# Final ranked summary
# ============================================================
print('\n' + '='*70)
print('RANKED SUMMARY — paper potential')
print('='*70)
fdf = pd.DataFrame(findings)
order = {'HIGH':0,'MED':1,'LOW':2}
fdf['_o'] = fdf['potential'].map(order)
fdf = fdf.sort_values('_o').drop(columns='_o')
for _, r in fdf.iterrows():
    print(f'\n[{r["potential"]}] {r["section"]}: {r["finding"]}')
    print(f'        magnitude: {r["magnitude"]}')
    print(f'        note: {r["note"]}')

fdf.to_csv('findings.csv', index=False)
print(f'\nSaved findings.csv ({len(fdf)} entries)')
print('Done.')