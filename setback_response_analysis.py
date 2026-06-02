"""
Paper 1: U-shape + Precool + Program>Climate + ICC/Persistence
Input:  dr_sessions.csv
Output: paper1_out/paper1_results.txt + *.csv
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUT = Path('paper1_out'); OUT.mkdir(exist_ok=True)
LOG = []
def log(s=''):
    print(s, flush=True); LOG.append(str(s))
def sect(t):
    log('\n' + '='*70); log(t); log('='*70)

# ============================================================
# Load + prep
# ============================================================
df = pd.read_csv('dr_sessions.csv')
df = df[df['HvacMode']=='cool'].copy()
df = df[df['Setback_Amplitude_Mean'].between(0, 6)]
df = df[df['N_DR_Rows'] >= 3]
need = ['Setback_Amplitude_Mean','Tout_onset','OptOut_Immediate','CDH_during',
        'floor_area_sqft','building_age_yrs','has_heatpump','number_occupants',
        'province_state','Hour_Bin','Month','Duration_Min']
df = df.dropna(subset=need).copy()

df['OptOut']      = df['OptOut_Immediate'].astype(int)
df['Setback']     = df['Setback_Amplitude_Mean']
df['Setback_sq']  = df['Setback']**2
df['Precool']     = df['Precool_Depth'].fillna(0)
df['HP']          = df['has_heatpump'].astype(int)
df['Weekend']     = df['Is_Weekend'].astype(int)
df['floor_1k']    = df['floor_area_sqft'] / 1000.0
df['age_dec']     = df['building_age_yrs'] / 10.0
df['log_occ']     = np.log1p(df['number_occupants'].fillna(df['number_occupants'].median()))

log(f'N = {len(df):,} sessions, {df["Identifier"].nunique():,} users, '
    f'{df["province_state"].nunique()} states')
log(f'Overall opt-out: {df["OptOut"].mean():.2%}')

# ============================================================
# F1. U-SHAPE (pooled logit, state FE, user-clustered SE)
# ============================================================
sect('F1. U-SHAPE')
fml = ('OptOut ~ Setback + Setback_sq + Precool + Tout_onset + CDH_during + '
       'Duration_Min + Weekend + HP + floor_1k + age_dec + log_occ + '
       'C(Hour_Bin) + C(Month) + C(province_state)')
m = smf.logit(fml, data=df).fit(disp=0, cov_type='cluster',
                                cov_kwds={'groups': df['Identifier']})
b1, b2 = m.params['Setback'], m.params['Setback_sq']
p1, p2 = m.pvalues['Setback'], m.pvalues['Setback_sq']
log(f'β(Setback)   = {b1:+.4f}  (p={p1:.2e})')
log(f'β(Setback²)  = {b2:+.4f}  (p={p2:.2e})')

if b2 > 0 and p2 < 0.05:
    opt = -b1/(2*b2)
    cov = m.cov_params().loc[['Setback','Setback_sq'], ['Setback','Setback_sq']].values
    grad = np.array([-1/(2*b2), b1/(2*b2**2)])
    se = float(np.sqrt(grad @ cov @ grad))
    log(f'Optimal setback = {opt:.2f} °F  95% CI [{opt-1.96*se:.2f}, {opt+1.96*se:.2f}]')
    log(f'U-shape CONFIRMED')
else:
    log('U-shape NOT confirmed at α=0.05')

pd.DataFrame({'coef':m.params,'se':m.bse,'z':m.tvalues,'p':m.pvalues,
              'ci_lo':m.conf_int()[0],'ci_hi':m.conf_int()[1]}
             ).to_csv(OUT/'f1_ushape_coefs.csv')
log(f'N={int(m.nobs)}, Pseudo-R²={m.prsquared:.4f}')

df['SB_bin'] = pd.cut(df['Setback'], bins=[0,1,2,3,4,6], include_lowest=True)
bin_tbl = df.groupby('SB_bin', observed=True).agg(
    n=('OptOut','count'), opt_out=('OptOut','mean'),
    temp_rise=('Temp_Rise','mean'), cool_red=('Cool_Reduction_Frac','mean')).reset_index()
bin_tbl['sb_mid'] = [0.5,1.5,2.5,3.5,5.0]
bin_tbl.to_csv(OUT/'f1_setback_bins.csv', index=False)
log(bin_tbl.to_string(index=False))

# Within-state robustness
sect('F1b. WITHIN-STATE U-SHAPE')
rows = []
for st, sub in df.groupby('province_state'):
    if len(sub) < 500 or sub['Setback'].nunique() < 20: continue
    try:
        ms = smf.logit('OptOut ~ Setback + Setback_sq + Precool + Tout_onset',
                       data=sub).fit(disp=0)
        b1s, b2s = ms.params['Setback'], ms.params['Setback_sq']
        p2s = ms.pvalues['Setback_sq']
        rows.append({'state':st,'n':len(sub),'b1':b1s,'b2':b2s,'p2':p2s,
                     'opt':(-b1s/(2*b2s) if b2s>0 else np.nan),
                     'u_ok':(b2s>0 and p2s<0.10)})
    except: pass
sdf = pd.DataFrame(rows).sort_values('n', ascending=False)
sdf.to_csv(OUT/'f1b_within_state.csv', index=False)
log(sdf.to_string(index=False))
log(f'{int(sdf["u_ok"].sum())}/{len(sdf)} states confirm U-shape (p<0.10)')

# ============================================================
# F2. PRECOOL
# ============================================================
sect('F2. PRECOOL EFFECT')
pc  = df[df['Precool_Depth'] < -1.0]
npc = df[df['Precool_Depth'].between(-0.5, 0.5)]
log(f'Precool (<-1°F):  n={len(pc):,}, OptOut={pc["OptOut"].mean():.2%}, '
    f'TempRise={pc["Temp_Rise"].mean():.2f}')
log(f'No precool:       n={len(npc):,}, OptOut={npc["OptOut"].mean():.2%}, '
    f'TempRise={npc["Temp_Rise"].mean():.2f}')
log(f'ΔOptOut = {pc["OptOut"].mean()-npc["OptOut"].mean():+.2%}')
log(f'β(Precool) in F1 model: {m.params["Precool"]:+.4f} (p={m.pvalues["Precool"]:.2e})')

# ============================================================
# F3. PROGRAM DESIGN > CLIMATE
# ============================================================
sect('F3. PROGRAM DESIGN > CLIMATE')
sa = df.groupby('province_state').agg(
    n=('OptOut','count'), opt_out=('OptOut','mean'),
    setback=('Setback','mean'), precool=('Precool_Depth','mean'),
    Tout=('Tout_onset','mean'), CDH=('CDH_during','mean')).reset_index()
sa = sa[sa['n']>=100].sort_values('opt_out', ascending=False)
sa.to_csv(OUT/'f3_state_agg.csv', index=False)
log(sa.to_string(index=False))

y = sa['opt_out'].values
r2_c = LinearRegression().fit(sa[['Tout','CDH']].values, y).score(sa[['Tout','CDH']].values, y)
r2_p = LinearRegression().fit(sa[['setback','precool']].values, y).score(sa[['setback','precool']].values, y)
r2_b = LinearRegression().fit(sa[['Tout','CDH','setback','precool']].values, y).score(
       sa[['Tout','CDH','setback','precool']].values, y)
log(f'\nState-level R² decomposition of opt-out:')
log(f'  climate only (Tout, CDH):      {r2_c:.3f}')
log(f'  program only (setback, pc):    {r2_p:.3f}')
log(f'  both:                          {r2_b:.3f}')
log(f'  → program explains {r2_p/max(r2_b,1e-6):.1%} of explainable variance')

m_noprog = smf.logit('OptOut ~ Tout_onset + CDH_during + Weekend + HP + '
                     'C(Hour_Bin) + C(Month)', data=df).fit(
    disp=0, cov_type='cluster', cov_kwds={'groups': df['Identifier']})
log(f'\nβ(Tout) without program controls: {m_noprog.params["Tout_onset"]:+.5f} '
    f'(p={m_noprog.pvalues["Tout_onset"]:.2e})')
log(f'β(Tout) with program controls:    {m.params["Tout_onset"]:+.5f} '
    f'(p={m.pvalues["Tout_onset"]:.2e})')

# ============================================================
# F4. ICC + PERSISTENCE
# ============================================================
sect('F4. ICC + PERSISTENCE')
multi = df[df.groupby('Identifier')['OptOut'].transform('count') >= 3].copy()
um = multi.groupby('Identifier')['OptOut'].mean()
vb = um.var()
vw = (multi['OptOut'] - multi.groupby('Identifier')['OptOut'].transform('mean')).var()
icc = vb / (vb + vw)
log(f'N users with ≥3 sessions: {multi["Identifier"].nunique():,}')
log(f'ICC(OptOut) = {icc:.3f}')

mseq = df.sort_values(['Identifier','Session_Start']).copy()
mseq['Prev_OO'] = mseq.groupby('Identifier')['OptOut'].shift(1)
v = mseq.dropna(subset=['Prev_OO'])
p_oo, p_st = v[v['Prev_OO']==1]['OptOut'].mean(), v[v['Prev_OO']==0]['OptOut'].mean()
gap = p_oo - p_st
log(f'P(OO | prev=OO)   = {p_oo:.2%}')
log(f'P(OO | prev=Stay) = {p_st:.2%}')
log(f'Persistence gap   = {gap:+.2%}')

vv = v.dropna(subset=['Setback','Tout_onset','floor_area_sqft','CDH_during']).copy()
vv['Prev_OO'] = vv['Prev_OO'].astype(int)
mp = smf.logit('OptOut ~ Prev_OO + Setback + Setback_sq + Precool + Tout_onset + '
               'CDH_during + Weekend + HP + floor_1k + C(Hour_Bin) + C(Month) + '
               'C(province_state)', data=vv).fit(
    disp=0, cov_type='cluster', cov_kwds={'groups': vv['Identifier']})
or_prev = np.exp(mp.params['Prev_OO'])
log(f'Controlled OR(Prev_OO) = {or_prev:.2f} (p={mp.pvalues["Prev_OO"]:.2e})')
log(f'  → past opt-out raises odds by {(or_prev-1)*100:.0f}% after controls')

pd.DataFrame([{'icc':icc,'p_after_oo':p_oo,'p_after_stay':p_st,'gap':gap,
               'or_prev_controlled':or_prev,'p_prev_controlled':mp.pvalues['Prev_OO']}
             ]).to_csv(OUT/'f4_persistence.csv', index=False)

with open(OUT/'paper1_results.txt','w') as f:
    f.write('\n'.join(LOG))
log(f'\nAll → {OUT}/')