"""
Paper 2: CATE via CausalForestDML. 4-arm version (drops 4-6°F due to positivity).
Input:  dr_sessions.csv
Output: paper2_out/cate_results.npz + tables
"""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from econml.dml import CausalForestDML
import joblib, warnings
warnings.filterwarnings('ignore')

OUT = Path('paper2_out'); OUT.mkdir(exist_ok=True)

# ============================================================
# 1. Load + filter
# ============================================================
df = pd.read_csv('dr_sessions.csv')
df = df[df['HvacMode']=='cool'].copy()
df = df[df['Setback_Amplitude_Mean'].between(0, 4)]   # drop 4-6°F: positivity violation
df = df[df['N_DR_Rows'] >= 3]
need = ['Setback_Amplitude_Mean','Cool_Reduction_Frac','OptOut_Immediate',
        'Tout_onset','CDH_during','RH_onset','floor_area_sqft','building_age_yrs',
        'has_heatpump','province_state','Hour_Bin','Month','number_occupants',
        'Duration_Min','Session_Seq']
df = df.dropna(subset=need).copy()
print(f'Filtered: {len(df):,} sessions, {df["Identifier"].nunique():,} users')

# Treatment: 4 bins, ref = arm 2 = (2,3] °F
df['T'] = pd.cut(df['Setback_Amplitude_Mean'], bins=[0,1,2,3,4],
                 labels=[0,1,2,3], include_lowest=True).astype(int)
df['Y'] = df['Cool_Reduction_Frac'] * (1 - df['OptOut_Immediate'].astype(int))
print('Arm distribution:'); print(df['T'].value_counts().sort_index().to_string())
print(f'Y mean={df["Y"].mean():.4f}, std={df["Y"].std():.4f}')

# ============================================================
# 2. Features
# ============================================================
df = df.sort_values(['Identifier','Session_Start']).reset_index(drop=True)
df['_oo']     = df['OptOut_Immediate'].astype(int)
df['_cumoo']  = df.groupby('Identifier')['_oo'].cumsum()
df['_cumn']   = df.groupby('Identifier').cumcount() + 1
df['user_prior_oo'] = np.where(df['_cumn']>1,
                               (df['_cumoo']-df['_oo']) / (df['_cumn']-1),
                               0.23)   # population prior at session 1
df = df.drop(columns=['_oo','_cumoo','_cumn'])

df['precool']  = df['Precool_Depth'].fillna(0)
df['HP']       = df['has_heatpump'].astype(int)
df['log_occ']  = np.log1p(df['number_occupants'])
df['floor_1k'] = df['floor_area_sqft']/1000.0
df['age_dec']  = df['building_age_yrs']/10.0

cont = ['Tout_onset','CDH_during','RH_onset','precool','Duration_Min',
        'floor_1k','age_dec','log_occ','HP','user_prior_oo','Session_Seq']
cats = ['Hour_Bin','Month','province_state','Is_Weekend']

X_cont = df[cont].astype(float).values
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = enc.fit_transform(df[cats].astype(str).values)
X = np.hstack([X_cont, X_cat])
X_names = cont + list(enc.get_feature_names_out(cats))
T = df['T'].values; Y = df['Y'].values; G = df['Identifier'].values
print(f'X shape: {X.shape}')

# ============================================================
# 3. Fit CausalForestDML
# ============================================================
print('\nFitting CausalForestDML...', flush=True)
est = CausalForestDML(
    model_y=lgb.LGBMRegressor(n_estimators=400, num_leaves=31,
                              min_child_samples=200, verbose=-1),
    model_t=lgb.LGBMClassifier(objective='multiclass', num_class=4,
                               n_estimators=400, num_leaves=31,
                               min_child_samples=200, verbose=-1),
    discrete_treatment=True, n_estimators=2000,
    min_samples_leaf=50, cv=GroupKFold(n_splits=5), random_state=42)
est.fit(Y=Y, T=T, X=X, groups=G)
joblib.dump(est, OUT/'cate_model.pkl')
print('Model saved.', flush=True)

# ============================================================
# 4. Arm gains per session
# ============================================================
arm_gain_vs0 = np.zeros((len(df), 4))
for k in range(1, 4):
    e = est.effect(X, T0=0, T1=k)
    arm_gain_vs0[:, k] = np.asarray(e).ravel()
arm_gain_vs_ref = arm_gain_vs0 - arm_gain_vs0[:, 2:3]
opt_arm = np.argmax(arm_gain_vs0, axis=1)

# Cross-fitted mu for reference arm
print('Cross-fitting reference outcome (T=2)...', flush=True)
mu_ref = np.zeros(len(df))
for tr, te in GroupKFold(5).split(X, T, G):
    tr2 = tr[T[tr]==2]
    if len(tr2) < 50: continue
    r = lgb.LGBMRegressor(n_estimators=400, num_leaves=31,
                          min_child_samples=200, verbose=-1)
    r.fit(X[tr2], Y[tr2])
    mu_ref[te] = r.predict(X[te])
mu_arm = mu_ref[:, None] + arm_gain_vs_ref

# ============================================================
# 5. Save + summaries
# ============================================================
np.savez(OUT/'cate_results.npz',
         X=X, T=T, Y=Y, groups=G,
         arm_gain_vs0=arm_gain_vs0, arm_gain_vs_ref=arm_gain_vs_ref,
         opt_arm=opt_arm, mu_arm=mu_arm, mu_ref=mu_ref,
         X_names=np.array(X_names, dtype=object))

print('\nOptimal arm distribution:')
print(pd.Series(opt_arm).value_counts().sort_index().to_string())
print('\nCross-tab observed vs optimal:')
print(pd.crosstab(T, opt_arm, rownames=['obs'], colnames=['opt']).to_string())

fi = pd.DataFrame({'feature':X_names, 'importance':est.feature_importances_}
                  ).sort_values('importance', ascending=False).head(15)
fi.to_csv(OUT/'feature_importance.csv', index=False)
print('\nTop drivers:'); print(fi.to_string(index=False))

ate = pd.DataFrame({'arm':[0,1,2,3],
                    'label':['0-1','1-2','2-3','3-4'],
                    'mean_gain_vs0':     arm_gain_vs0.mean(0),
                    'mean_gain_vs_ref':  arm_gain_vs_ref.mean(0),
                    'n_chosen':          [(opt_arm==k).sum() for k in range(4)],
                    'raw_Y_mean':        [Y[T==k].mean() for k in range(4)]})
ate.to_csv(OUT/'ate_by_arm.csv', index=False)
print('\n'); print(ate.to_string(index=False))
print('\nCATE done.')