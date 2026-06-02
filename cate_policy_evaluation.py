"""
Paper 2: AIPW policy value + bootstrap CI + overlap trimming + sensitivity + diagnostics.
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
warnings.filterwarnings('ignore')

OUT = Path('paper2_out')
d = np.load(OUT/'cate_results.npz', allow_pickle=True)
X, T, Y, G = d['X'], d['T'], d['Y'], d['groups']
opt_arm, mu_arm = d['opt_arm'], d['mu_arm']
X_names = list(d['X_names'])
n = len(Y)
N_ARMS = 4

# ============================================================
# 0. AIPW estimator
# ============================================================
def aipw(pi, Y, T, mu, e):
    i = np.arange(len(Y))
    mu_pi = mu[i, pi]; e_pi = e[i, pi]
    return (mu_pi + (T==pi).astype(float)/e_pi * (Y - mu_pi)).mean()

# ============================================================
# 1. Propensities
# ============================================================
print('Cross-fitting propensities...', flush=True)
propens = np.zeros((n, N_ARMS))
for tr, te in GroupKFold(5).split(X, T, G):
    clf = lgb.LGBMClassifier(objective='multiclass', num_class=N_ARMS,
                             n_estimators=400, num_leaves=31,
                             min_child_samples=200, verbose=-1)
    clf.fit(X[tr], T[tr])
    propens[te] = clf.predict_proba(X[te])
propens = np.clip(propens, 0.01, 0.99)
print('Propensity per arm (1%, 50%, 99%):')
for k in range(N_ARMS):
    q = np.quantile(propens[:,k], [0.01,0.5,0.99])
    print(f'  arm {k}: {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}')

# ============================================================
# 2. Policies
# ============================================================
state_cols = [i for i,nm in enumerate(X_names) if nm.startswith('province_state_')]
state_labels = [X_names[i].replace('province_state_','') for i in state_cols]
state_of = np.array([state_labels[r] for r in X[:, state_cols].argmax(axis=1)])
state_modal = {s: int(np.bincount(T[state_of==s], minlength=N_ARMS).argmax())
               for s in np.unique(state_of)}
pi_state = np.array([state_modal[s] for s in state_of])

policies = {
    'uniform_3_4F':   np.full(n, 3, dtype=int),
    'uniform_2_3F':   np.full(n, 2, dtype=int),
    'state_avg':      pi_state,
    'cate_targeted':  opt_arm,
    'observed':       T,
}

# ============================================================
# 3. PAIRWISE OVERLAP DIAGNOSTIC (CATE vs uniform_3_4F only)
# ============================================================
print('\n=== PAIRWISE TRIM: CATE vs uniform_3_4F ===')
def trim_for_pair(pi_a, pi_b, eps=0.05):
    ea = propens[np.arange(n), pi_a]
    eb = propens[np.arange(n), pi_b]
    return (ea >= eps) & (eb >= eps)

keep_pair = trim_for_pair(policies['cate_targeted'], policies['uniform_3_4F'])
print(f'Pair overlap: {keep_pair.sum():,}/{n:,} ({keep_pair.mean():.1%})')

v_c_p = aipw(policies['cate_targeted'][keep_pair], Y[keep_pair],
             T[keep_pair], mu_arm[keep_pair], propens[keep_pair])
v_u_p = aipw(policies['uniform_3_4F'][keep_pair], Y[keep_pair],
             T[keep_pair], mu_arm[keep_pair], propens[keep_pair])
print(f'CATE value:        {v_c_p:.4f}')
print(f'Uniform 3-4F:      {v_u_p:.4f}')
print(f'Pairwise lift:     {(v_c_p-v_u_p)/abs(v_u_p)*100:+.2f}%')

# ============================================================
# 4. CATE-vs-uniform DIFFERENTIAL DIAGNOSTIC
# ============================================================
print('\n=== WHERE DOES CATE DIFFER FROM UNIFORM 3-4F? ===')
diff_mask = policies['cate_targeted'] != policies['uniform_3_4F']
print(f'CATE differs from uniform 3-4F on: {diff_mask.sum():,}/{n:,} sessions '
      f'({diff_mask.mean():.1%})')
print('\nWhen differ, CATE picks:')
print(pd.Series(policies['cate_targeted'][diff_mask]).value_counts().sort_index().to_string())

print('\nFeature profile (differ vs same), continuous features only:')
cont_idx = [i for i,nm in enumerate(X_names)
            if not (nm.startswith('province_state_') or nm.startswith('Hour_Bin_')
                    or nm.startswith('Month_') or nm.startswith('Is_Weekend_'))]
profile = pd.DataFrame({
    'feature':     [X_names[i] for i in cont_idx],
    'mean_differ': [X[diff_mask, i].mean()  for i in cont_idx],
    'mean_same':   [X[~diff_mask, i].mean() for i in cont_idx],
})
profile['delta'] = profile['mean_differ'] - profile['mean_same']
print(profile.to_string(index=False))
profile.to_csv(OUT/'differ_vs_same_profile.csv', index=False)

# ============================================================
# 5. Conservative trimming (all policies, e>=0.05)
# ============================================================
min_e = np.full(n, 1.0)
for name, pi in policies.items():
    e_pi = propens[np.arange(n), pi]
    min_e = np.minimum(min_e, e_pi)
keep = min_e >= 0.05
print(f'\nConservative-trimmed: {keep.sum():,}/{n:,} ({keep.mean():.1%})')

Xk, Tk, Yk, Gk = X[keep], T[keep], Y[keep], G[keep]
propens_k = propens[keep]; mu_arm_k = mu_arm[keep]
policies_k = {name: pi[keep] for name, pi in policies.items()}
nk = keep.sum()

# ============================================================
# 6. Bootstrap on conservative-trimmed sample
# ============================================================
point = {name: aipw(pi, Yk, Tk, mu_arm_k, propens_k) for name, pi in policies_k.items()}

B = 500
uniq = np.unique(Gk)
user_rows = {u: np.where(Gk==u)[0] for u in uniq}
rng = np.random.default_rng(42)

print(f'Bootstrapping {B} iters...', flush=True)
boot = {k: [] for k in policies_k}
for b in range(B):
    samp = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([user_rows[u] for u in samp])
    for name, pi in policies_k.items():
        boot[name].append(aipw(pi[idx], Yk[idx], Tk[idx], mu_arm_k[idx], propens_k[idx]))

rows = []
for name in policies_k:
    a = np.array(boot[name])
    rows.append({'policy':name, 'value':point[name], 'se':a.std(),
                 'ci_lo':np.quantile(a,0.025), 'ci_hi':np.quantile(a,0.975)})
tbl = pd.DataFrame(rows)
base = tbl.loc[tbl['policy']=='uniform_3_4F','value'].iloc[0]
tbl['lift_vs_3_4F_pct'] = (tbl['value']-base)/abs(base)*100
tbl.to_csv(OUT/'policy_values.csv', index=False)
print('\n=== POLICY VALUES (conservative-trimmed) ===')
print(tbl.to_string(index=False))

diff_c = np.array(boot['cate_targeted']) - np.array(boot['uniform_3_4F'])
diff_u = np.array(boot['uniform_2_3F'])  - np.array(boot['uniform_3_4F'])
print(f'\nCATE vs uniform 3-4°F:  Δ={diff_c.mean():+.5f}  '
      f'95%CI [{np.quantile(diff_c,0.025):+.5f}, {np.quantile(diff_c,0.975):+.5f}]')
print(f'  Relative: {(diff_c.mean()/abs(base))*100:+.2f}% '
      f'[{np.quantile(diff_c,0.025)/abs(base)*100:+.2f}%, '
      f'{np.quantile(diff_c,0.975)/abs(base)*100:+.2f}%]   '
      f'P(lift>0)={(diff_c>0).mean():.3f}')
print(f'\nUniform 2-3°F vs 3-4°F: Δ={diff_u.mean():+.5f}  '
      f'95%CI [{np.quantile(diff_u,0.025):+.5f}, {np.quantile(diff_u,0.975):+.5f}]')

np.savez(OUT/'policy_bootstrap.npz', **{k: np.array(v) for k,v in boot.items()})

# ============================================================
# 7. Sensitivity
# ============================================================
print('\nSensitivity (outcome noise ρ):')
sens_rows = []
for rho in [0.0, 0.05, 0.10, 0.20, 0.30]:
    Yn = Yk + rng.standard_normal(nk)*rho*Yk.std()
    v_c = aipw(policies_k['cate_targeted'], Yn, Tk, mu_arm_k, propens_k)
    v_b = aipw(policies_k['uniform_3_4F'],  Yn, Tk, mu_arm_k, propens_k)
    sens_rows.append({'rho':rho, 'lift_pct':(v_c-v_b)/abs(v_b)*100})
pd.DataFrame(sens_rows).to_csv(OUT/'sensitivity.csv', index=False)
print(pd.DataFrame(sens_rows).to_string(index=False))

# ============================================================
# 8. Raw-mean sanity
# ============================================================
print('\nSanity: raw Y per observed arm (trimmed)')
for k in range(N_ARMS):
    m = Yk[Tk==k].mean(); n_k = int((Tk==k).sum())
    print(f'  arm {k}: n={n_k}, Y_mean={m:.4f}')
print(f'  overall: {Yk.mean():.4f}')

print('\nDone.')