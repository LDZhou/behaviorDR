"""Paper 2 figures (4-arm)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['figure.dpi']=150; plt.rcParams['font.family']='DejaVu Sans'

OUT=Path('paper2_out'); FIG=OUT/'figs'; FIG.mkdir(exist_ok=True)
d = np.load(OUT/'cate_results.npz', allow_pickle=True)
opt_arm = d['opt_arm']
tbl = pd.read_csv(OUT/'policy_values.csv')
fi  = pd.read_csv(OUT/'feature_importance.csv').head(10)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

ax[0].hist(opt_arm, bins=np.arange(-0.5, 4.5, 1), rwidth=0.7,
           color='#4472c4', edgecolor='black')
ax[0].set_xticks([0,1,2,3])
ax[0].set_xticklabels(['0-1','1-2','2-3','3-4'])
ax[0].set_xlabel('Predicted optimal setback (°F)'); ax[0].set_ylabel('# sessions')
ax[0].set_title('(a) Distribution of optimal setbacks'); ax[0].grid(axis='y', alpha=0.3)

order = ['uniform_3_4F','uniform_2_3F','state_avg','cate_targeted']
labels = ['Uniform\n3-4°F','Uniform\n2-3°F','State-avg','CATE-\ntargeted']
sub = tbl.set_index('policy').loc[order]
err = np.array([[r['value']-r['ci_lo'], r['ci_hi']-r['value']] for _,r in sub.iterrows()]).T
ax[1].bar(labels, sub['value'], yerr=err, capsize=4,
          color=['#c0504d','#ffc000','#70ad47','#4472c4'], edgecolor='black')
for i,v in enumerate(sub['value']):
    ax[1].text(i, v+err[1,i]+0.001, f'{v:.4f}', ha='center', fontsize=9)
ax[1].set_ylabel('Delivered savings fraction')
ax[1].set_title('(b) AIPW policy value (95% CI)'); ax[1].grid(axis='y', alpha=0.3)

ax[2].barh(fi['feature'][::-1], fi['importance'][::-1],
           color='#8064a2', edgecolor='black')
ax[2].set_xlabel('Heterogeneity importance')
ax[2].set_title('(c) Top CATE drivers'); ax[2].grid(axis='x', alpha=0.3)

plt.tight_layout(); plt.savefig(FIG/'fig1_cate_policy.png', bbox_inches='tight'); plt.close()
print('Paper 2 figs →', FIG)