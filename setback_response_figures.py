"""Paper 1 figures — 3-panel single figure."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['figure.dpi']=150; plt.rcParams['font.family']='DejaVu Sans'

OUT=Path('paper1_out'); FIG=OUT/'figs'; FIG.mkdir(exist_ok=True)

# ---- Load ----
bins = pd.read_csv(OUT/'f1_setback_bins.csv')
pt   = pd.read_csv(OUT/'f4_persistence.csv').iloc[0]

df = pd.read_csv('dr_sessions.csv')
df = df[df['HvacMode']=='cool']
df = df[df['Setback_Amplitude_Mean'].between(0, 6)]
df['OptOut'] = df['Opted_Out'].astype(int)
df = df.dropna(subset=['Tout_onset','Setback_Amplitude_Mean'])
df['Tout_bin'] = pd.cut(df['Tout_onset'], bins=np.arange(70, 106, 3))
df['SB_stratum'] = pd.cut(df['Setback_Amplitude_Mean'],
                          bins=[0, 2, 3.5, 6],
                          labels=['Mild (0-2°F)', 'Moderate (2-3.5°F)', 'Aggressive (3.5-6°F)'])

# ---- 3 panels ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# (a) U-shape
ax[0].bar(bins['sb_mid'], bins['opt_out']*100, width=0.8,
          color='#c0504d', alpha=0.85, edgecolor='black')
for _,r in bins.iterrows():
    ax[0].text(r['sb_mid'], r['opt_out']*100+0.7,
               f'{r["opt_out"]*100:.1f}%\nn={int(r["n"])}',
               ha='center', fontsize=8)
ax[0].set_xticks([0.5,1.5,2.5,3.5,5.0])
ax[0].set_xticklabels(['0-1','1-2','2-3','3-4','4-6'])
ax[0].set_xlabel('Setback amplitude (°F)')
ax[0].set_ylabel('Opt-out rate (%)')
ax[0].set_title('(a) U-shape setback dose-response')
ax[0].grid(axis='y', alpha=0.3)

# (b) Tout stratified by setback
colors = {'Mild (0-2°F)':'#70ad47',
          'Moderate (2-3.5°F)':'#4472c4',
          'Aggressive (3.5-6°F)':'#c0504d'}
for stratum, sub in df.groupby('SB_stratum', observed=True):
    g = sub.groupby('Tout_bin', observed=True).agg(
        tout=('Tout_onset','mean'),
        oo=('OptOut','mean'),
        n=('OptOut','count')).reset_index()
    g = g[g['n'] >= 50]
    ax[1].plot(g['tout'], g['oo']*100, marker='o', label=stratum,
               color=colors[stratum], linewidth=2, markersize=7)
ax[1].set_xlabel('Outdoor temperature at DR onset (°F)')
ax[1].set_ylabel('Opt-out rate (%)')
ax[1].set_title('(b) Heat amplifies aggressive-setback penalty')
ax[1].legend(title='Setback level', loc='best', fontsize=9)
ax[1].grid(alpha=0.3)

# (c) Persistence
ax[2].bar([0,1], [pt['p_after_stay']*100, pt['p_after_oo']*100],
          color=['#70ad47','#c0504d'], edgecolor='black', width=0.55)
for i,v in enumerate([pt['p_after_stay'], pt['p_after_oo']]):
    ax[2].text(i, v*100+1.5, f'{v*100:.1f}%', ha='center', fontweight='bold')
ax[2].set_xticks([0,1]); ax[2].set_xticklabels(['Stayed last','Opted-out last'])
ax[2].set_ylabel('P(opt-out next event) (%)')
ax[2].set_title(f'(c) Persistence: ICC={pt["icc"]:.2f}, gap={pt["gap"]*100:+.0f}pp')
ax[2].set_ylim(0, max(pt['p_after_oo']*100, pt['p_after_stay']*100)*1.3)
ax[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG/'fig1_main.png', bbox_inches='tight')
plt.close()
print('Paper 1 fig →', FIG/'fig1_main.png')