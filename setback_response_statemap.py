import pandas as pd, matplotlib.pyplot as plt
sa = pd.read_csv('paper1_out/f3_state_agg.csv')
# Approx state centroids (lon, lat)
cen = {'VA':(-78.6,37.5),'CO':(-105.5,39.0),'OR':(-120.5,44.0),'NC':(-79.8,35.5),
       'SC':(-81.0,34.0),'MN':(-94.3,46.0),'IL':(-89.2,40.0),'NE':(-99.9,41.5),
       'MD':(-76.8,39.0),'IA':(-93.5,42.0),'ON':(-85.0,50.0),'TX':(-99.0,31.0),'DE':(-75.5,39.0)}
fig,ax=plt.subplots(figsize=(7,4.5))
for _,r in sa.iterrows():
    if r['province_state'] in cen:
        lon,lat=cen[r['province_state']]
        ax.scatter(lon,lat,s=r['n']/5,c=[r['opt_out']],cmap='YlOrRd',
                   vmin=0.15,vmax=0.45,edgecolor='black',alpha=0.85)
        ax.text(lon,lat-1.5,r['province_state'],ha='center',fontsize=9)
ax.set_xlim(-125,-65); ax.set_ylim(26,52); ax.set_aspect(1.3)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Study coverage (dot size = # sessions, color = opt-out rate)')
plt.tight_layout(); plt.savefig('slide2_map.png',dpi=200)