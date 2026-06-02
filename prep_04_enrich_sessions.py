"""
Enrich dr_all_merged.parquet with metadata + weather + DR control signal.
Streams by user batches to stay memory-safe.

Input:  dr_all_merged.parquet (kept untouched)
Output: dr_data.parquet (replaces dr_all_merged.parquet's downstream role)
"""
import pandas as pd
import numpy as np
import gc
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

t_start = time.time()
print('01_enrich.py started', flush=True)

INPUT_PARQUET   = 'dr_all_merged.parquet'
OUTPUT_PARQUET  = 'dr_data.parquet'
META_PATH       = 'dyd_data/metadata/dyd_metadata_2024summer.csv'
GEO_CACHE       = Path('weather_cache/city_geocode.parquet')
WX_DIR          = Path('weather_cache/wx_by_grid')
USER_BATCH_SIZE = 500

# ============================================================
# 1. Build metadata + grid lookup
# ============================================================
print('\n[1/4] Loading metadata + geocode')
meta = pd.read_csv(META_PATH, dtype=str)
print(f'  Metadata: {len(meta):,} thermostats')

geo = pd.read_parquet(GEO_CACHE).dropna(subset=['lat','lon']).copy()
geo['lat'] = geo['lat'].astype(float)
geo['lon'] = geo['lon'].astype(float)
geo['lat_g'] = (geo['lat']/0.25).round() * 0.25
geo['lon_g'] = (geo['lon']/0.25).round() * 0.25

state_med = geo.groupby(['country','province_state']).agg(
    n_cities=('city','count'), med_lat=('lat','median')).reset_index()
geo = geo.merge(state_med, on=['country','province_state'], how='left')
geo['weather_is_fallback'] = ((geo['lat']==geo['med_lat']) &
                              (geo['n_cities']>1)).fillna(False).astype(bool)

for c in ['floor_area_sqft','number_floors','building_age_yrs',
          'number_occupants','number_cool_stages','number_heat_stages',
          'number_remote_sensors']:
    meta[c] = pd.to_numeric(meta[c], errors='coerce')
for c in ['allow_comp_with_aux','has_electric','has_heatpump']:
    meta[c] = meta[c].map({'true':True,'false':False})

id_meta = meta.merge(
    geo[['city','province_state','country','lat_g','lon_g','weather_is_fallback']],
    on=['city','province_state','country'], how='left'
).set_index('identifier')

KEEP_META = ['floor_area_sqft','number_floors','building_age_yrs','number_occupants',
             'number_cool_stages','number_heat_stages','number_remote_sensors',
             'allow_comp_with_aux','has_electric','has_heatpump','building_type',
             'model','country','province_state','city',
             'lat_g','lon_g','weather_is_fallback']

# ============================================================
# 2. Pre-load weather grids
# ============================================================
print('\n[2/4] Pre-loading weather grids')
wx_files = sorted(WX_DIR.glob('wx_*.parquet'))
wx_cache = {}
for f in wx_files:
    parts = f.stem.split('_')
    lat_g, lon_g = float(parts[1]), float(parts[2])
    df = pd.read_parquet(f)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    for c in df.columns:
        if c != 'time' and df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')
    wx_cache[(lat_g, lon_g)] = df
print(f'  {len(wx_cache)} grids, '
      f'{sum(d.memory_usage(deep=True).sum() for d in wx_cache.values())/1e9:.2f} GB')

WX_VARS = ['temperature_2m','relative_humidity_2m','dew_point_2m',
           'wind_speed_10m','shortwave_radiation','cloud_cover']

# ============================================================
# 3. Stream + enrich by user batches
# ============================================================
print('\n[3/4] Streaming enrichment')
all_uids = pd.read_parquet(INPUT_PARQUET, columns=['Identifier'])['Identifier'].unique()
print(f'  Users: {len(all_uids):,}')

DR_EXACT = {'CS DRsb','CS DRpc','CS DRPC','drsb','drpc',
            'DR Event sb','DR Event pc','DR Eventsb','DR Eventpc'}
DR_PREFIX = ('DR CTS',)

def is_dr_vec(cal):
    cal = cal.fillna('').astype(str).str.strip()
    mask = cal.isin(DR_EXACT)
    for p in DR_PREFIX:
        mask = mask | cal.str.startswith(p)
    return mask

parts = []
n_batches = (len(all_uids) + USER_BATCH_SIZE - 1) // USER_BATCH_SIZE

for bi in range(n_batches):
    t_b = time.time()
    batch_uids = list(all_uids[bi*USER_BATCH_SIZE:(bi+1)*USER_BATCH_SIZE])
    df = pd.read_parquet(INPUT_PARQUET, filters=[('Identifier','in',batch_uids)])
    df['date_time'] = pd.to_datetime(df['date_time'], utc=True)

    # 3a. metadata join
    meta_sub = id_meta.reindex(df['Identifier'].values)[KEEP_META].reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, meta_sub], axis=1)

    # 3b. weather attach (per grid)
    for c in WX_VARS:
        df[c] = np.nan
    out = []
    for (lat,lon), sub in df.groupby(['lat_g','lon_g'], dropna=False):
        if pd.isna(lat) or (lat,lon) not in wx_cache:
            out.append(sub); continue
        wx = wx_cache[(lat,lon)]
        sub = sub.sort_values('date_time').drop(columns=WX_VARS)
        merged = pd.merge_asof(sub, wx[['time']+WX_VARS],
                               left_on='date_time', right_on='time',
                               direction='nearest', tolerance=pd.Timedelta('1h'))
        merged = merged.drop(columns=['time'])
        out.append(merged)
    df = pd.concat(out, ignore_index=True)
    del out

    # 3c. DR signal features
    df = df.sort_values(['Identifier','date_time']).reset_index(drop=True)
    df['Comfort_Gap'] = df['Temp_F'] - df['Setpoint_Cool_F']
    df['is_DR'] = is_dr_vec(df['CalendarEvent'])

    non_event = (df['CalendarEvent'].fillna('').astype(str).str.strip() == '')
    user_normal = (df[non_event].groupby('Identifier')['Setpoint_Cool_F']
                   .median().rename('Normal_Setpoint_Cool'))
    df = df.merge(user_normal, left_on='Identifier', right_index=True, how='left')

    df['Setback_Amplitude'] = np.where(df['is_DR'],
                                       df['Setpoint_Cool_F'] - df['Normal_Setpoint_Cool'],
                                       np.nan)
    df['DR_onset'] = df['is_DR'] & (~df.groupby('Identifier')['is_DR']
                                    .shift(1, fill_value=False))
    df['_sp_min_2h'] = (df.groupby('Identifier')['Setpoint_Cool_F']
                        .transform(lambda s: s.rolling(24, min_periods=6).min()))
    df['Precool_Depth'] = np.where(df['DR_onset'],
                                   df['_sp_min_2h'] - df['Normal_Setpoint_Cool'],
                                   np.nan)
    df['T_diff_inout'] = df['temperature_2m'] - df['Temp_F']
    df['CDH_65'] = np.maximum(df['temperature_2m'] - 65, 0) / 12.0
    df = df.drop(columns=['_sp_min_2h'])

    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = df[c].astype('float32')

    pp = f'_part_{bi:03d}.parquet'
    df.to_parquet(pp, index=False)
    parts.append(pp)
    print(f'  Batch {bi+1}/{n_batches}: {len(batch_uids)} users, {len(df):,} rows, '
          f'{time.time()-t_b:.0f}s (total {time.time()-t_start:.0f}s)', flush=True)
    del df, meta_sub; gc.collect()

del wx_cache; gc.collect()

# ============================================================
# 4. Merge parts
# ============================================================
print('\n[4/4] Merging parts')
big = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
big.sort_values(['Identifier','date_time'], inplace=True)
big.to_parquet(OUTPUT_PARQUET, index=False)
for p in parts:
    Path(p).unlink()

print(f'\nOutput: {OUTPUT_PARQUET} '
      f'({Path(OUTPUT_PARQUET).stat().st_size/1e9:.2f} GB, '
      f'{len(big):,} rows, {len(big.columns)} cols)')

print('\n--- New field coverage ---')
for c in ['building_age_yrs','floor_area_sqft','has_heatpump','number_occupants',
          'temperature_2m','relative_humidity_2m','shortwave_radiation',
          'is_DR','Setback_Amplitude','Precool_Depth','T_diff_inout','CDH_65']:
    if c in big.columns:
        nn = big[c].notna().sum()
        print(f'  {c:25s}: {nn/len(big):>6.1%}')

print(f'\nTotal time: {time.time()-t_start:.0f}s')