"""
PASS 1: Scan DR users → merge → Parquet
New columns: TemperatureExpectedCool, TemperatureExpectedHeat, HvacMode,
             SensorOcc000, SensorTemp000, Climate
Output: dr_all_merged.parquet
"""
import pandas as pd
import numpy as np
import glob
import gc
import time
import os
import warnings
warnings.filterwarnings('ignore')

t_start = time.time()
print("01_extract_merge.py started", flush=True)

# ============================================================
# Config
# ============================================================
PARQUET_PATH = 'dr_all_merged.parquet'

DR_EXACT = {'CS DRsb', 'CS DRpc', 'CS DRPC', 'drsb', 'drpc',
            'DR Event sb', 'DR Event pc', 'DR Eventsb', 'DR Eventpc'}
DR_PREFIX = ('DR CTS',)

KEEP_COLS = [
    'date_time', 'Identifier', 'CalendarEvent', 'HvacMode', 'Climate',
    'Temperature_ctrl', 'TemperatureExpectedCool', 'TemperatureExpectedHeat',
    'Humidity', 'compCool1', 'compCool2', 'compHeat1', 'compHeat2',
    'auxHeat1', 'auxHeat2', 'fan',
    'SensorTemp000', 'SensorOcc000',
]

files = sorted(glob.glob('dyd_data/dr_thermostats/*.csv'))
print(f'Files: {len(files)}', flush=True)

# ============================================================
# PASS 1a: Scan for DR user IDs
# ============================================================
print(f'\n{"="*60}')
print('PASS 1a: Scanning DR users')
print('='*60)

dr_ids = set()
for i, f in enumerate(files):
    try:
        df = pd.read_csv(f, usecols=['Identifier', 'CalendarEvent'], dtype=str, engine='pyarrow')
    except Exception:
        df = pd.read_csv(f, usecols=['Identifier', 'CalendarEvent'], dtype=str)
    cal = df['CalendarEvent'].fillna('').str.strip()
    mask = cal.isin(DR_EXACT)
    for p in DR_PREFIX:
        mask = mask | cal.str.startswith(p)
    dr_ids.update(df.loc[mask, 'Identifier'].dropna().unique())
    del df; gc.collect()
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(files)}, DR users: {len(dr_ids)}', flush=True)

print(f'Total DR users: {len(dr_ids)}')
print(f'Pass 1a time: {time.time()-t_start:.0f}s')

# ============================================================
# PASS 1b: Merge DR user data → Parquet (batched)
# ============================================================
print(f'\n{"="*60}')
print('PASS 1b: Merging DR user data → Parquet')
print('='*60)
t1b = time.time()

batch_size = 50
batch_chunks = []
batch_num = 0
parquet_parts = []

for i, f in enumerate(files):
    try:
        df = pd.read_csv(f, dtype=str)
    except Exception as e:
        print(f'  Error reading {f}: {e}')
        continue

    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available]
    kept = df[df['Identifier'].isin(dr_ids)]

    if len(kept) > 0:
        batch_chunks.append(kept)
    del df; gc.collect()

    if (i+1) % batch_size == 0 or (i+1) == len(files):
        if batch_chunks:
            batch_df = pd.concat(batch_chunks, ignore_index=True)
            part_path = f'dr_part_{batch_num:03d}.parquet'

            batch_df['date_time'] = pd.to_datetime(batch_df['date_time'], errors='coerce')

            for col in ['Temperature_ctrl', 'TemperatureExpectedCool',
                        'TemperatureExpectedHeat', 'Humidity',
                        'compCool1', 'compCool2', 'compHeat1', 'compHeat2',
                        'auxHeat1', 'auxHeat2', 'fan',
                        'SensorTemp000', 'SensorOcc000']:
                if col in batch_df.columns:
                    batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce').astype('float32')

            batch_df['Temp_F'] = batch_df['Temperature_ctrl'] / 10.0 if 'Temperature_ctrl' in batch_df.columns else np.nan
            batch_df['Setpoint_Cool_F'] = batch_df['TemperatureExpectedCool'] / 10.0 if 'TemperatureExpectedCool' in batch_df.columns else np.nan
            batch_df['Setpoint_Heat_F'] = batch_df['TemperatureExpectedHeat'] / 10.0 if 'TemperatureExpectedHeat' in batch_df.columns else np.nan
            batch_df['Sensor_Temp_F'] = batch_df['SensorTemp000'] / 10.0 if 'SensorTemp000' in batch_df.columns else np.nan

            for col in ['CalendarEvent', 'HvacMode', 'Climate']:
                if col in batch_df.columns:
                    batch_df[col] = batch_df[col].fillna('').str.strip()

            drop_cols = ['Temperature_ctrl', 'TemperatureExpectedCool',
                         'TemperatureExpectedHeat', 'SensorTemp000']
            batch_df.drop(columns=[c for c in drop_cols if c in batch_df.columns], inplace=True)

            batch_df.to_parquet(part_path, index=False)
            parquet_parts.append(part_path)
            n_rows = len(batch_df)
            del batch_df
            batch_chunks = []
            gc.collect()
            batch_num += 1
            print(f'  Files {i+1}/{len(files)}, wrote {part_path} ({n_rows:,} rows)')

print(f'Wrote {len(parquet_parts)} parquet parts')
print(f'Pass 1b time: {time.time()-t1b:.0f}s')

# ============================================================
# PASS 2: Merge parts + sort
# ============================================================
print(f'\n{"="*60}')
print('PASS 2: Merging parts')
print('='*60)
t2 = time.time()

all_parts = [pd.read_parquet(p) for p in parquet_parts]
merged = pd.concat(all_parts, ignore_index=True)
del all_parts; gc.collect()

merged.sort_values(['Identifier', 'date_time'], inplace=True)
merged.reset_index(drop=True, inplace=True)
merged.to_parquet(PARQUET_PATH, index=False)

for p in parquet_parts:
    os.remove(p)

mem_gb = merged.memory_usage(deep=True).sum() / 1e9
n_users = merged['Identifier'].nunique()
print(f'Merged: {len(merged):,} rows, {n_users} users, {mem_gb:.1f} GB')
print(f'Columns: {list(merged.columns)}')

print(f'\n--- Data Quality ---')
for col in merged.columns:
    nna = merged[col].notna().sum()
    pct = nna / len(merged)
    print(f'  {col:25s}: {nna:>12,} ({pct:.0%})')

if 'HvacMode' in merged.columns:
    print(f'\nHvacMode distribution:')
    print(merged['HvacMode'].value_counts().head(10).to_string())

print(f'\nPass 2 time: {time.time()-t2:.0f}s')
print(f'Total time: {time.time()-t_start:.0f}s')
print(f'Output: {PARQUET_PATH} ({os.path.getsize(PARQUET_PATH)/1e9:.1f} GB)')