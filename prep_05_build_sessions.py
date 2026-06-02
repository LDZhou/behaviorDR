"""
Extract DR sessions with weather + setback + building features at session level.
Adapted from previous 02_session_extraction logic; reads enriched parquet.

Input:  dr_data.parquet
Output: dr_sessions.csv
"""
import pandas as pd
import numpy as np
import time
import gc
import warnings
warnings.filterwarnings('ignore')

t_start = time.time()
print('02_sessions.py started', flush=True)

PARQUET   = 'dr_data.parquet'
OUTPUT    = 'dr_sessions.csv'
CHUNK     = 500

BASELINE_WIN  = pd.Timedelta(hours=2)
POST_WIN      = pd.Timedelta(minutes=30)
RECOVERY_WIN  = pd.Timedelta(hours=1)

USER_OVERRIDE = {'hold','auto'}
DR_EXACT  = {'CS DRsb','CS DRpc','CS DRPC','drsb','drpc',
             'DR Event sb','DR Event pc','DR Eventsb','DR Eventpc'}
DR_PREFIX = ('DR CTS',)
CS_DR_SET = {'CS DRsb','CS DRpc','CS DRPC'}

# Building features carried forward at session level
BLDG_COLS = ['floor_area_sqft','building_age_yrs','number_occupants',
             'has_heatpump','has_electric','number_cool_stages',
             'building_type','province_state','country','city',
             'weather_is_fallback']

print('Reading user IDs')
all_uids = pd.read_parquet(PARQUET, columns=['Identifier'])['Identifier'].unique()
print(f'Users: {len(all_uids):,}')

def is_dr_vec(cal):
    cal = cal.fillna('').astype(str).str.strip()
    m = cal.isin(DR_EXACT)
    for p in DR_PREFIX:
        m = m | cal.str.startswith(p)
    return m

def process_user(uid, udf):
    udf = udf.sort_values('date_time').reset_index(drop=True)
    udf['is_dr'] = is_dr_vec(udf['CalendarEvent'])
    if not udf['is_dr'].any():
        return []

    # User-level building (constant per user)
    bldg = {c: udf[c].dropna().iloc[0] if c in udf.columns and udf[c].notna().any()
            else np.nan for c in BLDG_COLS}

    udf['block'] = (udf['is_dr'] != udf['is_dr'].shift()).cumsum()
    out = []

    for _, bdf in udf[udf['is_dr']].groupby('block'):
        s_start = bdf['date_time'].min()
        s_end   = bdf['date_time'].max()
        dur     = (s_end - s_start).total_seconds() / 60.0
        n_rows  = len(bdf)
        if n_rows < 3:
            continue

        dr_type = 'CS_DR' if bdf['CalendarEvent'].isin(CS_DR_SET).any() else 'Utility_DR'

        # Indoor temp / cooling / setpoint during DR
        dr_temp = bdf['Temp_F'].mean()
        dr_cool = bdf['compCool1'].mean() if 'compCool1' in bdf.columns else np.nan
        dr_hum  = bdf['Humidity'].mean() if 'Humidity' in bdf.columns else np.nan
        sp_start = bdf['Setpoint_Cool_F'].dropna().iloc[0] if bdf['Setpoint_Cool_F'].notna().any() else np.nan
        sp_end   = bdf['Setpoint_Cool_F'].dropna().iloc[-1] if bdf['Setpoint_Cool_F'].notna().any() else np.nan
        sp_changed = (sp_start != sp_end) if pd.notna(sp_start) and pd.notna(sp_end) else np.nan
        comfort_gap_mean = bdf['Comfort_Gap'].mean()
        comfort_gap_max  = bdf['Comfort_Gap'].max()

        # Weather at onset (use first DR row)
        first_row = bdf.iloc[0]
        Tout_onset = first_row.get('temperature_2m', np.nan)
        RH_onset   = first_row.get('relative_humidity_2m', np.nan)
        GHI_onset  = first_row.get('shortwave_radiation', np.nan)
        dew_onset  = first_row.get('dew_point_2m', np.nan)

        # DR control signal (from enriched columns)
        setback_mean = bdf['Setback_Amplitude'].mean()
        setback_max  = bdf['Setback_Amplitude'].max()
        precool_depth = first_row.get('Precool_Depth', np.nan)  # value at onset
        normal_sp = first_row.get('Normal_Setpoint_Cool', np.nan)

        # Avg Tout over the event (heat-stress integral)
        Tout_during  = bdf['temperature_2m'].mean()
        CDH_during   = bdf['CDH_65'].sum()  # cumulative cooling-degree-hours during event

        # Occupancy
        if 'SensorOcc000' in bdf.columns and bdf['SensorOcc000'].notna().any():
            occ_rate = (bdf['SensorOcc000'] > 0).mean()
            occupied = occ_rate > 0.3
        else:
            occ_rate = np.nan; occupied = np.nan

        # HvacMode
        modes = bdf['HvacMode'].replace('', np.nan).dropna() if 'HvacMode' in bdf.columns else pd.Series(dtype=str)
        hvac_mode = modes.mode().iloc[0] if len(modes) > 0 else ''

        # Baseline (2h before, non-DR)
        bl = udf[(udf['date_time'] >= s_start - BASELINE_WIN) &
                 (udf['date_time'] < s_start) & (~udf['is_dr'])]
        bl_temp = bl['Temp_F'].mean() if len(bl) > 0 else np.nan
        bl_cool = bl['compCool1'].mean() if len(bl) > 0 and 'compCool1' in bl.columns else np.nan
        bl_hum  = bl['Humidity'].mean() if len(bl) > 0 and 'Humidity' in bl.columns else np.nan
        bl_Tout = bl['temperature_2m'].mean() if len(bl) > 0 else np.nan

        # Post-DR (opt-out detection)
        post = udf[(udf['date_time'] > s_end) &
                   (udf['date_time'] <= s_end + POST_WIN)]
        first_post = post.iloc[0]['CalendarEvent'] if len(post) > 0 else ''
        first_post = first_post if isinstance(first_post, str) else ''
        first_post_t = post.iloc[0]['date_time'] if len(post) > 0 else pd.NaT

        pre_rows = udf[(udf['date_time'] < s_start) & (~udf['is_dr'])]
        pre_event = pre_rows.iloc[-1]['CalendarEvent'] if len(pre_rows) > 0 else ''
        pre_event = pre_event if isinstance(pre_event, str) else ''

        oo_immediate = first_post in USER_OVERRIDE
        oo_delay = (first_post_t - s_end).total_seconds()/60.0 if oo_immediate and pd.notna(first_post_t) else np.nan
        oo_method = first_post if oo_immediate else ''
        oo_hold_only = (first_post == 'hold')
        oo_state_change = oo_immediate and (pre_event not in USER_OVERRIDE)
        oo_orig_mask = post['CalendarEvent'].isin(USER_OVERRIDE) if len(post)>0 else pd.Series(dtype=bool)
        oo_original = oo_orig_mask.any() if len(oo_orig_mask)>0 else False

        # Recovery (1h after, no DR)
        rec = udf[(udf['date_time'] > s_end) &
                  (udf['date_time'] <= s_end + RECOVERY_WIN) & (~udf['is_dr'])]
        rec_temp = rec['Temp_F'].mean() if len(rec) > 0 else np.nan
        rec_cool = rec['compCool1'].mean() if len(rec) > 0 and 'compCool1' in rec.columns else np.nan

        # Derived
        temp_rise = dr_temp - bl_temp if pd.notna(dr_temp) and pd.notna(bl_temp) else np.nan
        cool_red  = bl_cool - dr_cool if pd.notna(bl_cool) and pd.notna(dr_cool) else np.nan
        rebound   = rec_cool - bl_cool if pd.notna(rec_cool) and pd.notna(bl_cool) else np.nan
        bl_cool_frac  = bl_cool/300.0 if pd.notna(bl_cool) else np.nan
        dr_cool_frac  = dr_cool/300.0 if pd.notna(dr_cool) else np.nan
        rec_cool_frac = rec_cool/300.0 if pd.notna(rec_cool) else np.nan
        cool_red_frac = bl_cool_frac - dr_cool_frac if pd.notna(bl_cool_frac) and pd.notna(dr_cool_frac) else np.nan
        rebound_frac  = rec_cool_frac - bl_cool_frac if pd.notna(rec_cool_frac) and pd.notna(bl_cool_frac) else np.nan

        # Temporal
        hour = s_start.hour
        if   12 <= hour < 15: hbin = 'early_aft'
        elif 15 <= hour < 18: hbin = 'peak_aft'
        elif 18 <= hour < 21: hbin = 'evening'
        else:                 hbin = 'other'
        if   dur <= 60:  dbin = 'short_le1h'
        elif dur <= 180: dbin = 'medium_1_3h'
        elif dur <= 300: dbin = 'long_3_5h'
        else:            dbin = 'vlong_gt5h'

        rec_dict = {
            'Identifier': uid, 'DR_Type': dr_type,
            'Session_Start': s_start, 'Session_End': s_end,
            'Duration_Min': round(dur,1), 'N_DR_Rows': n_rows,
            'Hour_of_Day': hour, 'Hour_Bin': hbin, 'Month': s_start.month,
            'Weekday': s_start.weekday(), 'Is_Weekend': int(s_start.weekday()>=5),
            'Duration_Bin': dbin, 'HvacMode': hvac_mode,
            # Indoor / setpoint / cooling
            'Avg_Baseline_Temp': bl_temp, 'Avg_DR_Temp': dr_temp,
            'Avg_Recovery_Temp': rec_temp, 'Temp_Rise': temp_rise,
            'Setpoint_Cool_Start': sp_start, 'Setpoint_Cool_End': sp_end,
            'Setpoint_Changed': sp_changed,
            'Baseline_Setpoint_Cool': bl['Setpoint_Cool_F'].mean() if len(bl)>0 else np.nan,
            'Comfort_Gap_Mean': comfort_gap_mean, 'Comfort_Gap_Max': comfort_gap_max,
            'Avg_Baseline_Cool': bl_cool, 'Avg_DR_Cool': dr_cool,
            'Avg_Recovery_Cool': rec_cool, 'Cool_Reduction': cool_red, 'Rebound_Cool': rebound,
            'Baseline_Cool_Frac': bl_cool_frac, 'DR_Cool_Frac': dr_cool_frac,
            'Recovery_Cool_Frac': rec_cool_frac, 'Cool_Reduction_Frac': cool_red_frac,
            'Rebound_Cool_Frac': rebound_frac,
            'Avg_Baseline_Humidity': bl_hum,
            'Occupancy_Rate': occ_rate, 'Occupied': occupied,
            # Weather
            'Tout_onset': Tout_onset, 'Tout_baseline': bl_Tout,
            'Tout_during': Tout_during, 'CDH_during': CDH_during,
            'RH_onset': RH_onset, 'GHI_onset': GHI_onset, 'Dew_onset': dew_onset,
            # DR control signal
            'Setback_Amplitude_Mean': setback_mean,
            'Setback_Amplitude_Max':  setback_max,
            'Precool_Depth':          precool_depth,
            'Normal_Setpoint_Cool':   normal_sp,
            # Opt-out
            'Opted_Out': oo_immediate,
            'OptOut_Immediate': oo_immediate,
            'OptOut_Method': oo_method,
            'OptOut_Delay_Min': oo_delay,
            'OptOut_Hold_Only': oo_hold_only,
            'OptOut_StateChange': oo_state_change,
            'OptOut_Original': oo_original,
        }
        # Building features (constant per user, attach to each session)
        rec_dict.update(bldg)
        out.append(rec_dict)
    return out

# ============================================================
# Chunked execution
# ============================================================
print(f'\nProcessing in chunks of {CHUNK}')
all_sessions = []
n_chunks = (len(all_uids) + CHUNK - 1) // CHUNK

for ci in range(n_chunks):
    t_c = time.time()
    uids = list(all_uids[ci*CHUNK:(ci+1)*CHUNK])
    df = pd.read_parquet(PARQUET, filters=[('Identifier','in',uids)])
    n_sess = 0
    for uid, udf in df.groupby('Identifier'):
        s = process_user(uid, udf)
        all_sessions.extend(s)
        n_sess += len(s)
    del df; gc.collect()
    print(f'  Chunk {ci+1}/{n_chunks}: {len(uids)} users, {n_sess} sessions, '
          f'{time.time()-t_c:.0f}s (total {time.time()-t_start:.0f}s)', flush=True)

sdf = pd.DataFrame(all_sessions)
sdf.sort_values(['Identifier','Session_Start'], inplace=True)
sdf['Session_Seq'] = sdf.groupby('Identifier').cumcount() + 1
sdf['Delivered']   = (~sdf['Opted_Out']).astype(int)
sdf.to_csv(OUTPUT, index=False)

print(f'\nSessions: {len(sdf):,}, Users: {sdf["Identifier"].nunique():,}')
print(f'\n--- Coverage ---')
for c in ['Tout_onset','Setback_Amplitude_Mean','Precool_Depth',
          'Comfort_Gap_Mean','floor_area_sqft','building_age_yrs','has_heatpump']:
    nn = sdf[c].notna().sum()
    print(f'  {c:30s}: {nn/len(sdf):.1%}')

print(f'\n--- Opt-out rates ---')
for col in ['OptOut_Immediate','OptOut_Hold_Only','OptOut_StateChange','OptOut_Original']:
    print(f'  {col:25s}: {sdf[col].mean():.2%}')

print(f'\nTotal time: {time.time()-t_start:.0f}s')