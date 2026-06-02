"""
Step 2: Batched weather fetch via Open-Meteo Archive API.
  - Multi-location requests (up to 200 grids per call)
  - Exponential backoff retry on 429 / timeout
  - Resumes from existing cache
"""
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from pathlib import Path

CACHE_DIR = Path('weather_cache')
GEO_CACHE = CACHE_DIR / 'city_geocode.parquet'
WX_DIR    = CACHE_DIR / 'wx_by_grid'
WX_DIR.mkdir(exist_ok=True, parents=True)

START, END = '2024-05-01', '2024-10-31'
HOURLY_VARS = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
               'wind_speed_10m', 'shortwave_radiation', 'cloud_cover']
BATCH_SIZE = 200
MAX_RETRIES = 5

# ========== Session with retry ==========
def make_session():
    s = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'],
        respect_retry_after_header=True,
    )
    s.mount('https://', HTTPAdapter(max_retries=retry, pool_maxsize=20))
    return s

session = make_session()

# ========== Build grid list ==========
geo = pd.read_parquet(GEO_CACHE)
geo = geo.dropna(subset=['lat', 'lon']).copy()
geo['lat_g'] = (geo['lat'].astype(float) / 0.25).round() * 0.25
geo['lon_g'] = (geo['lon'].astype(float) / 0.25).round() * 0.25
grids = geo[['lat_g', 'lon_g']].drop_duplicates().reset_index(drop=True)
print(f'Total unique 0.25° grids: {len(grids)}')

# ========== Skip already-cached ==========
def grid_path(lat, lon):
    return WX_DIR / f'wx_{lat:.2f}_{lon:.2f}.parquet'

cached_mask = grids.apply(lambda r: grid_path(r.lat_g, r.lon_g).exists(), axis=1)
todo = grids[~cached_mask].reset_index(drop=True)
print(f'Already cached: {cached_mask.sum()}')
print(f'To fetch: {len(todo)}')

# ========== Batched fetch ==========
def fetch_batch(batch_df):
    lats = ','.join(f'{x:.4f}' for x in batch_df['lat_g'])
    lons = ','.join(f'{x:.4f}' for x in batch_df['lon_g'])
    params = {
        'latitude': lats,
        'longitude': lons,
        'start_date': START,
        'end_date': END,
        'hourly': ','.join(HOURLY_VARS),
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'UTC',
    }
    r = session.get('https://archive-api.open-meteo.com/v1/archive',
                    params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else [data]

if len(todo) == 0:
    print('Nothing to fetch.')
else:
    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'\nWill issue {n_batches} batched requests of up to {BATCH_SIZE} locs each\n')

    for bi in range(n_batches):
        batch = todo.iloc[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        print(f'  Batch {bi+1}/{n_batches}: {len(batch)} grids...', flush=True)
        try:
            results = fetch_batch(batch)
        except Exception as e:
            print(f'    FAILED: {e}, sleeping 60s and skipping')
            time.sleep(60)
            continue

        if len(results) != len(batch):
            print(f'    WARN: got {len(results)} results for {len(batch)} grids')

        n_saved = 0
        for (_, row), res in zip(batch.iterrows(), results):
            try:
                wx = pd.DataFrame(res['hourly'])
                wx['time'] = pd.to_datetime(wx['time'], utc=True)
                wx.to_parquet(grid_path(row.lat_g, row.lon_g))
                n_saved += 1
            except Exception as e:
                print(f'    parse err ({row.lat_g},{row.lon_g}): {e}')
        print(f'    saved {n_saved}/{len(batch)}', flush=True)

        time.sleep(2)

# ========== Verify ==========
have = sum(1 for _, r in grids.iterrows() if grid_path(r.lat_g, r.lon_g).exists())
print(f'\n{"="*60}')
print(f'Final coverage: {have}/{len(grids)} grids ({have/len(grids):.1%})')
print(f'{"="*60}')

# Coverage on city level (for downstream join planning)
geo_with_grid = geo.copy()
geo_with_grid['has_wx'] = geo_with_grid.apply(
    lambda r: grid_path(r.lat_g, r.lon_g).exists(), axis=1)
print(f'City-level coverage: {geo_with_grid["has_wx"].sum()}/{len(geo_with_grid)} '
      f'({geo_with_grid["has_wx"].mean():.1%})')