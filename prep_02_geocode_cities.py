"""
Step 1b: Normalize city names, retry, then fall back to same-state nearest grid.
After this, every city should map to SOME grid (even if approximate).
"""
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import re
from pathlib import Path

CACHE_DIR = Path('weather_cache')
GEO_CACHE = CACHE_DIR / 'city_geocode.parquet'

def make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['GET'], respect_retry_after_header=True)
    s.mount('https://', HTTPAdapter(max_retries=retry, pool_maxsize=10))
    return s

session = make_session()

# ========== Name normalization ==========
def normalize_variants(name):
    """Generate candidate name variants from messy city string."""
    name = name.strip()
    variants = [name]

    # St. / St → Saint
    if re.search(r'\bSt\.?\s', name):
        variants.append(re.sub(r'\bSt\.?\s', 'Saint ', name))
    # Saint → St
    if 'Saint ' in name:
        variants.append(name.replace('Saint ', 'St. '))
        variants.append(name.replace('Saint ', 'St '))

    # Drop trailing tokens that confuse geocoder
    for suffix in [' County', ' Historic', ' Township', ' Twp',
                   ' Village', ' Borough', ' City']:
        if name.endswith(suffix):
            variants.append(name[: -len(suffix)].strip())

    # Common typo fix: "Patk" → "Park"
    if 'Patk' in name:
        variants.append(name.replace('Patk', 'Park'))

    # Compound names: try first 2 tokens (e.g. "South Prince George" → "Prince George")
    tokens = name.split()
    if len(tokens) >= 3:
        variants.append(' '.join(tokens[1:]))   # drop first
        variants.append(' '.join(tokens[:2]))   # keep first 2
        variants.append(tokens[-1])             # last token only

    # Dedup, preserve order
    seen = set()
    out = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

def geocode_one(name, state, country):
    try:
        r = session.get('https://geocoding-api.open-meteo.com/v1/search',
                        params={'name': name, 'count': 10, 'language': 'en'},
                        timeout=30)
        results = r.json().get('results', [])
        # Strict: country + state
        for res in results:
            if (res.get('country_code', '').upper() == country.upper() and
                state.upper() in res.get('admin1', '').upper()):
                return res['latitude'], res['longitude'], 'strict'
        # Loose: country only
        for res in results:
            if res.get('country_code', '').upper() == country.upper():
                return res['latitude'], res['longitude'], 'country_only'
    except Exception:
        pass
    return None, None, None

# ========== Phase 1: Variant retry on missing cities ==========
geo_full = pd.read_parquet(GEO_CACHE)
miss = geo_full[geo_full['lat'].isna()].copy()
print(f'Phase 1: variant-retry on {len(miss)} missing cities')

recovered = 0
for i, (idx, row) in enumerate(miss.iterrows()):
    variants = normalize_variants(row['city'])
    for v in variants:
        lat, lon, mode = geocode_one(v, row['province_state'], row['country'])
        if lat is not None:
            geo_full.loc[idx, 'lat'] = lat
            geo_full.loc[idx, 'lon'] = lon
            recovered += 1
            break
        time.sleep(0.3)
    if (i + 1) % 25 == 0:
        print(f'  {i+1}/{len(miss)}, recovered: {recovered}')

geo_full.to_parquet(GEO_CACHE)
print(f'Phase 1 recovered: {recovered}/{len(miss)}')

# ========== Phase 2: State-centroid fallback for remainder ==========
still = geo_full[geo_full['lat'].isna()].copy()
print(f'\nPhase 2: state-centroid fallback for {len(still)} remaining')

# Build state centroid from successfully geocoded cities
geocoded = geo_full[geo_full['lat'].notna()].copy()
state_centroids = (geocoded.groupby(['country', 'province_state'])
                   .agg(lat=('lat', 'median'), lon=('lon', 'median'),
                        n=('city', 'count'))
                   .reset_index())
print(f'  Built centroids for {len(state_centroids)} state/country combos')

fallback_count = 0
for idx, row in still.iterrows():
    match = state_centroids[
        (state_centroids['country'] == row['country']) &
        (state_centroids['province_state'] == row['province_state'])
    ]
    if len(match) > 0:
        geo_full.loc[idx, 'lat'] = float(match['lat'].iloc[0])
        geo_full.loc[idx, 'lon'] = float(match['lon'].iloc[0])
        fallback_count += 1

geo_full.to_parquet(GEO_CACHE)

# ========== Final report ==========
print(f'\n{"="*60}')
print(f'Phase 1 (name variants): +{recovered}')
print(f'Phase 2 (state centroid fallback): +{fallback_count}')
print(f'Total geocoded: {geo_full["lat"].notna().sum()}/{len(geo_full)} '
      f'({geo_full["lat"].notna().mean():.1%})')

unmatched = geo_full[geo_full['lat'].isna()]
if len(unmatched) > 0:
    print(f'\nStill unmatched (no state info?): {len(unmatched)}')
    print(unmatched[['city', 'province_state', 'country']].head(10).to_string())
else:
    print('\nAll cities mapped to a grid (some via state centroid fallback)')

# Mark which are exact vs fallback for downstream awareness
geo_full['geocode_method'] = 'unknown'
print(f'{"="*60}')