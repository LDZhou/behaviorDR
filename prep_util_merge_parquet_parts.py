"""
Memory-safe merge of part_*.parquet files into dr_data.parquet.
Uses pyarrow streaming — never loads all parts at once.
"""
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import time

t_start = time.time()
parts = sorted(Path('.').glob('_part_*.parquet'))
print(f'Found {len(parts)} part files')
if not parts:
    raise SystemExit('No _part_*.parquet files found')

# Peek schema from first part
first = pq.ParquetFile(parts[0])
schema = first.schema_arrow
print(f'Schema: {len(schema)} cols')
total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
print(f'Total rows to merge: {total_rows:,}')

# Stream write
OUTPUT = 'dr_data.parquet'
writer = pq.ParquetWriter(OUTPUT, schema, compression='snappy')
rows_written = 0
for i, p in enumerate(parts):
    pf = pq.ParquetFile(p)
    # Read + write in row-group chunks (pyarrow parquet default row group ~64MB)
    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx)
        writer.write_table(tbl)
        rows_written += tbl.num_rows
    print(f'  Merged {i+1}/{len(parts)}: {p.name} → running total {rows_written:,} rows '
          f'({time.time()-t_start:.0f}s)', flush=True)
writer.close()

size_gb = Path(OUTPUT).stat().st_size / 1e9
print(f'\nOutput: {OUTPUT} ({size_gb:.2f} GB, {rows_written:,} rows)')

# Clean up parts
for p in parts:
    p.unlink()
print(f'Cleaned up {len(parts)} part files')
print(f'Total time: {time.time()-t_start:.0f}s')