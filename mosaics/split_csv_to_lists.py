import csv
import os
from collections import defaultdict

# Input CSV file
csv_file = '/home/rfrench/Downloads/pdsrms-2025-07-24T12-55-20-262772-data-n.csv'
# Output directory
outdir = 'mosaics/ring_mosaic_lists/FILELIST_FMOVIE_FMONITOR'
os.makedirs(outdir, exist_ok=True)

data = defaultdict(list)

with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 2:
            data[row[0]].append(row[1])

for key, values in data.items():
    outpath = os.path.join(outdir, f'{key}.list')
    with open(outpath, 'w') as out:
        out.write('\n'.join(values) + '\n')