"""Run once: python download_airport_names.py. Then restart app.py."""
from pathlib import Path
from urllib.request import urlopen
import csv, io
url = "https://davidmegginson.github.io/ourairports-data/airports.csv"
with urlopen(url, timeout=60) as response:
    data = response.read()
rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
assert rows and {"iata_code", "name", "municipality", "iso_country"}.issubset(rows[0]), "Unexpected reference file format"
Path(__file__).with_name("airports.csv").write_bytes(data)
print("Airport names downloaded. Restart app.py to load them.")
