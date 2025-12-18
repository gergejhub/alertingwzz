# Wizz FZFG + VIS<150m Alert Monitor (v3)

## What this repo does
- Single-page monitor (`index.html`) hosted on GitHub Pages.
- A scheduled GitHub Action regenerates `data/status.json` every **5 minutes** and commits it back.

## Why v3
If the page shows **0 monitored stations** and the source says "Not built yet", it means `data/status.json` was never updated/committed (permissions or push issues).
v3 includes:
- `data/stations_override.txt`: if you list ICAO codes here, the build uses them directly (no scraping/mapping).
- Wikipedia + OurAirports mapping fallback if override is empty.
- More robust workflow push.

## Required GitHub settings
- Settings → Actions → General → Workflow permissions → **Read and write permissions**
- Settings → Pages → Deploy from branch → `main` / root

## Station list control (recommended)
Put ICAO codes in:
- `data/stations_override.txt` (one per line)

Optional:
- `data/extra_stations.txt`
- `data/exclude_stations.txt`

## Alert rules
- CRITICAL = METAR has FZFG AND VIS < 150 m
- WARNING = any TAF segment has FZFG AND VIS < 150 m
