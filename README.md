# FZFG + VIS<150m Alert Monitor (GitHub Pages, Wizz-themed)

## What this repo does
- Single-page web monitor (`index.html`) hosted on GitHub Pages.
- A scheduled GitHub Action runs every **5 minutes** (minimum supported) and regenerates `data/status.json`.
- The page reloads `data/status.json` every 30 seconds and triggers an audible alarm if any station is `CRITICAL`.

## Alert rules
- **CRITICAL (METAR)**: `FZFG` present AND visibility `< 150 m`
- **WARNING (TAF)**: any TAF segment contains `FZFG` AND visibility `< 150 m`

## Airport list (Wizz network)
The workflow tries to build the station list from:
1) Wizz Air official website (best-effort; may block automation)
2) Wikipedia list of Wizz Air destinations (fallback)
3) Cached file in `data/wizz_airports_cache.csv`

If some airports cannot be mapped to ICAO, check:
- `data/unmapped_airports.txt`
and add any missing ICAO codes to:
- `data/extra_stations.txt` (one ICAO per line)

## Deployment
1) Create a GitHub repo and upload these files.
2) Enable **GitHub Pages**: Settings → Pages → Deploy from branch → `main` / root.
3) Enable Actions write access: Settings → Actions → Workflow permissions → **Read and write**.
4) Run Actions workflow once: Actions → "Update METAR/TAF data" → Run workflow.
