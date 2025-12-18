# FZFG + VIS<150m Alert Monitor (GitHub Pages)

This repository hosts a single-page web monitor (`index.html`) plus an automated data builder.

## How it works
- GitHub Pages serves `index.html`.
- A scheduled GitHub Action runs every 5 minutes (minimum supported) to rebuild `data/status.json`.
- The page reloads `data/status.json` every 30 seconds and raises an audible alarm if any station is `CRITICAL`.

## Customization
- Edit `data/extra_stations.txt` to add ICAO stations.
- Edit `data/exclude_stations.txt` to remove ICAO stations.
- If some airports cannot be mapped, check `data/unmapped_airports.txt` and add the ICAO code manually in `extra_stations.txt`.

## Notes
Mapping of destination airports to ICAO is best-effort (Wikipedia + OurAirports). Always validate for operational use.
