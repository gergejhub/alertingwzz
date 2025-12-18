import io
#!/usr/bin/env python3

"""
Build data/status.json for GitHub Pages.

Strategy (GitHub-only):
- Pull "List of Wizz Air destinations" from Wikipedia.
- Filter out terminated/closed airports.
- Map airport names to ICAO codes using OurAirports dataset.
- Fetch METAR + TAF (raw) from aviationweather.gov Data API (server-side).
- Evaluate alert: FZFG + VIS<150m
  * CRITICAL: METAR trigger
  * WARNING: TAF segments trigger

Notes:
- Mapping is "best effort". Any unmapped airports are written to data/unmapped_airports.txt.
- You can add extra ICAO stations in data/extra_stations.txt (one per line) if needed.
- You can exclude ICAO stations in data/exclude_stations.txt (one per line).
"""

import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pycountry
import requests


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_Wizz_Air_destinations"
OURAIRPORTS_CSV = "https://ourairports.com/airports.csv"

METAR_URL = "https://aviationweather.gov/api/data/metar"
TAF_URL = "https://aviationweather.gov/api/data/taf"

THRESHOLD_M = 150
CHUNK = 90  # keep URL lengths sane


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)  # remove parentheses
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # strip generic words
    drop = {
        "international", "airport", "aerodrome", "airfield", "aeroport", "aéroport", "flughafen",
        "aeropuerto", "aeroporto", "aerodrom", "aeroporti", "terminal"
    }
    toks = [t for t in s.split(" ") if t and t not in drop]
    return " ".join(toks)


def country_to_iso2(name: str) -> Optional[str]:
    if not name:
        return None
    # common Wikipedia country label quirks
    fixes = {
        "North Macedonia": "North Macedonia",
        "United Kingdom (England)": "United Kingdom",
        "United Kingdom (Scotland)": "United Kingdom",
        "United Kingdom (Wales)": "United Kingdom",
        "United Kingdom (Northern Ireland)": "United Kingdom",
        "Russia": "Russian Federation",
        "Moldova": "Moldova",
        "Türkiye": "Turkey",
    }
    name = fixes.get(name, name)
    try:
        c = pycountry.countries.lookup(name)
        return c.alpha_2
    except Exception:
        return None


@dataclass
class AirportRow:
    ident: str
    name: str
    iso_country: str
    municipality: str
    type: str
    iata: str


def load_wizz_airports() -> pd.DataFrame:
    # Wikipedia tables are parseable via read_html
    tables = pd.read_html(WIKI_URL)
    # choose the first table with an "Airport" column
    t = None
    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        if "airport" in cols and "country" in cols:
            t = tbl
            break
    if t is None:
        raise RuntimeError("Could not find destination table on Wikipedia.")

    # Standardize column names
    t.columns = [str(c).strip() for c in t.columns]
    # Filter out terminated/closed
    if "Notes" in t.columns:
        notes = t["Notes"].astype(str)
        mask = ~notes.str.contains("Terminated", case=False, na=False) & ~notes.str.contains("Airport closed", case=False, na=False)
        t = t[mask]
    # keep essential cols
    keep = [c for c in ["Country", "Town", "Airport"] if c in t.columns]
    t = t[keep].copy()
    # Clean airport name (remove footnote brackets etc.)
    t["Airport"] = t["Airport"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    t["Town"] = t["Town"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    t["Country"] = t["Country"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    return t


def load_ourairports() -> List[AirportRow]:
    r = requests.get(OURAIRPORTS_CSV, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Filter to airports with ICAO-like ident (4 letters)
    df = df[df["ident"].astype(str).str.match(r"^[A-Z]{4}$", na=False)].copy()
    df["name"] = df["name"].astype(str)
    df["iso_country"] = df["iso_country"].astype(str)
    df["municipality"] = df["municipality"].fillna("").astype(str)
    df["type"] = df["type"].fillna("").astype(str)
    df["iata_code"] = df["iata_code"].fillna("").astype(str)

    out: List[AirportRow] = []
    for _, row in df.iterrows():
        out.append(
            AirportRow(
                ident=row["ident"],
                name=row["name"],
                iso_country=row["iso_country"],
                municipality=row["municipality"],
                type=row["type"],
                iata=row["iata_code"],
            )
        )
    return out


def build_name_index(airports: List[AirportRow]) -> Dict[str, List[AirportRow]]:
    idx: Dict[str, List[AirportRow]] = {}
    for a in airports:
        key = normalize_text(a.name)
        idx.setdefault(key, []).append(a)
    return idx


def pick_best(cands: List[AirportRow], iso2: Optional[str], town: str) -> Optional[AirportRow]:
    if not cands:
        return None

    # filter by country first if possible
    if iso2:
        c2 = [c for c in cands if c.iso_country == iso2]
        if c2:
            cands = c2

    # prefer large > medium > small (roughly)
    rank_type = {"large_airport": 3, "medium_airport": 2, "small_airport": 1}
    town_n = normalize_text(town)
    def score(a: AirportRow) -> Tuple[int, int]:
        tscore = rank_type.get(a.type, 0)
        mscore = 1 if town_n and town_n in normalize_text(a.municipality) else 0
        return (tscore, mscore)

    cands = sorted(cands, key=score, reverse=True)
    return cands[0] if cands else None


def fuzzy_match(target_name: str, airports: List[AirportRow], iso2: Optional[str]) -> Optional[AirportRow]:
    # best-effort fuzzy search within country (if known)
    tn = normalize_text(target_name)
    if not tn:
        return None
    pool = airports
    if iso2:
        pool = [a for a in airports if a.iso_country == iso2]

    best = None
    best_ratio = 0.0
    # heuristic: compare only airports that share a token
    tokens = set(tn.split())
    for a in pool:
        an = normalize_text(a.name)
        if not tokens.intersection(set(an.split())):
            continue
        # simple ratio
        ratio = _quick_ratio(tn, an)
        if ratio > best_ratio:
            best_ratio = ratio
            best = a

    if best and best_ratio >= 0.86:
        return best
    return None


def _quick_ratio(a: str, b: str) -> float:
    # lightweight similarity (no external libs)
    # Jaccard on tokens + prefix bonus
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / len(ta | tb)
    bonus = 0.08 if b.startswith(a[: min(8, len(a))]) else 0.0
    return min(1.0, j + bonus)


def resolve_stations(wizz_df: pd.DataFrame, airports: List[AirportRow]) -> Tuple[List[str], List[str]]:
    idx = build_name_index(airports)
    unmapped = []
    stations = set()

    for _, row in wizz_df.iterrows():
        country = str(row.get("Country", "")).strip()
        town = str(row.get("Town", "")).strip()
        ap_name = str(row.get("Airport", "")).strip()

        iso2 = country_to_iso2(country)
        key = normalize_text(ap_name)

        cands = idx.get(key, [])
        picked = pick_best(cands, iso2, town)

        if not picked:
            # fuzzy fallback
            picked = fuzzy_match(ap_name, airports, iso2)

        if picked:
            stations.add(picked.ident)
        else:
            unmapped.append(f"{country} | {town} | {ap_name}")

    return sorted(stations), unmapped


def chunks(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def fetch_raw(url: str, ids: List[str]) -> str:
    params = {"ids": ",".join(ids), "format": "raw"}
    r = requests.get(url, params=params, timeout=60, headers={"User-Agent": "wizz-fzfg-alert/1.0"})
    r.raise_for_status()
    return r.text or ""


def parse_metar_lines(txt: str) -> Dict[str, str]:
    out = {}
    for line in (txt or "").splitlines():
        t = line.strip()
        if not t:
            continue
        icao = t.split()[0].upper()
        if len(icao) == 4:
            out[icao] = t
    return out


def parse_taf_blocks(txt: str) -> Dict[str, str]:
    # blocks separated by blank lines
    blocks = []
    cur = []
    for line in (txt or "").splitlines():
        if line.strip():
            cur.append(line.strip())
        else:
            if cur:
                blocks.append(" ".join(cur))
                cur = []
    if cur:
        blocks.append(" ".join(cur))

    out = {}
    for b in blocks:
        toks = b.split()
        if not toks:
            continue
        icao = toks[0].upper()
        if icao == "TAF" and len(toks) > 1:
            icao = toks[1].upper()
        if len(icao) == 4:
            out[icao] = b
    return out


def vis_to_meters(raw: str) -> Optional[int]:
    if not raw:
        return None
    if "CAVOK" in raw:
        return 9999

    # METAR/TAF: 4-digit meters visibility (e.g., 0150, 0100, 9999)
    m = re.search(r"\b(\d{4})\b", raw)
    if m:
        return int(m.group(1), 10)

    # US-style statute miles e.g. 1/4SM, M1/4SM, 2SM
    m2 = re.search(r"\b(M)?(\d+)?(?:\s?(\d)/(\d))?SM\b", raw)
    # Examples:
    # 1/4SM -> groups: M? no, \d+? None, frac 1/4
    # 2SM -> \d+ = 2, frac None
    # 1 1/2SM (rare without space) not handled; but 1 1/2SM appears as 1 1/2SM with space; not common in AWC raw for non-US
    if m2:
        less = bool(m2.group(1))
        whole = m2.group(2)
        num = m2.group(3)
        den = m2.group(4)

        miles = 0.0
        if whole:
            miles += float(whole)
        if num and den:
            miles += float(num) / float(den)

        if miles > 0:
            meters = int(round(miles * 1609.344))
            # "M" means less than; still use computed meters
            return max(0, meters)

    return None


def split_taf_segments(taf: str) -> List[str]:
    if not taf:
        return []
    s = re.sub(r"\s+", " ", taf).strip()
    parts = re.split(r"\b(FM\d{6}|BECMG|TEMPO|PROB\d{2})\b", s)
    segs = []
    cur = ""
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"(FM\d{6}|BECMG|TEMPO|PROB\d{2})", p):
            if cur.strip():
                segs.append(cur.strip())
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur.strip():
        segs.append(cur.strip())
    return segs


def eval_metar(metar: Optional[str]) -> Tuple[Optional[int], bool, bool]:
    if not metar:
        return None, False, False
    fzfg = "FZFG" in metar
    vis = vis_to_meters(metar)
    trigger = (vis is not None) and (vis < THRESHOLD_M) and fzfg
    return vis, fzfg, trigger


def eval_taf(taf: Optional[str]) -> bool:
    if not taf:
        return False
    for seg in split_taf_segments(taf):
        if "FZFG" not in seg:
            continue
        vis = vis_to_meters(seg)
        if vis is not None and vis < THRESHOLD_M:
            return True
    return False


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    wizz = load_wizz_airports()
    airports = load_ourairports()
    stations, unmapped = resolve_stations(wizz, airports)

    # apply extras / excludes
    extras = read_lines(os.path.join(data_dir, "extra_stations.txt"))
    excludes = set(read_lines(os.path.join(data_dir, "exclude_stations.txt")))
    stations = sorted(set(stations).union(set(extras)) - excludes)

    # write stations list for transparency
    with open(os.path.join(data_dir, "stations_icao.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(stations) + ("\n" if stations else ""))

    with open(os.path.join(data_dir, "unmapped_airports.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(unmapped) + ("\n" if unmapped else ""))

    # fetch METAR/TAF in chunks
    metar_map: Dict[str, str] = {}
    taf_map: Dict[str, str] = {}

    for ch in chunks(stations, CHUNK):
        metar_txt = fetch_raw(METAR_URL, ch)
        taf_txt = fetch_raw(TAF_URL, ch)
        metar_map.update(parse_metar_lines(metar_txt))
        taf_map.update(parse_taf_blocks(taf_txt))
        time.sleep(0.4)

    rows = []
    for icao in stations:
        metar = metar_map.get(icao)
        taf = taf_map.get(icao)
        vis, fzfg, trig = eval_metar(metar)
        taf_trig = eval_taf(taf)
        alert = "CRITICAL" if trig else ("WARNING" if taf_trig else "OK")
        rows.append({
            "icao": icao,
            "alert": alert,
            "vis_m": vis,
            "fzfg": bool(fzfg or (taf and "FZFG" in taf)),
            "metar": metar or "",
            "taf": taf or "",
        })

    payload = {
        "ts_utc": utc_now_iso(),
        "threshold_m": THRESHOLD_M,
        "source": "Wikipedia (destinations) + OurAirports (ICAO mapping) + aviationweather.gov (METAR/TAF)",
        "station_count": len(stations),
        "stations": rows,
    }

    with open(os.path.join(data_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Built data/status.json with {len(stations)} stations. Unmapped: {len(unmapped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
