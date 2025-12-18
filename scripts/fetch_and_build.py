def load_wizz_airports() -> pd.DataFrame:
    # Fetch HTML with a browser-like User-Agent to avoid 403 in GitHub Actions
    r = requests.get(
        WIKI_URL,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (GitHubActions) wizz-fzfg-alert/1.0"}
    )
    r.raise_for_status()

    tables = pd.read_html(r.text)

    t = None
    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        if "airport" in cols and "country" in cols:
            t = tbl
            break
    if t is None:
        raise RuntimeError("Could not find destination table on Wikipedia (blocked or columns changed).")

    t.columns = [str(c).strip() for c in t.columns]

    if "Notes" in t.columns:
        notes = t["Notes"].astype(str)
        mask = (
            ~notes.str.contains("Terminated", case=False, na=False)
            & ~notes.str.contains("Airport closed", case=False, na=False)
        )
        t = t[mask]

    keep = [c for c in ["Country", "Town", "Airport"] if c in t.columns]
    t = t[keep].copy()

    t["Airport"] = t["Airport"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    t["Town"] = t["Town"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    t["Country"] = t["Country"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()
    return t

