from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("external/Club-Football-Match-Data-2000-2025")
MATCHES_CSV = DATA_DIR / "data" / "MATCHES.csv"

def main():
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"Missing {MATCHES_CSV}. Clone the dataset into external/.")

    # Read once; disable low_memory for mixed-type columns
    df = pd.read_csv(MATCHES_CSV, low_memory=False)

    # Schema sanity check
    must_have = ["MatchDate", "FTResult", "HomeTeam", "AwayTeam"]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise RuntimeError(f"MATCHES.csv missing required columns: {missing}")

    # Keep only the columns we need (use .get so code doesn't break if some stats are absent)
    X = pd.DataFrame({
        "date":      pd.to_datetime(df["MatchDate"], errors="coerce"),
        "home":      df["HomeTeam"],
        "away":      df["AwayTeam"],
        "result":    df["FTResult"].astype(str).str.upper(),
        "elo_h":     df.get("HomeElo"),
        "elo_a":     df.get("AwayElo"),
        "form3_h":   df.get("Form3Home"),
        "form3_a":   df.get("Form3Away"),
        "form5_h":   df.get("Form5Home"),
        "form5_a":   df.get("Form5Away"),
        "shots_h":   df.get("HomeShots"),
        "shots_a":   df.get("AwayShots"),
        "sot_h":     df.get("HomeTarget"),
        "sot_a":     df.get("AwayTarget"),
        "corners_h": df.get("HomeCorners"),
        "corners_a": df.get("AwayCorners"),
        "yc_h":      df.get("HomeYellow"),
        "yc_a":      df.get("AwayYellow"),
        "rc_h":      df.get("HomeRed"),
        "rc_a":      df.get("AwayRed"),
        "season":    df.get("Season"),
        "league":    df.get("Division"),
    }).dropna(subset=["date", "result"])

    # Encode y in {1,2,3} = {AwayWin, Draw, HomeWin}
    map_features = {"A": 1, "D": 2, "H": 3}
    X["y"] = X["result"].map(map_features).astype("Int64")
    X = X.dropna(subset=["y"])

    # Helper: safe numeric conversion
    def fnum(s): 
        return pd.to_numeric(s, errors="coerce")

    # Difference features (home - away), only from pre-match info
    X["d_elo"]   = fnum(X["elo_h"])   - fnum(X["elo_a"])
    X["d_form3"] = fnum(X["form3_h"]) - fnum(X["form3_a"])
    X["d_form5"] = fnum(X["form5_h"]) - fnum(X["form5_a"])
    # Add difference features only if both sides are present
    for h, a, n in [
        ("shots_h","shots_a","d_shots"),
        ("sot_h","sot_a","d_sot"),
        ("corners_h","corners_a","d_corners"),
        ("yc_h","yc_a","d_yc"),
        ("rc_h","rc_a","d_rc"),
    ]:
        if h in X and a in X:
            X[n] = fnum(X[h]) - fnum(X[a])

    # Year for chronological split (prefer season if present)
    if X["season"].notna().any():
        year = X["season"].astype(str).str.extract(r"(\d{4})")[0]
        X["year"] = pd.to_numeric(year, errors="coerce").fillna(X["date"].dt.year).astype(int)
    else:
        X["year"] = X["date"].dt.year.astype(int)

    train      = X[X["year"] <= 2022]
    validation = X[X["year"] == 2023]
    test       = X[X["year"] == 2024]

    feature_cols = [c for c in X.columns if c.startswith("d_")]

    print("Total usable matches:", len(X))
    print("Train/Validation/Test:", len(train), len(validation), len(test))
    print("Train class balance:", train["y"].value_counts(normalize=True).sort_index().to_dict())
    print("Features used:", feature_cols)

    # Save a thin CSV for inspection
    out_cols = ["date","home","away","y","year"] + feature_cols
    Path("data").mkdir(exist_ok=True, parents=True)
    X[out_cols].to_csv("data/stage1_features.csv", index=False)

if __name__ == "__main__":
    main()
