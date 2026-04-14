#!/usr/bin/env python3
"""
scrape_royaleapi.py
Fetch card-level community statistics from RoyaleAPI (second data source).
DSA210 Term Project

This script fetches per-card win rates and usage rates from RoyaleAPI's
public card statistics page, then enriches features.csv with real
deck meta scores and a new deck_popularity_score feature.

Usage:
    pip install requests beautifulsoup4
    python scrape_royaleapi.py

Outputs:
    data/card_stats_royaleapi.csv   — per-card community stats
    data/features.csv               — updated with real meta scores

Data Source: https://royaleapi.com/cards/popular (Ladder, last 7 days)
"""

import os
import sys
import json
import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# ── Configuration ──────────────────────────────────────────────────────────
CARDS_URL    = "https://royaleapi.com/cards/popular?cat=Ladder&mode=grid&time=7d&sort=win"
RAW_PATH     = "data/battles_raw.csv"
FEAT_PATH    = "data/features.csv"
STATS_PATH   = "data/card_stats_royaleapi.csv"
CR_API_KEY   = os.getenv("CR_API_KEY", "")
CR_API_URL   = "https://api.clashroyale.com/v1"

HEADERS_WEB = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Step 1: Fetch card stats from RoyaleAPI ────────────────────────────────
def scrape_card_stats() -> pd.DataFrame:
    """
    Scrape card win rates and usage rates from RoyaleAPI cards page.
    Returns DataFrame with columns: card_name, win_rate, usage_rate
    """
    print(f"  Fetching {CARDS_URL} ...")
    try:
        r = requests.get(CARDS_URL, headers=HEADERS_WEB, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] Failed to fetch RoyaleAPI: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    cards = []

    # Method 1: Look for card grid items with stats
    # RoyaleAPI uses various HTML structures; try multiple selectors
    for item in soup.select(".cards__card, .card_item, [class*='card']"):
        name_el = item.select_one(".card_name, .cards__card__name, [class*='name']")
        if not name_el:
            continue
        name = name_el.get_text(strip=True)
        if not name or len(name) < 2:
            continue

        # Extract percentages from text
        text = item.get_text(" ", strip=True)
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)

        if len(percentages) >= 2:
            cards.append({
                "card_name": name,
                "win_rate":  float(percentages[0]) / 100,
                "usage_rate": float(percentages[1]) / 100,
            })
        elif len(percentages) == 1:
            cards.append({
                "card_name": name,
                "win_rate":  float(percentages[0]) / 100,
                "usage_rate": np.nan,
            })

    # Method 2: Try finding a data table
    if not cards:
        for row in soup.select("tr"):
            cells = row.select("td")
            if len(cells) >= 3:
                name = cells[0].get_text(strip=True)
                percentages = []
                for cell in cells[1:]:
                    match = re.search(r'(\d+(?:\.\d+)?)\s*%', cell.get_text())
                    if match:
                        percentages.append(float(match.group(1)) / 100)
                if name and percentages:
                    cards.append({
                        "card_name": name,
                        "win_rate":  percentages[0] if len(percentages) > 0 else np.nan,
                        "usage_rate": percentages[1] if len(percentages) > 1 else np.nan,
                    })

    # Method 3: Extract from script tags (JSON data)
    if not cards:
        for script in soup.select("script"):
            text = script.string or ""
            # Look for JSON-like card data
            matches = re.findall(
                r'"name"\s*:\s*"([^"]+)"[^}]*?"winRate"\s*:\s*(\d+(?:\.\d+)?)',
                text
            )
            for name, wr in matches:
                cards.append({
                    "card_name": name,
                    "win_rate":  float(wr) / 100 if float(wr) > 1 else float(wr),
                    "usage_rate": np.nan,
                })

    if not cards:
        print("  [WARN] Could not parse card stats from HTML.")
        print("         RoyaleAPI may use JavaScript rendering.")
        print("         Falling back to manual CSV method...")
        return pd.DataFrame()

    df = pd.DataFrame(cards).drop_duplicates(subset="card_name")
    return df


# ── Step 2: Map card names ↔ card IDs via CR API ──────────────────────────
def get_card_name_to_id_map() -> dict:
    """
    Fetch all cards from CR API, return {normalized_name: card_id} dict.
    """
    if not CR_API_KEY:
        print("  [WARN] CR_API_KEY not set — cannot map card names to IDs.")
        return {}

    try:
        r = requests.get(
            f"{CR_API_URL}/cards",
            headers={"Authorization": f"Bearer {CR_API_KEY}"},
            timeout=10,
        )
        if r.status_code == 200:
            items = r.json().get("items", [])
            mapping = {}
            for c in items:
                name = c.get("name", "")
                cid  = c.get("id", 0)
                # Store multiple normalized forms for matching
                mapping[name.lower().strip()] = cid
                mapping[name.lower().replace(" ", "").strip()] = cid
                mapping[name.lower().replace("-", " ").strip()] = cid
            return mapping
        else:
            print(f"  [WARN] CR API returned {r.status_code}")
    except Exception as e:
        print(f"  [ERROR] CR API: {e}")
    return {}


def match_card_name(name: str, name_map: dict) -> int:
    """Fuzzy match a card name to its ID."""
    normalized = name.lower().strip()
    if normalized in name_map:
        return name_map[normalized]
    # Try without spaces
    no_space = normalized.replace(" ", "")
    if no_space in name_map:
        return name_map[no_space]
    # Try partial match
    for key, cid in name_map.items():
        if normalized in key or key in normalized:
            return cid
    return 0


# ── Step 3: Update features.csv ───────────────────────────────────────────
def update_features(card_stats: dict, usage_stats: dict):
    """
    Recompute deck_meta_score and add deck_popularity_score
    using RoyaleAPI community data.
    """
    if not os.path.exists(RAW_PATH):
        print(f"  [ERROR] {RAW_PATH} not found. Run collect_data.py first.")
        return

    df_raw  = pd.read_csv(RAW_PATH)
    df_feat = pd.read_csv(FEAT_PATH) if os.path.exists(FEAT_PATH) else None

    def deck_meta_score(card_ids_json):
        try:
            ids   = json.loads(card_ids_json)
            rates = [card_stats.get(str(cid), 0.5) for cid in ids]
            return float(np.mean(rates))
        except:
            return 0.5

    def deck_popularity_score(card_ids_json):
        try:
            ids   = json.loads(card_ids_json)
            rates = [usage_stats.get(str(cid), 0.0) for cid in ids]
            return float(np.mean(rates))
        except:
            return 0.0

    # Compute on raw data
    df_raw["team_deck_meta_score"] = df_raw["team_card_ids"].apply(deck_meta_score)
    df_raw["opp_deck_meta_score"]  = df_raw["opp_card_ids"].apply(deck_meta_score)
    df_raw["deck_meta_score_diff"] = df_raw["team_deck_meta_score"] - df_raw["opp_deck_meta_score"]

    df_raw["team_deck_popularity"] = df_raw["team_card_ids"].apply(deck_popularity_score)
    df_raw["opp_deck_popularity"]  = df_raw["opp_card_ids"].apply(deck_popularity_score)
    df_raw["deck_popularity_diff"] = df_raw["team_deck_popularity"] - df_raw["opp_deck_popularity"]

    # Update features.csv
    if df_feat is not None and len(df_feat) == len(df_raw):
        for col in ["team_deck_meta_score", "opp_deck_meta_score", "deck_meta_score_diff",
                     "team_deck_popularity", "opp_deck_popularity", "deck_popularity_diff"]:
            df_feat[col] = df_raw[col].values
        df_feat.to_csv(FEAT_PATH, index=False)
        print(f"  ✓ {FEAT_PATH} updated ({len(df_feat)} rows, {len(df_feat.columns)} cols)")
    else:
        print("  [WARN] features.csv row count mismatch — regenerate with collect_data.py")


# ── Fallback: Manual CSV input ─────────────────────────────────────────────
MANUAL_INSTRUCTIONS = """
═══════════════════════════════════════════════════════
  MANUAL FALLBACK — Copy card stats from RoyaleAPI
═══════════════════════════════════════════════════════

If automatic scraping failed, you can manually create
data/card_stats_royaleapi.csv with the following format:

  card_name,win_rate,usage_rate
  Goblin Barrel,0.56,0.18
  Log,0.52,0.35
  Hog Rider,0.51,0.22
  ...

Steps:
1. Open https://royaleapi.com/cards/popular?cat=Ladder
2. For each card, note win rate and usage rate
3. Save as data/card_stats_royaleapi.csv
4. Re-run: python scrape_royaleapi.py

The script will detect the CSV and use it.
"""


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  RoyaleAPI Card Stats Enrichment — DSA210")
    print("=" * 55)

    os.makedirs("data", exist_ok=True)
    card_stats = {}   # card_id -> win_rate
    usage_stats = {}  # card_id -> usage_rate

    # Check if manual CSV already exists
    if os.path.exists(STATS_PATH):
        print(f"\nFound existing {STATS_PATH} — loading ...")
        df_stats = pd.read_csv(STATS_PATH)
        print(f"  → {len(df_stats)} cards loaded")

        # Map names to IDs
        if "card_id" in df_stats.columns:
            for _, row in df_stats.iterrows():
                cid = str(int(row["card_id"]))
                card_stats[cid]  = row.get("win_rate", 0.5)
                usage_stats[cid] = row.get("usage_rate", 0.0)
        else:
            print("  Mapping card names to IDs via CR API ...")
            name_map = get_card_name_to_id_map()
            if name_map:
                for _, row in df_stats.iterrows():
                    cid = match_card_name(row["card_name"], name_map)
                    if cid:
                        card_stats[str(cid)]  = row.get("win_rate", 0.5)
                        usage_stats[str(cid)] = row.get("usage_rate", 0.0)
                print(f"  → {len(card_stats)} cards matched to IDs")
    else:
        # Try scraping
        print("\nScraping card stats from RoyaleAPI ...")
        df_stats = scrape_card_stats()

        if df_stats.empty:
            print(MANUAL_INSTRUCTIONS)
            sys.exit(0)

        # Save scraped stats
        df_stats.to_csv(STATS_PATH, index=False)
        print(f"  ✓ Saved {len(df_stats)} cards to {STATS_PATH}")

        # Map to IDs
        print("  Mapping card names to IDs via CR API ...")
        name_map = get_card_name_to_id_map()
        if name_map:
            df_stats["card_id"] = df_stats["card_name"].apply(
                lambda n: match_card_name(n, name_map))
            df_stats.to_csv(STATS_PATH, index=False)

            for _, row in df_stats.iterrows():
                cid = str(int(row["card_id"])) if row["card_id"] else "0"
                if cid != "0":
                    card_stats[cid]  = row.get("win_rate", 0.5)
                    usage_stats[cid] = row.get("usage_rate", 0.0)
            print(f"  → {len(card_stats)} cards matched")

    if not card_stats:
        print("\n  [ERROR] No card stats available. See manual instructions above.")
        sys.exit(1)

    # Summary
    rates = list(card_stats.values())
    print(f"\n  Card Win Rate Summary:")
    print(f"    Cards tracked : {len(card_stats)}")
    print(f"    Win rate range: {min(rates):.3f} — {max(rates):.3f}")
    print(f"    Win rate std  : {np.std(rates):.4f}")

    # Update features
    print(f"\nUpdating {FEAT_PATH} with RoyaleAPI community data ...")
    update_features(card_stats, usage_stats)

    # Verify
    if os.path.exists(FEAT_PATH):
        df = pd.read_csv(FEAT_PATH)
        diff = df.get("deck_meta_score_diff")
        if diff is not None:
            print(f"\n  deck_meta_score_diff (RoyaleAPI):")
            print(f"    mean = {diff.mean():.5f}")
            print(f"    std  = {diff.std():.5f}")
            print(f"    min  = {diff.min():.5f}")
            print(f"    max  = {diff.max():.5f}")

        pop = df.get("deck_popularity_diff")
        if pop is not None:
            print(f"\n  deck_popularity_diff (NEW):")
            print(f"    mean = {pop.mean():.5f}")
            print(f"    std  = {pop.std():.5f}")

    print("\n✓ Done. Now re-run: python analysis.py")


if __name__ == "__main__":
    main()
