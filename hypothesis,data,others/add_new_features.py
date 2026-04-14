#!/usr/bin/env python3
"""
add_new_features.py
Clash Royale — Add New Features Without Re-Collecting Data
DSA210 Term Project

Adds to the existing battles_raw.csv and features.csv:
  • underleveled_penalty  = avg(maxLevel - cardLevel) per deck
                           Requires maxLevel in battles_raw; if absent, approximates
                           from card rarity via /cards API endpoint.
  • player_win_rate       = wins / (wins + losses) per player tag
                           Requires wins/losses in battles_raw; if absent, fetches
                           from CR API (needs CR_API_KEY env var).

Usage:
    python add_new_features.py
    python add_new_features.py --no-api   # skip API calls, use approximations only
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd

API_KEY  = os.getenv("CR_API_KEY", "")
BASE_URL = "https://api.clashroyale.com/v1"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
RATE_LIMIT_SLEEP = 0.20

RAW_PATH  = "data/battles_raw.csv"
FEAT_PATH = "data/features.csv"

FEATURE_COLS = [
    "outcome",
    "trophy_diff", "elixir_diff", "level_diff", "deck_meta_score_diff",
    "team_avg_elixir", "opp_avg_elixir",
    "team_avg_level", "opp_avg_level",
    "team_deck_meta_score", "opp_deck_meta_score",
    "team_underleveled_penalty", "opp_underleveled_penalty", "underleveled_diff",
    "exp_level", "battle_count", "player_win_rate",
    "hour_of_day", "trophy_diff_category",
    "team_deck_popularity", "opp_deck_popularity", "deck_popularity_diff",
]


# ── API helpers ────────────────────────────────────────────────────────────
def api_get(path: str):
    try:
        r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        print(f"  [WARN] {r.status_code} for {path}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    return None


def enc(tag: str) -> str:
    return tag.replace("#", "%23")


# ── Card rarity → maxLevel mapping ───────────────────────────────────────
RARITY_MAX = {
    "Common": 14, "Rare": 12, "Epic": 10,
    "Legendary": 9, "Champion": 11,
}

def build_card_maxlevel_map() -> dict:
    """Fetch /cards and return {card_id: maxLevel}."""
    data = api_get("/cards")
    if not data:
        print("  [WARN] Could not fetch /cards — using default maxLevel=14")
        return {}
    mapping = {}
    for c in data.get("items", []):
        cid      = str(c.get("id", ""))
        rarity   = c.get("rarity", "Common")
        max_lvl  = RARITY_MAX.get(rarity, 14)
        mapping[cid] = max_lvl
    print(f"  → Card maxLevel map built for {len(mapping)} cards")
    return mapping


# ── Underleveled penalty ──────────────────────────────────────────────────
def compute_underleveled(card_ids_json: str, avg_level: float,
                         max_map: dict) -> float:
    """
    If max_map is populated: compute avg(maxLevel - level) per card ID.
    avg_level alone can't give per-card differences, so we approximate:
      penalty ≈ avg_maxLevel_for_deck - avg_level
    This is still much better than raw avg_level.
    """
    try:
        ids = json.loads(card_ids_json)
        max_levels = [max_map.get(str(cid), 14) for cid in ids]
        avg_max = float(np.mean(max_levels))
        return round(avg_max - avg_level, 4)
    except Exception:
        return round(14.0 - avg_level, 4)   # fallback: assume all Common (max=14)


# ── Player win rates ──────────────────────────────────────────────────────
def fetch_player_win_rates(tags: list) -> dict:
    """Fetch wins/losses from /players endpoint for each unique tag."""
    wr_map = {}
    print(f"  Fetching win/loss stats for {len(tags)} unique players ...")
    for i, tag in enumerate(tags, 1):
        time.sleep(RATE_LIMIT_SLEEP)
        data = api_get(f"/players/{enc(tag)}")
        if data:
            w = data.get("wins",   0)
            l = data.get("losses", 0)
            total = w + l
            wr_map[tag] = w / total if total > 0 else np.nan
        if i % 10 == 0 or i == len(tags):
            print(f"    {i}/{len(tags)} done")
    return wr_map


# ── Trophy category ───────────────────────────────────────────────────────
def trophy_diff_category(diff) -> str:
    if pd.isna(diff):  return "unknown"
    if diff > 10:      return "higher"
    if diff < -10:     return "lower"
    return "equal"


# ── Main ──────────────────────────────────────────────────────────────────
def main(no_api: bool):
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: {RAW_PATH} not found. Run collect_data.py first.")
        sys.exit(1)

    print(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"  → {len(df)} rows, columns: {list(df.columns)}")

    # ── Underleveled penalty ─────────────────────────────────────────────
    print("\n[1/3] Computing underleveled penalty ...")
    if no_api:
        print("  (--no-api mode: using maxLevel=14 approximation for all cards)")
        max_map = {}
    else:
        max_map = build_card_maxlevel_map()

    df["team_underleveled_penalty"] = df.apply(
        lambda r: compute_underleveled(r["team_card_ids"], r["team_avg_level"], max_map), axis=1)
    df["opp_underleveled_penalty"]  = df.apply(
        lambda r: compute_underleveled(r["opp_card_ids"],  r["opp_avg_level"],  max_map), axis=1)
    df["underleveled_diff"] = df["team_underleveled_penalty"] - df["opp_underleveled_penalty"]

    print(f"  underleveled_diff: mean={df['underleveled_diff'].mean():.4f}, "
          f"std={df['underleveled_diff'].std():.4f}")

    # ── Player win rate ──────────────────────────────────────────────────
    print("\n[2/3] Adding player win rate ...")

    # Check if already in raw (from updated collect_data.py)
    if "player_win_rate" in df.columns and df["player_win_rate"].notna().sum() > 0:
        print(f"  player_win_rate already in raw data "
              f"({df['player_win_rate'].notna().sum()} non-null values)")
    elif no_api or not API_KEY:
        print("  [SKIP] No API key or --no-api mode. player_win_rate set to NaN.")
        df["player_win_rate"] = np.nan
    else:
        unique_tags = df["player_tag"].dropna().unique().tolist()
        wr_map = fetch_player_win_rates(unique_tags)
        df["player_win_rate"] = df["player_tag"].map(wr_map)
        non_null = df["player_win_rate"].notna().sum()
        print(f"  → {non_null}/{len(df)} rows have player_win_rate")
        print(f"  Win rate stats: mean={df['player_win_rate'].mean():.3f}, "
              f"std={df['player_win_rate'].std():.3f}")

    # ── Trophy diff category (updated threshold) ──────────────────────
    print("\n[3/3] Updating trophy_diff_category to ±10 threshold ...")
    df["trophy_diff_category"] = df["trophy_diff"].apply(trophy_diff_category)
    print(f"  Distribution:\n{df['trophy_diff_category'].value_counts().to_string()}")

    # ── Save updated raw ────────────────────────────────────────────────
    df.to_csv(RAW_PATH, index=False)
    print(f"\n✓ {RAW_PATH} updated ({len(df)} rows, {len(df.columns)} cols)")

    # ── Rebuild features.csv ────────────────────────────────────────────
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_feat = df[available].reset_index(drop=True)
    df_feat.to_csv(FEAT_PATH, index=False)
    print(f"✓ {FEAT_PATH} rebuilt ({len(df_feat)} rows, {len(df_feat.columns)} cols)")
    print(f"  Columns: {list(df_feat.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-api", action="store_true",
                        help="Skip API calls, use approximations only")
    args = parser.parse_args()
    main(no_api=args.no_api)
