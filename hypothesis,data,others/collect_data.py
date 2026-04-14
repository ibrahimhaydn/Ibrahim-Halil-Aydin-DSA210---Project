#!/usr/bin/env python3
"""
collect_data.py
Clash Royale Match Outcome Prediction — Data Collection Script
DSA210 Term Project

Usage:
    export CR_API_KEY="your_api_key_here"
    python collect_data.py

Output:
    data/battles_raw.csv   — raw battle records (add to .gitignore)
    data/features.csv      — featurized version (safe to upload)
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────
API_KEY          = os.getenv("CR_API_KEY", "")
BASE_URL         = "https://api.clashroyale.com/v1"
PERSONAL_TAG     = "#2UC90QQY"
MAX_CLAN_MEMBERS = 50
MAX_BATTLES      = 25        # battles fetched per player (API max is 25)
RATE_LIMIT_SLEEP = 0.15      # seconds between requests

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}


# ── Helpers ────────────────────────────────────────────────────────────────
def enc(tag: str) -> str:
    return tag.replace("#", "%23")


def api_get(path: str):
    try:
        r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        print(f"  [WARN] {r.status_code} for {path}")
    except requests.RequestException as e:
        print(f"  [ERROR] {e}")
    return None


# ── Data Fetching ──────────────────────────────────────────────────────────
def get_profile(tag: str) -> dict:
    data = api_get(f"/players/{enc(tag)}")
    return data if isinstance(data, dict) else {}


def get_battlelog(tag: str) -> list:
    data = api_get(f"/players/{enc(tag)}/battlelog")
    return data if isinstance(data, list) else []


def get_clan_member_tags(player_tag: str) -> list:
    profile = get_profile(player_tag)
    clan_tag = profile.get("clan", {}).get("tag", "")
    if not clan_tag:
        print("  [WARN] Player has no clan.")
        return []
    data = api_get(f"/clans/{enc(clan_tag)}/members")
    if not data:
        return []
    members = data.get("items", [])
    tags = [m["tag"] for m in members if m["tag"] != player_tag]
    return tags[:MAX_CLAN_MEMBERS]


def get_all_card_ids() -> list:
    data = api_get("/cards")
    if data:
        return [c["id"] for c in data.get("items", [])]
    return []


# ── Battle Parsing ─────────────────────────────────────────────────────────
def parse_deck(cards: list) -> dict:
    if not cards or len(cards) < 8:
        return {"avg_elixir": np.nan, "avg_level": np.nan, "card_ids": [],
                "underleveled_penalty": np.nan}
    levels     = [c.get("level", 1)       for c in cards]
    max_levels = [c.get("maxLevel", 14)   for c in cards]  # maxLevel from API
    penalty    = float(np.mean([m - l for m, l in zip(max_levels, levels)]))
    return {
        "avg_elixir":          float(np.mean([c.get("elixirCost", 0) for c in cards])),
        "avg_level":           float(np.mean(levels)),
        "card_ids":            [c.get("id", 0) for c in cards],
        "underleveled_penalty": penalty,   # avg levels-below-max (lower = closer to max)
    }


def parse_battle(battle: dict, player_tag: str, exp_level, battle_count,
                 player_win_rate=np.nan):
    # Accept all 1v1 battle types (PvP, pathOfLegend, tournament, etc.)
    # Exclude 2v2 by checking team/opponent list length
    btype = battle.get("type", "")
    team_list = battle.get("team", [])
    opp_list  = battle.get("opponent", [])
    if len(team_list) != 1 or len(opp_list) != 1:
        return None  # skip 2v2
    if btype in ("clanMate", "friendly", "clanWarCollectionDay"):
        return None  # skip non-competitive

    team     = (battle.get("team", []) or [{}])[0]
    opponent = (battle.get("opponent", []) or [{}])[0]

    t_crowns = team.get("crowns", 0)
    o_crowns = opponent.get("crowns", 0)
    if t_crowns == o_crowns:
        return None  # skip draws

    outcome = 1 if t_crowns > o_crowns else 0
    t_deck  = parse_deck(team.get("cards", []))
    o_deck  = parse_deck(opponent.get("cards", []))

    bt = battle.get("battleTime", "")
    hour_of_day = np.nan
    if bt:
        try:
            dt = datetime.strptime(bt, "%Y%m%dT%H%M%S.%fZ")
            hour_of_day = dt.hour
        except ValueError:
            pass

    t_trophies = team.get("startingTrophies") or np.nan
    o_trophies = opponent.get("startingTrophies") or np.nan

    t_diff = (t_trophies - o_trophies) if not (np.isnan(t_trophies) or np.isnan(o_trophies)) else np.nan
    t_pen  = t_deck["underleveled_penalty"]
    o_pen  = o_deck["underleveled_penalty"]

    return {
        "outcome":                   outcome,
        "team_trophies":             t_trophies,
        "opp_trophies":              o_trophies,
        "trophy_diff":               t_diff,
        "team_avg_elixir":           t_deck["avg_elixir"],
        "opp_avg_elixir":            o_deck["avg_elixir"],
        "elixir_diff":               t_deck["avg_elixir"] - o_deck["avg_elixir"],
        "team_avg_level":            t_deck["avg_level"],
        "opp_avg_level":             o_deck["avg_level"],
        "level_diff":                t_deck["avg_level"] - o_deck["avg_level"],
        # underleveled_penalty: avg levels below max — lower means more upgraded deck
        "team_underleveled_penalty": t_pen,
        "opp_underleveled_penalty":  o_pen,
        "underleveled_diff":         (t_pen - o_pen) if not (np.isnan(t_pen) or np.isnan(o_pen)) else np.nan,
        "team_card_ids":             json.dumps(t_deck["card_ids"]),
        "opp_card_ids":              json.dumps(o_deck["card_ids"]),
        "exp_level":                 exp_level,
        "battle_count":              battle_count,
        "player_win_rate":           player_win_rate,   # wins/(wins+losses) from profile
        "hour_of_day":               hour_of_day,
        "battle_time":               bt,
        "player_tag":                player_tag,
        "team_crowns":               t_crowns,
        "opp_crowns":                o_crowns,
    }


# ── Deck Meta Score ────────────────────────────────────────────────────────
def fetch_card_win_rates() -> dict:
    royale_key = os.getenv("ROYALE_API_KEY", "")
    if royale_key:
        try:
            r = requests.get(
                "https://api.royaleapi.com/cards/stats",
                headers={"auth": royale_key},
                timeout=10,
            )
            if r.status_code == 200:
                stats = r.json()
                return {str(c["id"]): float(c.get("winRate", 50)) / 100
                        for c in stats if "id" in c}
        except Exception as e:
            print(f"  [WARN] RoyaleAPI unavailable: {e}")

    print("  [INFO] Using uniform card win rates (set ROYALE_API_KEY for real stats).")
    all_ids = get_all_card_ids()
    return {str(cid): 0.5 for cid in all_ids}


def add_deck_meta_score(df: pd.DataFrame, win_rates: dict) -> pd.DataFrame:
    def score(card_ids_json: str) -> float:
        try:
            ids   = json.loads(card_ids_json)
            rates = [win_rates.get(str(cid), 0.5) for cid in ids]
            return float(np.mean(rates))
        except Exception:
            return 0.5

    df["team_deck_meta_score"] = df["team_card_ids"].apply(score)
    df["opp_deck_meta_score"]  = df["opp_card_ids"].apply(score)
    df["deck_meta_score_diff"] = df["team_deck_meta_score"] - df["opp_deck_meta_score"]
    return df


# ── Trophy Category ────────────────────────────────────────────────────────
# Threshold ±10: matchmaking keeps diffs very small (IQR = -9 to +8),
# so ±100 put almost everyone in "equal". ±10 gives meaningful groups.
def trophy_diff_category(diff) -> str:
    if pd.isna(diff):  return "unknown"
    if diff > 10:      return "higher"
    if diff < -10:     return "lower"
    return "equal"


# ── Featurize ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "outcome",
    "trophy_diff", "elixir_diff", "level_diff", "deck_meta_score_diff",
    "team_avg_elixir", "opp_avg_elixir",
    "team_avg_level", "opp_avg_level",
    "team_deck_meta_score", "opp_deck_meta_score",
    "team_underleveled_penalty", "opp_underleveled_penalty", "underleveled_diff",
    "exp_level", "battle_count", "player_win_rate",
    "hour_of_day", "trophy_diff_category",
]

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trophy_diff_category"] = df["trophy_diff"].apply(trophy_diff_category)
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────
def player_win_rate(profile: dict) -> float:
    """wins / (wins + losses) from player profile; nan if unavailable."""
    w = profile.get("wins",   0)
    l = profile.get("losses", 0)
    total = w + l
    return w / total if total > 0 else np.nan


def collect() -> pd.DataFrame:
    rows = []

    print(f"\n[1/2] Collecting battles for {PERSONAL_TAG} ...")
    profile      = get_profile(PERSONAL_TAG)
    exp_level    = profile.get("expLevel", np.nan)
    battle_count = profile.get("battleCount", np.nan)
    wr           = player_win_rate(profile)
    battles      = get_battlelog(PERSONAL_TAG)

    for b in battles[:MAX_BATTLES]:
        row = parse_battle(b, PERSONAL_TAG, exp_level, battle_count, wr)
        if row:
            rows.append(row)
    print(f"  → {len(rows)} ladder battles collected")

    print(f"\n[2/2] Fetching clan members ...")
    member_tags = get_clan_member_tags(PERSONAL_TAG)
    print(f"  → {len(member_tags)} members found")

    for i, tag in enumerate(member_tags, 1):
        time.sleep(RATE_LIMIT_SLEEP)
        prof = get_profile(tag)
        exp  = prof.get("expLevel", np.nan)
        bc   = prof.get("battleCount", np.nan)
        wr   = player_win_rate(prof)

        time.sleep(RATE_LIMIT_SLEEP)
        battles = get_battlelog(tag)

        for b in battles[:MAX_BATTLES]:
            row = parse_battle(b, tag, exp, bc, wr)
            if row:
                rows.append(row)

        if i % 10 == 0 or i == len(member_tags):
            print(f"  → {i}/{len(member_tags)} members | total: {len(rows)} battles")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: CR_API_KEY environment variable not set.")
        print("       export CR_API_KEY='your_api_key_here'")
        sys.exit(1)

    print("=" * 55)
    print("  Clash Royale Data Collection — DSA210")
    print("=" * 55)

    df_raw = collect()

    if df_raw.empty:
        print("\n[ERROR] No battles collected. Check your API key and network.")
        sys.exit(1)

    print("\nComputing deck meta scores ...")
    win_rates = fetch_card_win_rates()
    df_raw    = add_deck_meta_score(df_raw, win_rates)

    os.makedirs("data", exist_ok=True)
    df_raw.to_csv("data/battles_raw.csv", index=False)
    print(f"\n✓ Raw data  → data/battles_raw.csv  ({len(df_raw)} rows)")

    df_feat = featurize(df_raw)
    df_feat.to_csv("data/features.csv", index=False)
    print(f"✓ Features  → data/features.csv    ({len(df_feat)} rows, {len(df_feat.columns)} cols)")

    print(f"\n── Summary ──────────────────────────────────")
    print(f"Total battles  : {len(df_raw)}")
    print(f"Win rate       : {df_raw['outcome'].mean():.2%}")
    print(f"Unique players : {df_raw['player_tag'].nunique()}")
