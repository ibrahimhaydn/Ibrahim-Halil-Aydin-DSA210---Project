#!/usr/bin/env python3
"""
compute_meta.py
Clash Royale — Empirical Card Win Rate & Deck Meta Score
DSA210 Term Project

Reads battles_raw.csv, computes per-card win rates from observed data,
then rewrites features.csv with meaningful deck_meta_score columns.

Usage:
    python compute_meta.py
"""

import json
import os
import numpy as np
import pandas as pd

RAW_PATH  = "data/battles_raw.csv"
FEAT_PATH = "data/features.csv"
MIN_APPEARANCES = 5   # minimum times a card must appear to get a real rate


# ── Load raw data ──────────────────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run collect_data.py first."
        )
    return pd.read_csv(path)


# ── Compute per-card win rates ─────────────────────────────────────────────
def compute_card_win_rates(df: pd.DataFrame) -> dict:
    """
    For each card ID, count how many times it appeared in a winning deck
    vs. total appearances across all matches (both sides).

    Perspective: we treat each card appearance independently.
    A card in the team deck contributes outcome=1 if team won.
    A card in the opp deck contributes outcome=0 if team won (opp lost).
    """
    card_wins   = {}   # card_id -> total wins
    card_total  = {}   # card_id -> total appearances

    for _, row in df.iterrows():
        outcome = row["outcome"]   # 1 = team won, 0 = opp won

        # Team deck cards → win if outcome==1
        try:
            team_ids = json.loads(row["team_card_ids"])
            for cid in team_ids:
                cid = str(cid)
                card_wins[cid]  = card_wins.get(cid, 0)  + (1 if outcome == 1 else 0)
                card_total[cid] = card_total.get(cid, 0) + 1
        except Exception:
            pass

        # Opponent deck cards → win if outcome==0 (opp won)
        try:
            opp_ids = json.loads(row["opp_card_ids"])
            for cid in opp_ids:
                cid = str(cid)
                card_wins[cid]  = card_wins.get(cid, 0)  + (1 if outcome == 0 else 0)
                card_total[cid] = card_total.get(cid, 0) + 1
        except Exception:
            pass

    # Build win rate dict; fall back to 0.5 for rarely-seen cards
    win_rates = {}
    for cid in card_total:
        if card_total[cid] >= MIN_APPEARANCES:
            win_rates[cid] = card_wins[cid] / card_total[cid]
        else:
            win_rates[cid] = 0.5   # not enough data

    return win_rates


# ── Print top/bottom cards ─────────────────────────────────────────────────
def print_card_summary(win_rates: dict, card_total: dict, top_n: int = 10):
    df_cards = pd.DataFrame({
        "card_id":   list(win_rates.keys()),
        "win_rate":  list(win_rates.values()),
        "n":         [card_total.get(c, 0) for c in win_rates],
    }).sort_values("win_rate", ascending=False)

    df_cards = df_cards[df_cards["n"] >= MIN_APPEARANCES]

    print(f"\n  Top {top_n} highest win-rate cards:")
    print(df_cards.head(top_n).to_string(index=False))
    print(f"\n  Bottom {top_n} lowest win-rate cards:")
    print(df_cards.tail(top_n).to_string(index=False))
    print(f"\n  Total unique cards tracked: {len(df_cards)}")
    print(f"  Win rate range: {df_cards['win_rate'].min():.3f} — {df_cards['win_rate'].max():.3f}")
    print(f"  Win rate std:   {df_cards['win_rate'].std():.4f}")


# ── Recompute deck meta scores ─────────────────────────────────────────────
def deck_score(card_ids_json: str, win_rates: dict) -> float:
    try:
        ids   = json.loads(card_ids_json)
        rates = [win_rates.get(str(cid), 0.5) for cid in ids]
        return float(np.mean(rates))
    except Exception:
        return 0.5


def update_features(df_raw: pd.DataFrame, win_rates: dict) -> pd.DataFrame:
    """Load features.csv and overwrite deck meta score columns."""
    if not os.path.exists(FEAT_PATH):
        raise FileNotFoundError(f"'{FEAT_PATH}' not found.")

    df_feat = pd.read_csv(FEAT_PATH)

    # Recompute from raw data (which has card_ids)
    df_raw = df_raw.copy()
    df_raw["team_deck_meta_score"] = df_raw["team_card_ids"].apply(
        lambda x: deck_score(x, win_rates))
    df_raw["opp_deck_meta_score"]  = df_raw["opp_card_ids"].apply(
        lambda x: deck_score(x, win_rates))
    df_raw["deck_meta_score_diff"] = (
        df_raw["team_deck_meta_score"] - df_raw["opp_deck_meta_score"])

    # Overwrite the three meta columns in features.csv
    for col in ["team_deck_meta_score", "opp_deck_meta_score", "deck_meta_score_diff"]:
        if col in df_feat.columns:
            df_feat[col] = df_raw[col].values

    return df_feat


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Computing Empirical Card Win Rates — DSA210")
    print("=" * 55)

    print(f"\nLoading {RAW_PATH} ...")
    df_raw = load_raw(RAW_PATH)
    print(f"  → {len(df_raw)} battles loaded")

    print("\nComputing per-card win rates ...")
    # Need card_total for the summary
    card_wins_tmp  = {}
    card_total_tmp = {}
    for _, row in df_raw.iterrows():
        outcome = row["outcome"]
        for side, won in [("team_card_ids", outcome == 1),
                          ("opp_card_ids",  outcome == 0)]:
            try:
                for cid in json.loads(row[side]):
                    cid = str(cid)
                    card_wins_tmp[cid]  = card_wins_tmp.get(cid, 0)  + (1 if won else 0)
                    card_total_tmp[cid] = card_total_tmp.get(cid, 0) + 1
            except Exception:
                pass

    win_rates = {}
    for cid in card_total_tmp:
        if card_total_tmp[cid] >= MIN_APPEARANCES:
            win_rates[cid] = card_wins_tmp[cid] / card_total_tmp[cid]
        else:
            win_rates[cid] = 0.5

    print_card_summary(win_rates, card_total_tmp)

    print(f"\nUpdating {FEAT_PATH} with empirical meta scores ...")
    df_feat = update_features(df_raw, win_rates)
    df_feat.to_csv(FEAT_PATH, index=False)

    # Show new variance
    diff = df_feat["deck_meta_score_diff"]
    print(f"\n  deck_meta_score_diff stats:")
    print(f"    mean = {diff.mean():.5f}")
    print(f"    std  = {diff.std():.5f}")
    print(f"    min  = {diff.min():.5f}")
    print(f"    max  = {diff.max():.5f}")

    if diff.std() < 0.001:
        print("\n  ⚠️  Std is still near zero — dataset may be too small")
        print("     for reliable per-card rates. Consider collecting more data.")
    else:
        print("\n  ✓ Deck meta scores now have real variance — re-run analysis.py")

    print(f"\n✓ {FEAT_PATH} updated.")
    print("  Next step: python analysis.py")
