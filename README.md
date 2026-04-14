# Predicting Clash Royale Match Outcomes

**DSA 210 – Introduction to Data Science | 2025–2026 Spring Term**
**Ibrahim Halil Aydin**

---

## Project Overview

This project investigates whether the outcome of a Clash Royale ladder match (win / loss) can be predicted using only pre-match observable features. Personal battle data from the official Clash Royale API is combined with enriched card and player statistics to build a binary classification model. The central question is: **which pre-match factors actually determine who wins — deck strength, card levels, trophy ranking, or player experience?**

## Hypothesis

**H₀:** Pre-match features (deck meta score, card levels, trophy difference, player experience) have no significant relationship with match outcome.

**H₁:** At least some pre-match features — particularly deck meta score and player win rate — are statistically significant predictors of match outcome.

## Data Sources

| Source | Variables | Method |
|--------|-----------|--------|
| [Clash Royale API](https://developer.clashroyale.com) | Battle outcome, crowns, trophies, 8-card deck for both players, expLevel, battleCount | Python script via /players/{tag}/battlelog |
| [RoyaleAPI](https://royaleapi.com/cards/popular) | Card win rate, card usage rate | Manual extraction via browser JavaScript |
| Empirical (computed) | Per-card win rate from collected battles | compute_meta.py |
| Derived | Player win rate = wins / (wins + losses) | From /players profile endpoint |

## Dataset

- **979** competitive 1v1 ladder battle records
- Collected from personal account **#2UC90QQY** and up to 50 clan members
- 2v2, friendly, and draw matches excluded
- **Class balance:** 53.3% Win / 46.7% Loss

## Engineered Features

| Feature | Description |
|---------|-------------|
| trophy_diff | Team starting trophies − Opponent starting trophies |
| elixir_diff | Team avg elixir cost − Opponent avg elixir cost |
| level_diff | Team avg card level − Opponent avg card level |
| underleveled_diff | Team avg (maxLevel − level) − Opponent avg (maxLevel − level) |
| deck_meta_score_diff | Team deck meta score − Opponent deck meta score |
| player_win_rate | Player's historical win rate (wins / total battles) |
| exp_level | Player King Tower level |
| battle_count | Total battles played |
| hour_of_day | UTC hour of the battle |
| trophy_diff_category | higher / equal / lower (±10 trophy threshold) |

> data/battles_raw.csv contains player tags and is excluded via .gitignore. Only data/features.csv (anonymized) is uploaded.

## Repository Structure

```
Ibrahim-Halil-Aydin-DSA210---Project/
├── collect_data.py          # Step 1 — fetch data from Clash Royale API
├── scrape_royaleapi.py      # Step 2 — enrich with RoyaleAPI card stats
├── compute_meta.py          # Step 3 — compute empirical card win rates
├── add_new_features.py      # Step 4 — add new features to existing dataset
├── analysis.py              # Step 5 — EDA plots + hypothesis tests
├── requirements.txt
├── README.md
├── .gitignore
├── Proposal_Ibrahim_Halil_Aydin.pdf
├── data/
│   ├── features.csv              # Featurized dataset (979 rows, 19 features)
│   └── card_stats_royaleapi.csv  # Community card statistics
└── plots/
    ├── 01_outcome_distribution.png
    ├── 02_feature_distributions.png
    ├── 03_winrate_by_trophy_bucket.png
    ├── 04_winrate_by_hour.png
    ├── 05_winrate_by_explevel.png
    ├── 06_correlation_heatmap.png
    └── 07_boxplots_by_outcome.png
```

## Key Findings (EDA + Hypothesis Testing)

### EDA Highlights
- Win rate is 53.3%, close to expected ~50% due to symmetric ladder matchmaking
- Players with higher experience levels show consistently higher win rates
- Trophy differences are extremely small (IQR: −9 to +8), confirming effective matchmaking
- Deck meta score distributions visually differ between wins and losses

### Hypothesis Test Results

| Test | Feature | Result | p-value |
|------|---------|--------|---------|
| Two-sample t-test | Deck meta score diff | ✅ SIGNIFICANT | p < 0.001 (t = 9.95) |
| Chi-square | Trophy diff category | ✗ Not significant | p = 0.541 |
| Pearson correlation | Battle count | ✅ SIGNIFICANT | p = 0.036 (r = 0.067) |
| Pearson correlation | Exp level | ✅ SIGNIFICANT | p < 0.001 (r = 0.127) |
| Pearson correlation | Player win rate | ✅ SIGNIFICANT | p < 0.001 (r = 0.285) |

**Key insight:** Deck meta score and player historical win rate are the strongest pre-match predictors. Trophy difference is not predictive — Clash Royale's matchmaking effectively neutralizes trophy advantage by pairing players within ±9 trophies.

## How to Reproduce

```bash
pip install -r requirements.txt
export CR_API_KEY="your_api_key_here"
python collect_data.py
python scrape_royaleapi.py
python compute_meta.py
python add_new_features.py
python analysis.py
```

## AI Usage Disclosure

AI tools were used during the coding, debugging, and feature engineering phases of this project. All code has been reviewed and tested by the student. Statistical methods align with DSA210 course content.
