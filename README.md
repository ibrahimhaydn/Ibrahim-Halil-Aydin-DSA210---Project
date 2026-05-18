# Predicting Clash Royale Match Outcomes

**DSA 210, Introduction to Data Science | 2025-2026 Spring Term**
**Ibrahim Halil Aydin**

---

## Project Overview

This project investigates whether the outcome of a Clash Royale ladder match (win / loss) can be predicted using only pre-match observable features. Personal battle data from the official Clash Royale API is combined with enriched card and player statistics to build a binary classification model. The central question is: **which pre-match factors actually determine who wins, deck strength, card levels, trophy ranking, or player experience?**

## Hypothesis

**H₀:** Pre-match features (deck meta score, card levels, trophy difference, player experience) have no significant relationship with match outcome.

**H₁:** At least some pre-match features, particularly deck meta score and player win rate, are statistically significant predictors of match outcome.

## Data Sources

| Source | Variables | Method |
|--------|-----------|--------|
| [Clash Royale API](https://developer.clashroyale.com) | Battle outcome, crowns, trophies, 8-card deck (ID, level, elixir cost) for both players, expLevel, battleCount, wins, losses | Python script via `/players/{tag}/battlelog` and `/players/{tag}` endpoints |
| [RoyaleAPI](https://royaleapi.com/cards/popular) | Card win rate, card usage rate | Manual extraction via browser JavaScript |
| Empirical (computed) | Per-card win rate from collected battles | `compute_meta.py`, counts wins per card across all matches |
| Derived | Player win rate = wins / (wins + losses) | Computed from `/players` profile endpoint |

## Dataset

- **979** competitive 1v1 ladder battle records
- Collected from personal account **#2UC90QQY** and up to 50 clan members
- 2v2, friendly, and draw matches excluded
- **Class balance:** 53.3% Win / 46.7% Loss

## Engineered Features

| Feature | Description |
|---------|-------------|
| `trophy_diff` | Team starting trophies − Opponent starting trophies |
| `elixir_diff` | Team avg elixir cost − Opponent avg elixir cost |
| `level_diff` | Team avg card level − Opponent avg card level |
| `underleveled_diff` | Team avg (maxLevel − level) − Opponent avg (maxLevel − level) |
| `deck_meta_score_diff` | Team deck meta score − Opponent deck meta score |
| `team/opp_deck_meta_score` | Mean win rate of cards in each deck (from RoyaleAPI + empirical) |
| `player_win_rate` | Player's historical win rate from profile (wins / total battles) |
| `exp_level` | Player King Tower level (overall account progression) |
| `battle_count` | Total battles played (proxy for experience) |
| `hour_of_day` | UTC hour of the battle |
| `trophy_diff_category` | higher / equal / lower (±10 trophy threshold) |

> `data/battles_raw.csv` contains player tags and is excluded via `.gitignore`. Only `data/features.csv` (anonymized, featurized) is uploaded.

## Repository Structure

```
Ibrahim-Halil-Aydin-DSA210---Project/
├── scripts/
│   ├── collect_data.py          # Step 1, fetch data from Clash Royale API
│   ├── scrape_royaleapi.py      # Step 2, enrich with RoyaleAPI card stats
│   ├── compute_meta.py          # Step 3, compute empirical card win rates
│   ├── add_new_features.py      # Step 4, add underleveled_penalty, player_win_rate
│   ├── analysis.py              # Step 5, EDA plots + hypothesis tests
│   └── ml_models.py             # Step 6, ML models (Logistic Regression, KNN,
│                                #          Decision Tree, Random Forest,
│                                #          Gradient Boosting, Voting Classifier)
├── notebooks/
│   ├── milestone1_eda_and_tests.ipynb  # Executed notebook with detailed verbal
│   │                                   # interpretations for every EDA figure
│   │                                   # and hypothesis test (assumptions +
│   │                                   # connection to the research question)
│   └── milestone2_ml_models.ipynb      # Executed notebook walking through every
│                                       # ML step (split, scaling, CV, each
│                                       # model, hyperparameter tuning, metrics,
│                                       # coefficient interpretation) with
│                                       # lecture references and verbal analysis
├── data/
│   ├── features.csv              # Featurized dataset (979 rows, 19 features)
│   └── card_stats_royaleapi.csv  # Community card statistics
│   (battles_raw.csv, gitignored, contains player tags)
│
├── plots/
│   ├── 01_outcome_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_winrate_by_trophy_bucket.png
│   ├── 04_winrate_by_hour.png
│   ├── 05_winrate_by_explevel.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_boxplots_by_outcome.png
│   ├── 08_confusion_matrices.png   # ML, confusion matrix per model
│   ├── 09_roc_curves.png           # ML, ROC curves overlaid
│   ├── 10_pr_curves.png            # ML, Precision-Recall curves
│   ├── 11_feature_importance.png   # ML, Random Forest variable importance
│   └── 12_model_comparison.png     # ML, all metrics, all models
│
├── requirements.txt
├── README.md
├── .gitignore
└── Proposal_Ibrahim_Halil_Aydin.pdf
```

## Key Findings (EDA + Hypothesis Testing)

### EDA Highlights
- Win rate is 53.3%, close to the expected ~50% due to symmetric ladder matchmaking
- Players with higher experience levels (expLevel) show consistently higher win rates
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

**Key insight:** Deck meta score and player historical win rate are the strongest pre-match predictors. Trophy difference is not predictive, Clash Royale's matchmaking system effectively neutralizes trophy advantage by pairing players within ±9 trophies.

## Machine Learning Results (Milestone 2)

The full walk-through with lecture references, verbal interpretations, and assumptions is in [`notebooks/milestone2_ml_models.ipynb`](notebooks/milestone2_ml_models.ipynb). This section gives the executive summary.

### Methodology, strictly course-aligned

The pipeline uses *only* techniques presented in DSA 210 lectures (Weeks 7-10). No method outside that surface is used.

| Lecture | Concepts applied in this project |
|---|---|
| **Week 7**, Regression & Cross-validation | 80/20 stratified train/test split, 5-fold stratified cross-validation |
| **Week 8**, KNN & Tree Models | `StandardScaler` (slides 6-7), `KNeighborsClassifier` with k-tuning over {1, 3, 5, 7, 11, 15, 21, 31} (slide 9 *"Best K for Loan Data"*), `DecisionTreeClassifier` with `GridSearchCV` over `max_depth ∈ {5, 10, 15, 20, 25, 30, None}`, `min_samples_split ∈ {2, 20, 50, 100}`, `min_samples_leaf ∈ {1, 5, 10}` (84 combinations, slide 16), `RandomForestClassifier` with feature importance (slide 22) |
| **Week 9a**, Logistic Regression | `LogisticRegression` with L2 regularization (`C = 1.0`), standardized-coefficient interpretation |
| **Week 9b**, Performance Metrics | Confusion matrix, accuracy, precision, recall, F1, ROC-AUC, ROC curves, Precision-Recall curves, classification report |
| **Week 10**, Ensemble Learning | `RandomForestClassifier`, `GradientBoostingClassifier`, `VotingClassifier` (soft voting) |

### Pipeline at a glance

1. Load the 979-row dataset, drop the redundant categorical `trophy_diff_category`, and impute missing values with the column median (319 missing in `trophy_diff`).
2. Split 80/20 stratified on outcome → 783 training rows, 196 test rows, both at 53.3% win rate.
3. Apply `StandardScaler` fit on the training set only, then `transform` the test set (no leakage).
4. Run 5-fold cross-validation on the training set for every model.
5. Tune KNN (8 values of k) and the Decision Tree (84 hyperparameter combinations) by CV.
6. Fit each finalised model on the full training set and evaluate exactly once on the held-out test set.

### Final Test-Set Results (n = 196)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|----------|-------|----|---------|
| **Logistic Regression** | **0.689** | **0.720** | 0.686 | **0.702** | **0.736** |
| Gradient Boosting | 0.658 | 0.673 | 0.705 | 0.688 | 0.711 |
| Voting (soft) | 0.653 | 0.664 | 0.714 | 0.688 | 0.711 |
| Decision Tree (tuned) | 0.648 | 0.658 | 0.714 | 0.685 | 0.709 |
| KNN (k = 31) | 0.653 | 0.664 | 0.714 | 0.688 | 0.696 |
| Random Forest | 0.628 | 0.648 | 0.667 | 0.657 | 0.668 |

**Logistic Regression wins on every metric except recall**, accuracy 0.689, F1 0.702, and ROC-AUC 0.736 all top the leaderboard. The win is consistent across thresholds: in the ROC plot, the LR curve is above every other model's at almost every operating point.

### Why the simplest model wins

Two factors explain why a 17-parameter linear model beats 200-tree ensembles on this dataset:

1. **The signal is approximately linear.** Milestone 1 showed that the strongest features relate to outcome monotonically (Pearson *r*, Welch's *t*, none requiring an interaction term). Logistic Regression encodes exactly that kind of relationship; tree ensembles cannot extract substantially more.
2. **The dataset is small.** With 783 training rows, every model faces a bias/variance trade-off that favours the simpler hypothesis class. Random Forest and Gradient Boosting have the *capacity* to overfit, and 5-fold CV showed them doing exactly that, their CV variance was higher than Logistic Regression's.

The lesson lines up with the lecture material's principle that model complexity should match data complexity. Logistic Regression's coefficients are the right size, and adding flexibility costs more than it adds.

### Most Influential Features (Logistic Regression, standardized coefficients)

Because features were standardized, magnitudes are directly comparable. The odds ratio = exp(coefficient) tells you how a one-standard-deviation change in the feature multiplies the odds of winning.

| Feature | Coefficient | Odds ratio | Interpretation |
|---|---|---|---|
| `player_win_rate` | **+0.645** | 1.91 | A 1-SD increase in historical win rate nearly doubles the odds of winning. Strongest skill signal. |
| `opp_deck_meta_score` | −0.388 | 0.68 | A 1-SD stronger opponent deck cuts win-odds to ~68%. |
| `deck_meta_score_diff` | +0.231 | 1.26 | A 1-SD stronger personal deck (vs. opponent) raises win-odds ~26%. |
| `exp_level` | −0.131 | 0.88 | Negative because of correlation with `player_win_rate` (Week 7 *Correlated predictors*). |
| `trophy_diff` | +0.126 | 1.13 | Small coefficient × small variation = negligible total effect. |

The remaining 12 coefficients are all below 0.1 in magnitude, present in the model but contributing little to predictions.

### Convergence with Milestone 1 hypothesis tests

Every Milestone 2 finding lines up with a Milestone 1 hypothesis-test result:

| Milestone 1 finding | Milestone 2 confirmation |
|---|---|
| `deck_meta_score_diff` is the strongest single predictor (t-test, *p* < 10⁻³). | Top-2 in Random Forest importance; LR coefficient +0.231. |
| `player_win_rate` correlates strongest with outcome (Pearson *r* = 0.285). | Top-1 in Random Forest importance; LR coefficient +0.645 (largest). |
| `trophy_diff_category` independent of outcome (χ², *p* ≈ 0.54). | LR coefficient +0.126 has tiny standardised effect; low importance. |
| `hour_of_day`, `elixir_diff` non-significant. | Coefficients near zero; low importance. |

The two halves of the project corroborate each other, the strongest internal-validity signal a single-dataset study can produce.

### Answer to the research question

> *Can pre-match observable features reliably predict Clash Royale match outcomes, and which factors matter most?*

**Yes, to a useful but not perfect degree.** A Logistic Regression model achieves 69% accuracy and 0.74 ROC-AUC on the held-out test set, well above the 53.3% naïve baseline but far from 100%. The deciding factors are, in order: **(1) player historical win rate**, **(2) deck meta strength differential**, and **(3) player experience**. Trophy advantage, hour of day, elixir cost, and average card-level differences carry essentially no predictive weight at the per-battle level, the matchmaker neutralises the first, and the rest are flat across outcomes.

## How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Clash Royale API key
export CR_API_KEY="your_api_key_here"

# Step 1, Collect battle data
python scripts/collect_data.py

# Step 2, Enrich with RoyaleAPI stats (requires data/card_stats_royaleapi.csv)
python scripts/scrape_royaleapi.py

# Step 3, Compute empirical card win rates
python scripts/compute_meta.py

# Step 4, Add new features (no re-collection needed)
python scripts/add_new_features.py

# Step 5, Run EDA and hypothesis tests (script form)
python scripts/analysis.py
# Outputs: plots/01-07.png + hypothesis test results to stdout

# Step 5b (alternative), Open the executed notebook with detailed interpretations
jupyter notebook notebooks/milestone1_eda_and_tests.ipynb

# Step 6, Train and evaluate ML models (script form)
python scripts/ml_models.py
# Outputs: plots/08-12.png + model metrics to stdout

# Step 6b (alternative), Open the executed ML notebook with full walk-through
jupyter notebook notebooks/milestone2_ml_models.ipynb
```

## AI Usage Disclosure

Per the DSA 210 academic integrity policy, this section documents the specific ways an AI assistant (Anthropic Claude) was used in this project. Every output described below was reviewed, edited, and tested by the student before being committed. Statistical methods and analytical decisions follow DSA 210 lecture content.

**1. Data collection script (`scripts/collect_data.py`)**
Prompt: "Write a Python script that paginates through the /players/{tag}/battlelog endpoint of the Clash Royale API, respects the 30 calls per minute rate limit, and saves each battle as a row with columns for team and opponent decks, trophies, crowns, and outcome."
Output: a `requests`-based collector with retry logic and a `time.sleep` rate limiter. I customised the player tag, added clan-member enumeration, and added the win/loss/draw filter that excludes 2v2 and friendly battles.

**2. Card statistics scraper (`scripts/scrape_royaleapi.py`)**
Prompt: "Help me extract per-card win rate and usage rate from royaleapi.com/cards/popular into a CSV. The page is rendered client-side, so suggest the simplest approach."
Output: a browser JavaScript snippet that reads the rendered DOM and downloads a CSV, plus a Python loader that joins it against my collected dataset on card ID. I had to debug a dtype mismatch (card ID was integer in one table and string in the other), which Claude helped me identify after I described the symptom.

**3. Feature engineering (`scripts/add_new_features.py`, `scripts/compute_meta.py`)**
Prompt: "Write a feature that computes mean (maxLevel minus cardLevel) per deck without re-collecting the raw data. Use the existing battles_raw.csv."
Output: a vectorised pandas implementation. I extended it with `player_win_rate` derived from the `/players` profile endpoint after Claude suggested the wins / (wins + losses) formulation.

**4. EDA and hypothesis tests (`scripts/analysis.py`, `notebooks/milestone1_eda_and_tests.ipynb`)**
Prompt: "Write a Welch's t-test for deck_meta_score_diff grouped by outcome, with an assumption check (Shapiro-Wilk for normality, Levene for equal variance) printed beforehand."
Output: a scipy.stats based snippet that I reused for every hypothesis test in Milestone 1. The verbal interpretations in the notebook were drafted with Claude as a sounding board (I described what I saw in each figure, Claude rephrased it into report-style English), then I edited each paragraph for accuracy.

**5. Machine learning pipeline (`scripts/ml_models.py`, `notebooks/milestone2_ml_models.ipynb`)**
Prompt: "Set up a stratified 5-fold cross-validation loop for sklearn that reports per-fold accuracy and the mean, then evaluates the final model once on a held-out test set."
Output: the `StratifiedKFold` + `cross_val_score` boilerplate that I adapted for all six models. I picked the hyperparameter grids myself from the Week 8 slide ranges (max_depth 5 to 30, min_samples_split 20 to 100 for the Decision Tree; eight k values for KNN).

**6. Interpretation of model results**
Prompt: "Why might Logistic Regression beat Random Forest and Gradient Boosting on a 783-row dataset where the strongest predictors are monotonically related to the outcome?"
Output: the bias-variance and signal-linearity explanation that appears in the report's "Why the Simplest Model Wins" section. I cross-checked the argument against the lecture material before using it.


No AI tool was used to generate the dataset itself, the hypothesis test p-values, or the model accuracies reported. All numbers come from running the scripts and notebooks on the collected data.
