# Walmart Capstone (Team 2): PromoPilot Targeted Promotion Pipeline

This pipeline decides which promotion each EverTrail July customer should get: one of `NO_OFFER`, `PCT_10`, `PCT_20` or `FREE_SHIP`. The goal is the most **expected incremental net contribution margin (NCM)** over 14 days, within Walmart's budget and customer-experience constraints.

> **Status: all ten steps (01–10) are implemented and run end to end, from the raw CSVs to `policy_submission.csv`.**

## Pipeline

| # | Notebook | Responsibility |
|---|---|---|
| 01 | `01_data_validation` | Validate raw files: schema, keys, missingness, duplicates, timing, financials. Save clean core tables |
| 02 | `02_build_historical_features` | 43 pre-send features per historical assignment |
| 03 | `03_construct_14_day_outcomes` | 14-day outcomes (`purchase_14d`, `ncm_14d`, promo cost, returns). Builds the canonical `historical_feature_outcome` table |
| 04 | `04_business_analysis` | KPIs and EDA, and randomized (exploration) offer effects |
| 05 | `05_logistic_propensity_model` | Required Logistic Regression purchase-propensity baseline |
| 06 | `06_tlearner_xgboost` | T-Learner: four XGBoost NCM models (one per action), trained on exploration rows |
| 07 | `07_score_july_population` | Rebuild the 43 features at `policy_as_of_ts`. Score propensity and NCM under every action |
| 08 | `08_estimate_promotion_cost` | Expected promo cost for each July customer × action, via XGBoost on exploration rows |
| 09 | `09_milp_policy_optimization` | MILP (PuLP + CBC): one action per customer, under the official constraints |
| 10 | `10_generate_outputs` | `policy_submission.csv`, the three required baselines, comparison and reporting tables and figures |

Each step reads the **saved output of the step before it**:

```text
data/raw → 01 → data/processed/clean → 02 → historical_features → 03 → historical_feature_outcome ─┬→ 04 KPIs
                                                                                                    ├→ 05 logistic model ─┐
                                                                                                    └→ 06 T-Learner models ┤
policy_scoring_population + clean tables + src/features.py ───────────────────────→ 07 July features + scores ←┘
→ 08 july_customer_action_values → 09 optimized_policy_detailed → 10 policy_submission.csv + baselines + dashboard
```

## Run it

1. Put the 11 raw CSVs in `data/raw/`. They are git-ignored, never modified, and their SHA-256 hashes are checked by Step 01.
2. Install the packages:
   ```
   pip install -r requirements.txt
   ```
3. Run all notebooks in order. This stops at the first failure; `python run_pipeline.py 06 10` runs just Steps 06–10:
   ```
   python run_pipeline.py
   ```
   Alternatively, open `notebooks/01_…` through `10_…` in Jupyter and choose **Run All** on each, in order.
4. Launch the dashboard (it only reads saved outputs):
   ```
   streamlit run dashboard/app.py
   ```

**Final submission:** `policy_submission.csv` (repository root, copied to `outputs/final/`). It has exactly four columns, in this order: `customer_id, recommended_action, estimated_incremental_ncm, estimated_promo_cost`. There is one row for each of the 2,776 scoring customers.

## Folder structure

```text
data/raw/                 supplied CSVs (not in git)
data/processed/clean/     Step 01 clean tables
data/processed/           historical_features, historical_outcomes, historical_feature_outcome, july_features (csv + parquet)
src/                      common.py (paths/loaders), features.py (feature builder, Steps 02 & 07), modeling.py (XGBoost pipeline, Steps 06 & 08)
notebooks/                01–10
outputs/models/           logistic, T-Learner (×4), promo-cost (×3) pipelines + metadata JSON
outputs/tables/           DQ, KPI, model metrics, July scores, customer-action values, policy, constraint audit, baseline comparison
outputs/figures/          decision-useful charts from Steps 04, 05, 06, 10
outputs/final/            policy_submission.csv
dashboard/app.py          Streamlit dashboard
docs/                     pipeline walkthrough (PDF + Markdown source)
```

## Key rules and assumptions

### Timing (no leakage)
- **Historical features** use events with timestamp `< send_ts`. Look-back windows are `[send_ts − N days, send_ts)`.
- **July features** are built by the same `src/features.build_features` with `cut="policy_as_of_ts", inclusive=True`, so they use events `<= policy_as_of_ts`.
- **Leakage tests:** Steps 02 and 07 check the latest event used per feature family, and inject fake future events to confirm that no feature changes.
- **Outcome window:** `send_ts <= order_ts < send_ts + 14 days`. Only same-customer, same-campaign, completed orders count.

### Data cleaning and outcomes (Steps 01–03)
- **Cleaning:** 160 exact duplicate `order_items` rows are removed. Cancelled orders are never purchases. Anonymous web sessions are excluded from features.
- **NCM:** `ncm_14d = net_revenue − cogs − shipping_cost − refund_amount_applicable`. Net revenue already nets out discounts and shipping revenue.
- **Return timing** (`RETURN_RULE` in Step 03):
  - Only refunds returned inside the same 14-day window count.
  - Refunds after the window are kept separately in `ncm_14d_all_returns` for sensitivity checks.
  - This matters: $83.8K of refunds arrive after the window, versus $15.4K inside it.
- **Promo cost:**
  - PCT offers: the discount on redeemed orders.
  - FREE_SHIP: an inferred $5.99 waived fee on redeemed orders below $75.08, flagged `promo_cost_inferred`.
  - NO_OFFER: 0.

### Logistic baseline (Step 05)
- Trained on CP01–CP05, `C` tuned on CP06, tested on CP07, then refit on CP01–CP07.
- It predicts **who will buy**, not who an offer changes. It is used only for required baseline #2.

### T-Learner (Step 06)
- **Models:** one `XGBRegressor` pipeline per action learns `E[ncm_14d | X, action]`.
- **Incremental NCM:** for each action, the predicted NCM under that action minus the predicted NCM under NO_OFFER.
- **Training data:** only randomized `exploration` rows. The `business_rule` rows were targeted, so they are confounded.
- **Tuning:** one shared configuration is chosen on CP06 using pooled factual RMSE, tested once on CP07, then refit on all CP01–CP07 exploration rows.
- **Grid adjustment:** the small grid was extended toward stronger regularization, because CP06 favoured the smallest models.
- **Validation:** the final models' average effects reproduce the randomized Step 04 estimates.
- **Limitation:** individual counterfactuals are never observed, so only factual accuracy and a held-out policy value (normalized IPW on CP07) are reported.

### Expected promotion cost (Step 08)
- **Models:** one XGBoost per commercial action on exploration rows learns `E[promo_cost_14d | X, action]`. NO_OFFER = 0.
- **Clipping:** final costs are `max(prediction, 0)`.

### Optimization (Step 09)
- **Objective:** maximize `Σ x[i,a] · estimated_incremental_ncm[i,a]`.
- **Constraints:**
  - exactly one action per customer;
  - expected promo spend ≤ **$62,000**;
  - offers ≤ **floor(60% · N)**;
  - `PCT_20` ≤ **15% of offered** customers;
  - **measurement holdout:** `ceil(5% · N)` customers chosen at random (`seed 42`) *before* optimizing and forced to NO_OFFER.
- **Solver:** CBC through PuLP. The run must return `Optimal`.
- **Windows-on-ARM note:** PuLP 2.9 bundles CBC, but on Windows-on-ARM with x64 Python it looks for a missing arm64 binary, so Step 09 points to the bundled x64 `cbc.exe`.

### Required baselines (Step 10)
All baselines are scored with the same July predictions and costs. Each keeps the holdout on NO_OFFER and respects the caps.
1. **ALL_NO_OFFER.**
2. **PROPENSITY_PCT10:** customers ranked by Logistic propensity get PCT_10 until the budget or the 60% cap is reached.
3. **HISTORICAL_MIX:** the `business_rule` commercial mix (PCT_10 59.8% / PCT_20 13.1% / FREE_SHIP 27.1%) is applied to a seeded random 60% of non-holdout customers. The historical offer rate of 74.8% exceeds the cap, so the number of offers is scaled down to the cap.

## Required packages
- **Phase 1:** pandas, numpy, scikit-learn, matplotlib, joblib, pyarrow, jupyter / nbconvert / nbclient.
- **Phase 2:** xgboost, pulp (pinned to 2.9.x because it bundles the CBC binary), streamlit.

The pipeline was tested on Python 3.14 (Anaconda). See `requirements.txt`.
