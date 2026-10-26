# Walmart Capstone (Team 2): PromoPilot Modular Pipeline

This pipeline decides which promotion each EverTrail July customer should get: one of `NO_OFFER`, `PCT_10`, `PCT_20` or `FREE_SHIP`. The goal is the most **incremental net contribution margin (NCM)** over 14 days, within the budget and customer-experience limits.

> **Status: Steps 01–05 are implemented and run end to end. Steps 06–10 are planned and not implemented yet.**

## Pipeline architecture

| # | Notebook | Responsibility | Status |
|---|---|---|---|
| 01 | `01_data_validation` | Ingest raw files; check schema, keys, missingness, duplicates, timing and financials; clean core tables | done |
| 02 | `02_build_historical_features` | Pre-send customer, purchase, return, web, email and product features | done |
| 03 | `03_construct_14_day_outcomes` | 14-day purchase, financial, promo-cost, return and NCM outcomes; build the canonical table | done |
| 04 | `04_business_analysis` | KPI and EDA by offer, campaign, tenure/value and behavioral segment | done |
| 05 | `05_logistic_propensity_model` | Required Logistic Regression purchase-propensity baseline | done |
| 06 | `06_tlearner_xgboost` | Four treatment-specific NCM models and incremental NCM | planned |
| 07 | `07_score_july_population` | Rebuild features at `policy_as_of_ts`; score each action | planned |
| 08 | `08_estimate_promotion_cost` | Expected promo cost per customer-action pair | planned |
| 09 | `09_milp_policy_optimization` | Constrained customer-level allocation | planned |
| 10 | `10_generate_outputs` | `policy_submission.csv` and reporting tables | planned |

Each step reads the **saved output of the step before it** instead of rebuilding that logic:

```text
data/raw/*.csv -> 01 -> data/processed/clean/*.parquet -> 02 -> historical_features -> 03 -> historical_feature_outcome -> 04 (KPIs) / 05 (model)
```

## Setup and execution

1. Put the 11 raw CSVs (`campaigns`, `customers`, `historical_assignments`, `web_sessions`, `email_events`, `orders`, `order_items`, `products`, `returns`, `offer_catalog`, `policy_scoring_population`) in `data/raw/`. The pipeline never modifies them, and Step 01 checks their SHA-256 hashes.
2. Install the packages: `pip install -r requirements.txt`.
3. Run the notebooks in order: 01 → 02 → 03 → 04 → 05. For example:
   `jupyter nbconvert --to notebook --execute --inplace notebooks/0[1-5]_*.ipynb`

Shared helpers are in `src/`:
- `common.py` holds paths, constants and loaders.
- `features.py` holds the Step 02 feature logic. Step 07 will reuse it with `cut="policy_as_of_ts", inclusive=True`.

## Outputs by step

| Step | Outputs |
|---|---|
| 01 | `outputs/tables/data_quality_summary.csv` (79 tests), `schema_summary.csv`, `missingness_summary.csv`, `data/processed/clean/*.parquet`, `order_items_clean.csv` |
| 02 | `data/processed/historical_features.{csv,parquet}`, `outputs/tables/feature_schema.csv` |
| 03 | `data/processed/historical_outcomes.{csv,parquet}`, `historical_feature_outcome.{csv,parquet}`, `outputs/tables/outcome_schema.csv` |
| 04 | `outputs/tables/kpi_by_{offer,campaign,tenure_value,behavior_segment,membership}.csv`, `incremental_ncm_*.csv`, `outputs/figures/04_*.png` |
| 05 | `outputs/models/logistic_propensity_model.joblib`, `logistic_model_features.json`, `outputs/tables/logistic_{metrics,validation_grid,test_predictions,test_decile_lift}.csv`, `outputs/figures/05_*.png` |

## Key definitions and assumptions

### Table grain
`historical_feature_outcome` has **one row per `assignment_id`**, meaning one customer × one historical campaign. That is 26,600 rows covering CP01–CP07.

### Leakage rule
- A historical feature can only use events with timestamp **`< send_ts`**. Windows are `[send_ts − N days, send_ts)`.
- July features will use events with timestamp **`<= policy_as_of_ts`**.
- Step 02 proves this in two ways:
  - a per-family check that the maximum of (event_ts − send_ts) is below 0;
  - a perturbation test: fake events are injected at or after `send_ts`, and the features must not change.

### Cleaning (Step 01)
- 160 exact duplicate `order_items` rows are removed.
- Cancelled orders (all of which have zero financials) are kept for audit but are never purchases.
- 220 anonymous web sessions are kept in the raw data but excluded from customer features.
- Organic preperiod orders are used only as purchase history.

### 14-day outcome
An order counts as an outcome for an assignment only if all of these hold:
- it has the same customer and the same campaign;
- its status is `completed`;
- `send_ts <= order_ts < send_ts + 14d`.

`purchase_14d` = 1 if there is at least one such order. `aov_14d` = net revenue / orders, and is missing for non-purchasers.

### NCM
`ncm_14d = net_revenue − cogs − shipping_cost − refund_amount_applicable`

Net revenue already equals gross − discount + shipping revenue, so the formula doesn't double count. Step 03 checks that it reconciles.

### Return timing
This rule is set by `RETURN_RULE` in Step 03.
- By default, a refund counts toward an assignment only if the return happens **inside that assignment's 14-day window**.
- Refunds after the window are recorded separately (`post_window_*`). `ncm_14d_all_returns` is provided for sensitivity checks.
- Returns arrive 3–31 days after the order, so most refunds on campaign orders fall after the window: $83.8K post-window versus $15.4K in-window.

### Promo cost
- PCT offers: the observed discount on redeemed orders.
- FREE_SHIP: the waived fee is inferred and flagged (`promo_cost_inferred`). The inference uses the observed shipping rule: a $5.99 fee applies only when merchandise is under about $75, and this rule holds on every order that didn't use FREE_SHIP.
- NO_OFFER: 0.

### Profile fields
There's no history table with effective dates for profile fields, so they are treated as descriptors available at decision time. `products.return_rate_prior` isn't used because the documentation doesn't say when it was measured.

### Logistic baseline
- Train on CP01–CP05, choose `C` on CP06, and test once on CP07. The final model is then refit on CP01–CP07.
- Preprocessing and the model are saved together as one sklearn `Pipeline`:
  - numeric features: median imputation with missing indicators, then scaling;
  - categorical features: a `"Missing"` category, then one-hot encoding.
- Treatment, delivery, IDs and every outcome field are excluded from X.

## Required packages
pandas, numpy, scikit-learn, matplotlib, joblib, pyarrow, plus jupyter/nbconvert to execute the notebooks. They are listed in `requirements.txt`, and the pipeline was tested on Python 3.14 (Anaconda).
