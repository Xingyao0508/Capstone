"""Draws the workflow / lineage / roadmap diagrams used in Walmart_Pipeline_Walkthrough.md (run: python docs/make_diagrams.py)."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
INK, MUTED, GREY = "#0b0b0b", "#52514e", "#898781"
C = {"raw": ("#eef0f3", GREY), "nb": ("#e3eefb", "#2a78d6"), "out": ("#fdf0e6", "#eb6834"),
     "master": ("#e4f5ee", "#1baf7a"), "future": ("#ffffff", "#b5b3ad")}
H = 1.0


def canvas(w, h):
    global H
    H = h
    fig, ax = plt.subplots(figsize=(w, h)); fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(-0.005, 1.005); ax.set_ylim(-0.005, 1.005); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, kind, title, body="", fs=7.2, tfs=8.4, dashed=False):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", fc=fc, ec=ec, lw=1.2,
                                ls="--" if dashed else "-"))
    pad = 0.07 / H
    ax.text(x + 0.012, y + h - pad, title, ha="left", va="top", fontsize=tfs, fontweight="bold", color=MUTED if dashed else INK)
    if body:
        ax.text(x + 0.012, y + h - pad - tfs * 1.55 / 72 / H, body, ha="left", va="top", fontsize=fs,
                color=MUTED if dashed else INK, linespacing=1.4)


def arrow(ax, p, q, color=INK, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def note(ax, x, y, t, ha="left"):
    ax.text(x, y, t, fontsize=6.8, color=MUTED, style="italic", ha=ha, va="center", linespacing=1.3)


# ---------- Figure 1: complete workflow ----------
fig, ax = canvas(7.2, 9.6)
for i, (k, t) in enumerate([("raw", "Raw input"), ("nb", "Notebook + transformation"), ("out", "Saved output file(s)"), ("master", "Master modeling table")]):
    ax.add_patch(FancyBboxPatch((i * 0.25, 0.985), 0.03, 0.012, boxstyle="round,pad=0", fc=C[k][0], ec=C[k][1]))
    ax.text(0.04 + i * 0.25, 0.991, t, fontsize=7.2, va="center", color=MUTED)
XR, WR, XN, WN, XO, WO = 0.0, 0.215, 0.255, 0.385, 0.68, 0.32
r1, r2, r3 = (0.80, 0.165), (0.605, 0.15), (0.40, 0.165)
box(ax, XR, r1[0], WR, r1[1], "raw", "data/raw/ (11 CSVs)",
    "customers, campaigns\nhistorical_assignments\nweb_sessions\nemail_events\norders, order_items\nproducts, returns\noffer_catalog\npolicy_scoring_population", fs=6.6)
box(ax, XN, r1[0], WN, r1[1], "nb", "01_data_validation.ipynb",
    "79 data-quality tests: shapes, keys,\nlinks between tables, missing values,\nduplicates, timing, money totals, offers.\nCleaning: drop 160 duplicate item rows;\nadd is_completed / is_identified flags")
box(ax, XO, r1[0], WO, r1[1], "out", "Step 01 outputs",
    "data/processed/clean/\n  11 x <table>.parquet\n  order_items_clean.csv\noutputs/tables/\n  data_quality_summary.csv\n  schema_summary.csv\n  missingness_summary.csv", fs=6.8)
box(ax, XN, r2[0], WN, r2[1], "nb", "02_build_historical_features.ipynb",
    "src/features.py build_features():\n43 features per assignment from events\nstrictly before send_ts.\nLeakage tests: max event lag < 0;\ninjected future events change nothing")
box(ax, XO, r2[0], WO, r2[1], "out", "Step 02 outputs",
    "data/processed/\n  historical_features\n  .csv / .parquet\n  26,600 rows x 50 cols\noutputs/tables/\n  feature_schema.csv", fs=6.8)
box(ax, XN, r3[0], WN, r3[1], "nb", "03_construct_14_day_outcomes.ipynb",
    "Eligible order: same customer + campaign,\ncompleted, send_ts <= order_ts < +14 days.\nBuilds purchase_14d, revenue and cost\ntotals, promo cost, in-window refunds,\nncm_14d; joins to historical_features")
box(ax, XO, r3[0], WO, r3[1], "out", "Step 03 outputs",
    "data/processed/\n  historical_outcomes\n  .csv / .parquet\n  26,600 rows x 20 cols\noutputs/tables/\n  outcome_schema.csv", fs=6.8)
box(ax, 0.12, 0.275, 0.76, 0.075, "master", "data/processed/historical_feature_outcome.csv / .parquet  (Step 03)",
    "26,600 rows x 69 cols  |  one row = one customer x one historical campaign assignment", fs=7)
box(ax, 0.0, 0.125, 0.485, 0.115, "nb", "04_business_analysis.ipynb",
    "KPIs (conversion, AOV, redemption, promo spend,\nNCM) by offer, campaign, tenure x value,\nintent, membership; targeting-bias check;\nexploration incremental NCM with 95% CIs", fs=6.9)
box(ax, 0.515, 0.125, 0.485, 0.115, "nb", "05_logistic_propensity_model.ipynb",
    "X = 43 pre-send features, Y = purchase_14d\ntrain CP01-05 | tune C on CP06 | test CP07\nimpute + scale + one-hot + LogisticRegression\nin one sklearn Pipeline; refit on CP01-07", fs=6.9)
box(ax, 0.0, 0.0, 0.485, 0.095, "out", "Step 04 outputs (outputs/)",
    "tables/kpi_by_{offer, campaign, tenure_value,\n  behavior_segment, membership}.csv\ntables/incremental_ncm_*.csv (2)  |  figures/04_*.png (3)", fs=6.7)
box(ax, 0.515, 0.0, 0.485, 0.095, "out", "Step 05 outputs (outputs/)",
    "models/logistic_propensity_model.joblib + _features.json\ntables/logistic_{metrics, validation_grid,\n  test_predictions, test_decile_lift}.csv\nfigures/05_logistic_calibration_lift.png", fs=6.7)
ym = lambda r: r[0] + r[1] / 2
arrow(ax, (XR + WR, ym(r1)), (XN, ym(r1)))
for r in (r1, r2, r3):
    arrow(ax, (XN + WN, ym(r)), (XO, ym(r)))
arrow(ax, (XO + WO / 2, r1[0]), (XN + WN / 2, r2[0] + r2[1]), rad=-0.12); note(ax, 0.66, 0.778, "clean tables", "right")
arrow(ax, (XO + WO / 2, r2[0]), (XN + WN / 2, r3[0] + r3[1]), rad=-0.12); note(ax, 0.66, 0.583, "historical_features", "right")
arrow(ax, (XR + WR / 2, r1[0]), (XN, ym(r3)), rad=0.25, color=GREY, ls="--")
note(ax, 0.01, 0.62, "clean orders,\nreturns and\nassignments\nalso read by 03")
arrow(ax, (XN + WN / 2, r3[0]), (0.40, 0.35)); arrow(ax, (XO + WO / 2, r3[0]), (0.66, 0.35), rad=-0.1)
note(ax, 0.01, 0.37, "features (02) +\noutcomes (03)\njoined on\nassignment_id")
arrow(ax, (0.30, 0.275), (0.2425, 0.24)); arrow(ax, (0.70, 0.275), (0.7575, 0.24))
note(ax, 0.79, 0.257, "+ feature_schema.csv\n+ outcome_schema.csv")
arrow(ax, (0.2425, 0.125), (0.2425, 0.095)); arrow(ax, (0.7575, 0.125), (0.7575, 0.095))
fig.savefig(OUT / "workflow_full.png", dpi=220); plt.close(fig)

# ---------- Figure 2: output lineage ----------
fig, ax = canvas(7.2, 3.9)
rows = [(0.80, "raw", "data/raw/*.csv", "11 supplied tables (never modified)"),
        (0.56, "out", "data/processed/clean/*.parquet", "01: de-duplicated items, status / identity flags"),
        (0.32, "out", "data/processed/historical_features", "02: 26,600 x 50 (IDs/context + 43 features)"),
        (0.08, "out", "data/processed/historical_outcomes", "03: 26,600 x 20 (14-day outcomes)")]
for y, k, t, b in rows:
    box(ax, 0.0, y, 0.45, 0.17, k, t, b, fs=7)
for (y1, *_), (y2, *_) in zip(rows[:-1], rows[1:]):
    arrow(ax, (0.225, y1), (0.225, y2 + 0.17))
box(ax, 0.53, 0.30, 0.47, 0.19, "master", "historical_feature_outcome", "03: features + outcomes joined on assignment_id\n26,600 x 69 - single source of truth", fs=7)
arrow(ax, (0.45, 0.405), (0.53, 0.405)); arrow(ax, (0.45, 0.165), (0.53, 0.33))
box(ax, 0.53, 0.0, 0.22, 0.22, "nb", "04 business analysis", "KPI tables\nincremental NCM\n3 charts", fs=7)
box(ax, 0.78, 0.0, 0.22, 0.22, "nb", "05 logistic model", "model .joblib\nmetrics + grid\nCP07 propensities", fs=7)
arrow(ax, (0.64, 0.30), (0.64, 0.22)); arrow(ax, (0.89, 0.30), (0.89, 0.22))
box(ax, 0.53, 0.60, 0.47, 0.37, "future", "Planned - NOT implemented yet", "06 T-Learner trains on historical_feature_outcome\n07 reruns src/features.py at policy_as_of_ts and\n     can reuse logistic_propensity_model.joblib\n08 promo cost  ->  09 MILP  ->  10 policy_submission.csv", fs=7, dashed=True)
arrow(ax, (0.765, 0.49), (0.765, 0.60), color="#b5b3ad", ls="--")
fig.savefig(OUT / "lineage.png", dpi=220); plt.close(fig)

# ---------- Figure 3: roadmap ----------
fig, ax = canvas(7.2, 1.05)
steps = [("01-05", "DONE", "historical table,\nKPIs, logistic\nbaseline", "master", False),
         ("06", "T-Learner", "predict ncm_14d\nunder each of\nthe 4 actions", "future", True),
         ("07", "Score July", "features at\npolicy_as_of_ts;\n2,776 customers", "future", True),
         ("08", "Promo cost", "expected cost\nper customer\nx action", "future", True),
         ("09", "MILP", "$62k budget,\n<=60% offers,\nPCT_20 <=15%,\n>=5% holdout", "future", True),
         ("10", "Outputs", "policy_submission\n.csv + reports", "future", True)]
w, g = 0.152, 0.0176
for i, (n, t, b, k, d) in enumerate(steps):
    x = i * (w + g)
    box(ax, x, 0.0, w, 1.0, k, f"{n} {t}", b, fs=6.5, tfs=7.4, dashed=d)
    if i:
        arrow(ax, (x - g, 0.5), (x, 0.5), color=GREY)
fig.savefig(OUT / "roadmap.png", dpi=220); plt.close(fig)
print("diagrams written to", OUT)
