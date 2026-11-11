"""PromoPilot July policy dashboard. Reads saved Step 09/10 outputs only (no modeling). Run: streamlit run dashboard/app.py"""
from pathlib import Path
import pandas as pd
import streamlit as st

T = Path(__file__).resolve().parents[1] / "outputs" / "tables"
ACTIONS = ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]
BUDGET, MAX_OFFER, MAX_PCT20, MIN_HOLD = 62_000, 0.60, 0.15, 0.05


st.set_page_config(page_title="PromoPilot July Policy", layout="wide")


@st.cache_data
def load():
    return (pd.read_csv(T / "optimized_policy_detailed.csv"), pd.read_csv(T / "policy_action_summary.csv"),
            pd.read_csv(T / "baseline_policy_comparison.csv"), pd.read_csv(T / "customer_policy_audit.csv"))


pol, act, cmp, aud = load()
opt = cmp.set_index("policy").loc["OPTIMIZED_MILP"]
st.title("PromoPilot · July promotion policy")
st.caption("Expected values from the T-Learner (NCM) and expected-cost models; MILP-optimized under the official constraints.")
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Policy mix", "Baseline comparison", "Customer lookup"])

with tab1:
    c = st.columns(5)
    c[0].metric("Expected incremental NCM", f"${opt.total_expected_incremental_ncm:,.0f}")
    c[1].metric("Expected promo spend", f"${opt.expected_promo_spend:,.0f}", f"limit ${BUDGET:,}", delta_color="off")
    c[2].metric("Offer share", f"{opt.offer_share:.1%}", f"limit {MAX_OFFER:.0%}", delta_color="off")
    c[3].metric("PCT_20 share of offers", f"{opt.pct20_share_among_offered:.1%}", f"limit {MAX_PCT20:.0%}", delta_color="off")
    c[4].metric("Measurement holdout", f"{pol.measurement_holdout.mean():.1%}", f"min {MIN_HOLD:.0%}", delta_color="off")
    st.progress(min(opt.expected_promo_spend / BUDGET, 1.0), text=f"Budget used: {opt.expected_promo_spend / BUDGET:.1%}")
    st.progress(min(opt.offer_share / MAX_OFFER, 1.0), text=f"Offer cap used: {opt.offer_share / MAX_OFFER:.1%}")

with tab2:
    a = act.set_index("action").reindex(ACTIONS)
    l, r = st.columns(2)
    l.subheader("Customers by action"); l.bar_chart(a.customers)
    r.subheader("Expected incremental NCM by action ($)"); r.bar_chart(a.expected_incremental_ncm)
    st.dataframe(a.style.format({"share": "{:.1%}", "expected_incremental_ncm": "${:,.0f}", "mean_incremental_ncm": "${:,.2f}",
                                 "expected_promo_spend": "${:,.0f}", "mean_purchase_probability": "{:.1%}"}), width="stretch")

with tab3:
    st.subheader("Expected incremental NCM vs all NO_OFFER ($)")
    st.bar_chart(cmp.set_index("policy").total_expected_incremental_ncm)
    show = ["policy"] + [f"n_{x}" for x in ACTIONS] + ["offer_share", "pct20_share_among_offered", "expected_promo_spend",
                                                       "total_expected_ncm", "total_expected_incremental_ncm", "incremental_ncm_per_promo_dollar"]
    st.dataframe(cmp[show].style.format({"offer_share": "{:.1%}", "pct20_share_among_offered": "{:.1%}", "expected_promo_spend": "${:,.0f}",
                                         "total_expected_ncm": "${:,.0f}", "total_expected_incremental_ncm": "${:,.0f}",
                                         "incremental_ncm_per_promo_dollar": "{:.2f}"}, na_rep="–"), width="stretch", hide_index=True)

with tab4:
    l, r = st.columns([1, 2])
    q = l.text_input("Customer ID contains")
    acts = r.multiselect("Recommended action", ACTIONS, default=ACTIONS)
    cols = ["customer_id", "recommended_action", "predicted_purchase_probability", "predicted_ncm", "estimated_incremental_ncm",
            "estimated_promo_cost", "measurement_holdout", "customer_tier", "is_member"]
    view = aud[aud.recommended_action.isin(acts)]
    if q:
        view = view[view.customer_id.astype(str).str.contains(q.strip())]
    st.write(f"{len(view):,} customers")
    st.dataframe(view[cols].style.format({"predicted_purchase_probability": "{:.1%}", "predicted_ncm": "${:,.2f}",
                                          "estimated_incremental_ncm": "${:,.2f}", "estimated_promo_cost": "${:,.2f}"}),
                 width="stretch", hide_index=True)
    if len(view) == 1:
        row = view.iloc[0]
        st.subheader(f"Customer {row.customer_id}: predicted NCM under each action")
        st.bar_chart(pd.Series({x: row[f"predicted_ncm_{x}"] for x in ACTIONS}, name="predicted NCM ($)"))
