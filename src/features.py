"""Step 02 feature logic (pre-decision features), reusable by Step 07 at policy_as_of_ts.

Cutoff rule: historical rows use events with ts < send_ts (inclusive=False);
July scoring uses ts <= policy_as_of_ts (inclusive=True).
"""
import numpy as np
import pandas as pd

PROFILE_CAT = ["region", "acquisition_channel", "customer_tier", "preferred_category", "shipping_zone"]
PROFILE_NUM = ["customer_tenure_days", "is_member", "marketing_consent"]
FAMILIES = {
    "profile": PROFILE_NUM + PROFILE_CAT,
    "purchase": ["prior_order_count", "orders_30d", "orders_90d", "days_since_last_order", "gross_revenue_90d",
                 "net_revenue_90d", "avg_order_value_90d", "discount_amount_90d", "cogs_90d", "shipping_revenue_90d",
                 "shipping_cost_90d", "historical_ncm_90d", "prior_offer_redemption_rate"],
    "returns": ["returns_90d", "refund_amount_90d", "historical_return_rate"],
    "web": ["sessions_7d", "sessions_30d", "pages_viewed_30d", "avg_pages_per_session_30d", "cart_add_sessions_30d",
            "cart_add_rate_30d", "days_since_last_session", "top_category_viewed_30d"],
    "email": ["prior_emails_delivered", "prior_email_opens", "prior_email_clicks", "historical_open_rate",
              "historical_click_rate", "days_since_last_email_open", "days_since_last_email_click"],
    "product": ["units_purchased_90d", "distinct_products_90d", "distinct_categories_90d", "top_purchase_category_90d"],
}
CATEGORICAL = PROFILE_CAT + ["top_category_viewed_30d", "top_purchase_category_90d"]
FEATURES = [f for fam in FAMILIES.values() for f in fam]


def _prior(spine, ev, ts, key, cut, inclusive, audit, family):
    """Join events to spine rows by customer and keep only events known at the row cutoff."""
    m = spine[[key, "customer_id", cut]].merge(ev, on="customer_id")
    m = m[(m[ts] <= m[cut]) if inclusive else (m[ts] < m[cut])].copy()
    m["_age_d"] = (m[cut] - m[ts]).dt.total_seconds() / 86400
    lag = (m[ts] - m[cut]).max() if len(m) else pd.Timedelta(-1, "D")
    audit[family] = max(audit.get(family, lag), lag)
    return m


def _top(df, key, cat, weight=None):
    g = df.groupby([key, cat]).size() if weight is None else df.groupby([key, cat])[weight].sum()
    g = g.rename("_n").reset_index().sort_values([key, "_n", cat], ascending=[True, False, True])
    return g.drop_duplicates(key).set_index(key)[cat]


def _div(a, b):
    return (a / b.where(b > 0)).astype(float)


def build_features(spine, t, key="assignment_id", cut="send_ts", inclusive=False):
    """spine: one row per decision (key, customer_id, cut). t: dict of Step 01 clean tables.
    Returns (features DataFrame with one row per key, audit dict of max event_ts - cutoff per family)."""
    audit = {}
    s = spine[[key, "customer_id", cut]].copy()
    out = s.set_index(key)
    w = lambda m, d: m[m._age_d <= d]
    # Profile
    cu = t["customers"].set_index("customer_id")
    prof = s.join(cu, on="customer_id")
    out["customer_tenure_days"] = ((prof[cut] - prof.signup_date).dt.total_seconds() / 86400).values
    for c in ["is_member", "marketing_consent"] + PROFILE_CAT:
        out[c] = prof[c].values
    # Purchase history (completed orders only)
    orders = t["orders"][t["orders"].is_completed]
    po = _prior(s, orders, "order_ts", key, cut, inclusive, audit, "purchase")
    rets = t["returns"][["order_id", "return_ts", "refund_amount"]]
    pr = po[[key, "order_id", cut]].merge(rets, on="order_id")
    pr = pr[(pr.return_ts <= pr[cut]) if inclusive else (pr.return_ts < pr[cut])]
    if len(pr):
        audit["purchase_refunds"] = (pr.return_ts - pr[cut]).max()
    po = po.merge(pr[[key, "order_id", "refund_amount"]], on=[key, "order_id"], how="left")
    po["known_refund"] = po.refund_amount.fillna(0)
    po["ncm"] = po.net_revenue - po.cogs - po.shipping_cost - po.known_refund
    g_all = po.groupby(key)
    out["prior_order_count"] = g_all.size()
    out["orders_30d"] = w(po, 30).groupby(key).size()
    p90 = w(po, 90).groupby(key)
    out["orders_90d"] = p90.size()
    out["days_since_last_order"] = g_all._age_d.min()
    for src, dst in [("merchandise_gross_revenue", "gross_revenue_90d"), ("net_revenue", "net_revenue_90d"),
                     ("discount_amount", "discount_amount_90d"), ("cogs", "cogs_90d"),
                     ("shipping_revenue", "shipping_revenue_90d"), ("shipping_cost", "shipping_cost_90d"),
                     ("ncm", "historical_ncm_90d")]:
        out[dst] = p90[src].sum()
    # Prior offer redemption: prior commercial offers (excl. current) redeemed by a completed order known at cutoff
    ha = t["historical_assignments"]
    pa = _prior(s, ha.rename(columns={"send_ts": "prior_send_ts", "assignment_id": "prior_assignment_id"}),
                "prior_send_ts", key, cut, inclusive, audit, "prior_assignments")
    if key == "assignment_id":
        pa = pa[pa.prior_assignment_id != pa[key]]
    off = pa[pa.offer_type != "NO_OFFER"]
    red = orders[orders.offer_redeemed == 1][["customer_id", "campaign_id", "order_ts"]]
    off = off.merge(red, on=["customer_id", "campaign_id"], how="left")
    off["redeemed"] = ((off.order_ts <= off[cut]) if inclusive else (off.order_ts < off[cut])).astype(int)
    off = off.groupby([key, "prior_assignment_id"]).redeemed.max().reset_index()
    out["prior_offer_redemption_rate"] = off.groupby(key).redeemed.mean()
    # Returns history (return events known at cutoff)
    rt = rets.merge(t["orders"][["order_id", "customer_id"]], on="order_id")
    pre = _prior(s, rt, "return_ts", key, cut, inclusive, audit, "returns")
    out["returns_90d"] = w(pre, 90).groupby(key).size()
    out["refund_amount_90d"] = w(pre, 90).groupby(key).refund_amount.sum()
    n_ret = po[po.known_refund > 0].groupby(key).order_id.nunique().reindex(out.index, fill_value=0)
    out["historical_return_rate"] = _div(n_ret, out.prior_order_count.fillna(0))
    # Web behaviour (identified sessions only)
    ws = t["web_sessions"][t["web_sessions"].customer_id.notna()]
    pw = _prior(s, ws, "session_ts", key, cut, inclusive, audit, "web")
    w30 = w(pw, 30)
    out["sessions_7d"] = w(pw, 7).groupby(key).size()
    out["sessions_30d"] = w30.groupby(key).size()
    out["pages_viewed_30d"] = w30.groupby(key).pages_viewed.sum()
    out["cart_add_sessions_30d"] = w30.groupby(key).added_to_cart.sum()
    out["days_since_last_session"] = pw.groupby(key)._age_d.min()
    out["top_category_viewed_30d"] = _top(w30, key, "category_viewed")
    # Email engagement (delivered prior campaign emails + opens/clicks known at cutoff)
    out["prior_emails_delivered"] = pa[pa.email_delivered == 1].groupby(key).size()
    pe = _prior(s, t["email_events"].drop(columns=["assignment_id", "campaign_id"]), "event_ts", key, cut, inclusive,
                audit, "email")
    for et, n in [("open", "opens"), ("click", "clicks")]:
        e = pe[pe.event_type == et].groupby(key)
        out[f"prior_email_{n}"] = e.size()
        out[f"days_since_last_email_{et}"] = e._age_d.min()
    # Product / category history (deduplicated items of completed prior-90d orders)
    items = t["order_items"].merge(t["products"][["product_id", "category"]], on="product_id")
    pi = w(po, 90)[[key, "order_id"]].merge(items, on="order_id")
    gi = pi.groupby(key)
    out["units_purchased_90d"] = gi.quantity.sum()
    out["distinct_products_90d"] = gi.product_id.nunique()
    out["distinct_categories_90d"] = gi.category.nunique()
    out["top_purchase_category_90d"] = _top(pi, key, "category", "quantity")
    # Fill counts/sums with 0; rates/recency stay missing when undefined
    zero = ["prior_order_count", "orders_30d", "orders_90d", "gross_revenue_90d", "net_revenue_90d",
            "discount_amount_90d", "cogs_90d", "shipping_revenue_90d", "shipping_cost_90d", "historical_ncm_90d",
            "returns_90d", "refund_amount_90d", "sessions_7d", "sessions_30d", "pages_viewed_30d",
            "cart_add_sessions_30d", "prior_emails_delivered", "prior_email_opens", "prior_email_clicks",
            "units_purchased_90d", "distinct_products_90d", "distinct_categories_90d"]
    out[zero] = out[zero].fillna(0)
    out["avg_order_value_90d"] = _div(out.net_revenue_90d, out.orders_90d)
    out["avg_pages_per_session_30d"] = _div(out.pages_viewed_30d, out.sessions_30d)
    out["cart_add_rate_30d"] = _div(out.cart_add_sessions_30d, out.sessions_30d)
    out["historical_open_rate"] = _div(out.prior_email_opens, out.prior_emails_delivered)
    out["historical_click_rate"] = _div(out.prior_email_clicks, out.prior_emails_delivered)
    return out.reset_index()[[key, "customer_id", cut] + FEATURES], audit
