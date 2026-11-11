"""Shared model plumbing for Steps 06 and 08: feature lists from the Step 02 schema and the XGBoost pipeline factory."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from src.common import TABLES

ACTIONS = ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]
COMMERCIAL = ACTIONS[1:]
XGB_FIXED = dict(objective="reg:squarederror", min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                 tree_method="hist", random_state=42, n_jobs=4)
XGB_GRID = [dict(n_estimators=n, max_depth=d, learning_rate=lr) for n in (200, 400) for d in (2, 3) for lr in (0.03, 0.05)]
XGB_GRID += [dict(n_estimators=100, max_depth=1, learning_rate=0.03), dict(n_estimators=200, max_depth=1, learning_rate=0.03),
             dict(n_estimators=100, max_depth=2, learning_rate=0.03)]  # more-regularized corner: CP06 favoured the smallest models


def feature_lists():
    """Numeric / categorical predictors registered by Step 02 (context columns excluded)."""
    fs = pd.read_csv(TABLES / "feature_schema.csv").query("family != 'context'")
    return fs[fs.type == "numeric"].feature.tolist(), fs[fs.type == "categorical"].feature.tolist()


def make_xgb_pipe(num, cat, params):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("xgb", XGBRegressor(**XGB_FIXED, **params))])


def reg_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": len(y), "rmse": mean_squared_error(y, p) ** 0.5, "mae": mean_absolute_error(y, p),
            "r2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), "obs_mean": y.mean(), "pred_mean": p.mean()}
