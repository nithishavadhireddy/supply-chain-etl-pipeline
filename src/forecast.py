"""
forecast.py

Demand forecasting using Random Forest on historical supply chain data.
Predicts units_sold per SKU for the next N days.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)


def add_forecast_features(df: pd.DataFrame, target_col: str = "units_sold") -> pd.DataFrame:
    """Add time-based and rolling features for forecasting."""
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values(["sku", "order_date"])

    # Calendar features
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["month"] = df["order_date"].dt.month
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["is_month_end"] = df["order_date"].dt.is_month_end.astype(int)

    # Rolling demand features per SKU
    for window in [7, 14, 30]:
        df[f"demand_roll_mean_{window}d"] = (
            df.groupby("sku")[target_col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"demand_roll_std_{window}d"] = (
            df.groupby("sku")[target_col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    # Lag features
    for lag in [1, 7, 14]:
        df[f"demand_lag_{lag}d"] = df.groupby("sku")[target_col].transform(lambda x: x.shift(lag))

    df = df.dropna()
    return df


FEATURE_COLS = [
    "day_of_week", "month", "week_of_year", "is_month_end",
    "stock_quantity", "stockout_flag", "days_of_supply",
    "demand_roll_mean_7d", "demand_roll_mean_14d", "demand_roll_mean_30d",
    "demand_roll_std_7d", "demand_roll_std_14d", "demand_roll_std_30d",
    "demand_lag_1d", "demand_lag_7d", "demand_lag_14d",
]


def train_forecast_model(df: pd.DataFrame, config: dict) -> Tuple[RandomForestRegressor, dict]:
    target = config.get("target_col", "units_sold")
    df = add_forecast_features(df, target)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get("random_state", 42)
    )

    model = RandomForestRegressor(
        n_estimators=config.get("n_estimators", 200),
        max_depth=config.get("max_depth", 8),
        random_state=config.get("random_state", 42),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 3),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        "r2": round(model.score(X_test, y_test), 4),
    }
    logger.info("Forecast model metrics: %s", metrics)
    return model, metrics


def generate_sku_forecasts(model: RandomForestRegressor, df: pd.DataFrame,
                            horizon_days: int = 30) -> pd.DataFrame:
    """Generate next N-day demand forecasts per SKU using the last known feature values."""
    df = add_forecast_features(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    latest = df.sort_values("order_date").groupby("sku").last().reset_index()

    forecasts = []
    for _, row in latest.iterrows():
        X = row[available].values.reshape(1, -1)
        pred = model.predict(X)[0]
        forecasts.append({
            "sku": row["sku"],
            "forecast_units": round(max(pred, 0), 1),
            "horizon_days": horizon_days,
            "forecast_total": round(max(pred, 0) * horizon_days, 1),
        })

    return pd.DataFrame(forecasts).sort_values("forecast_total", ascending=False)
