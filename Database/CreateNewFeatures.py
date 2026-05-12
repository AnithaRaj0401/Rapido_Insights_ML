import numpy as np
import pandas as pd

RUSH_HOUR_WINDOWS = set(range(7, 10)) | set(range(17, 21))
DEFAULT_LONG_DISTANCE_KM = 15.0
MAX_SIGNUP_DAYS = 3650

def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _normalize_rate(series):
    series = _safe_numeric(series)
    series = series.where(series <= 1, series / 100)
    return series.clip(0, 1)


def compute_driver_reliability_score(drivers: pd.DataFrame) -> pd.Series:
    if drivers is None or drivers.empty:
        return pd.Series(dtype=float)

    df = drivers.copy()
    df["avg_driver_rating"] = _safe_numeric(df.get("avg_driver_rating", pd.Series(dtype=float)))
    df["acceptance_rate"] = _normalize_rate(df.get("acceptance_rate", pd.Series(dtype=float)))
    df["delay_rate"] = _normalize_rate(df.get("delay_rate", pd.Series(dtype=float)))

    avg_rating_norm = (df["avg_driver_rating"] / 5).clip(0, 1)
    delay_penalty = (1 - df["delay_rate"]).clip(0, 1)

    score = (
        0.5 * avg_rating_norm
        + 0.3 * df["acceptance_rate"]
        + 0.2 * delay_penalty
    )

    return score.clip(0, 1).fillna(0)


def compute_customer_loyalty_score(customers: pd.DataFrame) -> pd.Series:
    if customers is None or customers.empty:
        return pd.Series(dtype=float)

    df = customers.copy()
    df["completed_rides"] = _safe_numeric(df.get("completed_rides", pd.Series(dtype=float)))
    df["total_bookings"] = _safe_numeric(df.get("total_bookings", pd.Series(dtype=float)))
    df["cancellation_rate"] = _normalize_rate(df.get("cancellation_rate", pd.Series(dtype=float)))
    df["customer_signup_days_ago"] = _safe_numeric(df.get("customer_signup_days_ago", pd.Series(dtype=float)))

    completed_ratio = np.where(
        df["total_bookings"] > 0,
        df["completed_rides"] / df["total_bookings"],
        0,
    )
    completed_ratio = pd.Series(completed_ratio).clip(0, 1)

    tenure_score = (df["customer_signup_days_ago"] / MAX_SIGNUP_DAYS).clip(0, 1)

    score = (
        0.45 * completed_ratio
        + 0.35 * (1 - df["cancellation_rate"])
        + 0.20 * tenure_score
    )

    return score.clip(0, 1).fillna(0)


def create_features(
    bookings: pd.DataFrame,
    customers: pd.DataFrame | None = None,
    drivers: pd.DataFrame | None = None,
    long_distance_threshold: float = DEFAULT_LONG_DISTANCE_KM,
) -> pd.DataFrame:
    if bookings is None or bookings.empty:
        return pd.DataFrame()

    df = bookings.copy()

    if "fare" in df.columns and "distance" in df.columns:
        df["fare_per_km"] = _safe_numeric(df["fare"]) / _safe_numeric(df["distance"]).replace({0: np.nan})

    time_col = None
    if "actual_ride_time_min" in df.columns:
        time_col = "actual_ride_time_min"
    elif "estimated_ride_time_min" in df.columns:
        time_col = "estimated_ride_time_min"

    if time_col is not None and "fare" in df.columns:
        df["fare_per_min"] = _safe_numeric(df["fare"]) / _safe_numeric(df[time_col]).replace({0: np.nan})

    if "hour_of_day" in df.columns:
        df["rush_hour_flag"] = df["hour_of_day"].apply(
            lambda x: 1 if x in RUSH_HOUR_WINDOWS else 0
        ).astype(int)
    elif "booking_time" in df.columns:
        hours = pd.to_datetime(df["booking_time"], errors="coerce").dt.hour
        df["rush_hour_flag"] = hours.apply(
            lambda x: 1 if x in RUSH_HOUR_WINDOWS else 0
        ).astype(int)

    if "ride_distance_km" in df.columns:
        df["long_distance_flag"] = (
            _safe_numeric(df["ride_distance_km"]).fillna(0) > long_distance_threshold
        ).astype(int)

    df["city_pair"] = df.apply(
        lambda row: " -> ".join(
            filter(
                None,
                [
                    str(row.get("pickup_location", "")).strip(),
                    str(row.get("drop_location", "")).strip(),
                ],
            )
        ),
        axis=1,
    )

    if drivers is not None and not drivers.empty and "driver_id" in df.columns:
        drivers = drivers.copy()
        drivers["driver_reliability_score"] = compute_driver_reliability_score(drivers)
        df = df.merge(
            drivers[["driver_id", "driver_reliability_score"]],
            on="driver_id",
            how="left",
        )

    if customers is not None and not customers.empty and "customer_id" in df.columns:
        customers = customers.copy()
        customers["customer_loyalty_score"] = compute_customer_loyalty_score(customers)
        df = df.merge(
            customers[["customer_id", "customer_loyalty_score"]],
            on="customer_id",
            how="left",
        )

    return df
