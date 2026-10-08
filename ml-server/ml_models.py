import calendar
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DEFAULT_WEEKLY_LIMIT = 5000.0


# ─── helpers ─────────────────────────────────────────────────────────────────
def _num(x, default=0.0):
    """Convert to a JSON-safe float (no NaN / inf)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _next_month_start(d: date) -> date:
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def _forecast(series: pd.Series) -> float:
    """
    Forecast the next value of a (gap-filled) time series.

    Linear regression on the most recent points, blended with a recent moving
    average so a short downward trend can never collapse the forecast to 0.
    """
    values = series.astype(float).values
    n = len(values)
    if n == 0:
        return 0.0
    recent_mean = float(np.mean(values[-4:]))
    if n < 3:
        return recent_mean

    y = values[-8:]
    X = np.arange(len(y)).reshape(-1, 1)
    lr_pred = float(LinearRegression().fit(X, y).predict([[len(y)]])[0])
    lr_pred = max(lr_pred, 0.0)
    return 0.6 * lr_pred + 0.4 * recent_mean


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Keep only the calendar day (YYYY-MM-DD). Timestamps like "...T18:30:00Z" would
    # otherwise become timezone-aware and break week matching (all weeks -> 0).
    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"].astype(str).str.slice(0, 10), errors="coerce"
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["transaction_date", "amount"])
    df["transaction_date"] = df["transaction_date"].dt.normalize()

    iso = df["transaction_date"].dt.isocalendar()
    df["week"] = iso.week.astype(int)
    df["year"] = iso.year.astype(int)  # ISO year (avoids Dec/Jan mismatches)
    df["week_start"] = df["transaction_date"] - pd.to_timedelta(
        df["transaction_date"].dt.dayofweek, unit="D"
    )
    df["month_start"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()
    df["is_weekend"] = df["transaction_date"].dt.dayofweek.isin([5, 6]).astype(int)
    if "category_name" not in df.columns:
        df["category_name"] = None
    df["category_name"] = df["category_name"].fillna("Uncategorized").astype(str)
    return df


# ─── main entry ──────────────────────────────────────────────────────────────
def run_analysis(transactions: list, weekly_limit=None, today_str=None) -> dict:
    df = pd.DataFrame(transactions)
    if df.empty or len(df) < 3:
        return {"error": "Not enough data"}

    df = build_features(df)
    if len(df) < 3:
        return {"error": "Not enough valid transactions (check dates and amounts)"}

    try:
        today = pd.to_datetime(today_str).date() if today_str else None
    except Exception:
        today = None
    if today is None:
        today = date.today()
    # Data dated in the future should not be ignored
    today = max(today, df["transaction_date"].max().date())

    threshold = _num(weekly_limit, DEFAULT_WEEKLY_LIMIT)
    if threshold <= 0:
        threshold = DEFAULT_WEEKLY_LIMIT

    # ─── Weekly series (gap-filled so quiet weeks count as 0) ────────────────
    this_week_start = _week_start(today)
    first_week = df["week_start"].min().date()
    all_weeks = pd.date_range(first_week, this_week_start, freq="7D")
    weekly_sum = df.groupby("week_start")["amount"].sum()
    weekly_full = weekly_sum.reindex(all_weeks, fill_value=0.0)

    # Train on completed weeks when we have enough; the current week is partial.
    completed = weekly_full[weekly_full.index.date < this_week_start]
    train_weekly = completed if len(completed) >= 2 else weekly_full

    predicted_spending = _forecast(train_weekly)
    # Never predict 0 if the user has any recent spending history
    if predicted_spending <= 0 and weekly_full.sum() > 0:
        predicted_spending = float(weekly_full[weekly_full > 0].tail(4).mean())

    next_week_start = this_week_start + timedelta(days=7)
    next_week_number = int(next_week_start.isocalendar()[1])

    # ─── Monthly prediction ──────────────────────────────────────────────────
    next_month_start = _next_month_start(today)
    days_next_month = calendar.monthrange(next_month_start.year, next_month_start.month)[1]
    from_weekly = predicted_spending * days_next_month / 7.0

    this_month_start = _month_start(today)
    monthly_sum = df.groupby("month_start")["amount"].sum()
    monthly_full = monthly_sum.reindex(
        pd.date_range(df["month_start"].min(), pd.Timestamp(this_month_start), freq="MS"),
        fill_value=0.0,
    )
    completed_months = monthly_full[monthly_full.index.date < this_month_start]
    if len(completed_months) >= 2:
        predicted_monthly = 0.5 * from_weekly + 0.5 * _forecast(completed_months)
    else:
        predicted_monthly = from_weekly
    predicted_monthly = max(predicted_monthly, 0.0)

    # ─── Smart budget recommendation ─────────────────────────────────────────
    recommended_budget = round(predicted_spending * 0.85, 2)
    recommended_monthly_budget = round(predicted_monthly * 0.85, 2)

    # ─── Overspending risk (Decision Tree + explanation) ─────────────────────
    weekly = pd.DataFrame({"week_start": weekly_full.index, "amount": weekly_full.values})
    weekend_w = df[df["is_weekend"] == 1].groupby("week_start")["amount"].sum()
    food_w = df[df["category_name"].str.lower() == "food"].groupby("week_start")["amount"].sum()
    txn_w = df.groupby("week_start").size()
    weekly["weekend_amt"] = weekly["week_start"].map(weekend_w).fillna(0.0)
    weekly["food_amt"] = weekly["week_start"].map(food_w).fillna(0.0)
    weekly["txn_count"] = weekly["week_start"].map(txn_w).fillna(0).astype(float)
    weekly["food_pct"] = np.where(weekly["amount"] > 0, weekly["food_amt"] / weekly["amount"] * 100, 0.0)
    weekly["overspend"] = (weekly["amount"] > threshold).astype(int)

    feat_cols = ["amount", "weekend_amt", "food_pct"]
    overspend_risk = 1 if predicted_spending > threshold else 0
    reason = "Spending appears within normal range."

    if len(weekly) >= 2 and weekly["overspend"].nunique() > 1:
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(weekly[feat_cols], weekly["overspend"])
        pred_features = pd.DataFrame(
            [[predicted_spending, weekly["weekend_amt"].mean(), weekly["food_pct"].mean()]],
            columns=feat_cols,
        )
        overspend_risk = int(dt.predict(pred_features)[0]) or overspend_risk

    if overspend_risk:
        last = weekly.iloc[-1]
        if predicted_spending > threshold:
            reason = (f"Predicted spending ₹{predicted_spending:,.0f} is above your "
                      f"weekly limit of ₹{threshold:,.0f}.")
        elif last["weekend_amt"] > threshold * 0.4:
            reason = (f"High weekend spending detected (₹{last['weekend_amt']:,.0f}). "
                      "Consider reducing weekend expenses.")
        elif last["food_pct"] > 50:
            reason = (f"Food is {last['food_pct']:.0f}% of weekly spending. "
                      "Budget for meals more strictly.")
        else:
            reason = "Spending trend is rising and may exceed your weekly limit."

    # ─── Category drift detection ────────────────────────────────────────────
    drift_alerts = []
    cat_weekly = df.groupby(["week_start", "category_name"])["amount"].sum().reset_index()
    if len(cat_weekly["week_start"].unique()) >= 2:
        latest_week = cat_weekly["week_start"].max()
        avg_by_cat = cat_weekly.groupby("category_name")["amount"].mean()
        latest_cat = cat_weekly[cat_weekly["week_start"] == latest_week].set_index("category_name")["amount"]
        for cat, latest_val in latest_cat.items():
            avg = avg_by_cat.get(cat, latest_val)
            if avg > 0 and latest_val > avg * 1.5:
                drift_alerts.append(f"Unusual spike in {cat}: ₹{latest_val:,.0f} vs avg ₹{avg:,.0f}")

    # ─── K-Means behavioural segmentation ────────────────────────────────────
    cluster_label = "Steady Spender"
    active = weekly[weekly["amount"] > 0].reset_index(drop=True)
    if len(active) >= 3:
        seg = active[["amount", "txn_count", "weekend_amt"]].values
        scaler = StandardScaler()
        seg_scaled = scaler.fit_transform(seg)
        km = KMeans(n_clusters=min(3, len(active)), random_state=42, n_init=10)
        active["cluster"] = km.fit_predict(seg_scaled)
        last_cluster = int(active["cluster"].iloc[-1])
        profiles = []
        for i, center in enumerate(km.cluster_centers_):
            amt, txns, wknd = scaler.inverse_transform([center])[0]
            profiles.append({"id": i, "amt": amt, "txns": txns, "wknd": wknd})
        mine = next(p for p in profiles if p["id"] == last_cluster)
        highest_amt = max(p["amt"] for p in profiles)
        if mine["wknd"] > np.mean([p["wknd"] for p in profiles]) * 1.2:
            cluster_label = "Weekend Splurger"
        elif mine["txns"] > np.mean([p["txns"] for p in profiles]) * 1.2:
            cluster_label = "Daily Small Spender"
        elif mine["amt"] >= highest_amt * 0.9:
            cluster_label = "Rare but Heavy Spender"

    # ─── Weekend vs weekday ──────────────────────────────────────────────────
    weekend_total = float(df[df["is_weekend"] == 1]["amount"].sum())
    weekday_total = float(df[df["is_weekend"] == 0]["amount"].sum())
    weekend_insight = None
    if weekend_total > weekday_total * 1.3:
        weekend_insight = "You spend significantly more on weekends. Consider setting a weekend budget."

    # ─── Trends for charts ───────────────────────────────────────────────────
    weekly_trend = [
        {"year": int(ws.isocalendar()[0]), "week": int(ws.isocalendar()[1]),
         "label": f"W{int(ws.isocalendar()[1])}", "total": round(_num(v), 2)}
        for ws, v in zip(weekly_full.index, weekly_full.values)
    ][-12:]
    monthly_trend = [
        {"label": ms.strftime("%b %Y"), "total": round(_num(v), 2)}
        for ms, v in zip(monthly_full.index, monthly_full.values)
    ][-12:]

    return {
        "predicted_spending": round(_num(predicted_spending), 2),
        "predicted_monthly": round(_num(predicted_monthly), 2),
        "recommended_budget": _num(recommended_budget),
        "recommended_monthly_budget": _num(recommended_monthly_budget),
        "overspend_risk": bool(overspend_risk),
        "reason": reason,
        "cluster_label": cluster_label,
        "week_number": next_week_number,
        "next_month": next_month_start.strftime("%B %Y"),
        "drift_alerts": drift_alerts,
        "weekend_insight": weekend_insight,
        "weekly_trend": weekly_trend,
        "monthly_trend": monthly_trend,
        "weekend_total": round(weekend_total, 2),
        "weekday_total": round(weekday_total, 2),
    }
