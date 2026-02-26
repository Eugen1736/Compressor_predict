import pandas as pd
import numpy as np
from datetime import timedelta


def analyze_structure(df: pd.DataFrame) -> None:
    """Print information about column types, numeric statistics and categorical frequencies."""
    print("DataFrame info:\n", df.info())
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()

    if numeric:
        stats = df[numeric].agg(["mean", "std", "min", "max"]).T
        print("\nNumeric statistics (mean, std, min, max):\n", stats)

    if categorical:
        print("\nCategorical value counts (top 10 for each):")
        for col in categorical:
            print(f"\nColumn {col}:")
            print(df[col].value_counts(dropna=False).head(10))


def make_date_index(df: pd.DataFrame, n_rows: int) -> pd.DatetimeIndex:
    """Build a datetime index preserving the original spacing (approximately).

    We look for any column whose name contains "date" or "time" (case
    insensitive) and can be parsed into datetimes.  The first eligible column
    is used to extract spacing.  If none are found we fall back to a simple
    minute-based sequence starting now.
    """
    # search for potential date/time columns
    candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time"))]
    for col in candidates:
        series = pd.to_datetime(df[col], errors="coerce")
        if series.notna().sum() > 0:
            series = series.dropna()
            if len(series) >= 2:
                deltas = series.diff().dropna()
                median_delta = deltas.median()
            else:
                median_delta = pd.Timedelta(seconds=1)
            start = series.iloc[0]
            return pd.date_range(start=start, periods=n_rows, freq=median_delta)

    # fallback to simple range of minutes if no date column found
    return pd.date_range(start=pd.Timestamp.now(), periods=n_rows, freq="T")


def generate_base(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Generate a DataFrame with the same schema and numerical statistics as `df`."""
    result = pd.DataFrame()
    date_idx = make_date_index(df, n_rows)
    if "Date&time" in df.columns:
        result["Date&time"] = date_idx

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        # draw from normal distribution; clip to original min/max if present
        values = np.random.normal(loc=mean, scale=std, size=n_rows)
        if df[col].min() >= 0:
            values = np.clip(values, a_min=0, a_max=None)
        if not np.isfinite(std) or std == 0:
            values = np.full(n_rows, mean)
        result[col] = values

    # copy over any non-numeric, non-datetime columns by sampling with replacement
    other_cols = [c for c in df.columns if c not in numeric_cols and c != "Date&time"]
    for col in other_cols:
        result[col] = np.random.choice(df[col].values, size=n_rows, replace=True)

    return result


def add_noise(df: pd.DataFrame, pct: float = 0.2) -> pd.DataFrame:
    """Apply symmetric noise of up to ±pct to all numeric columns."""
    noisy = df.copy()
    numeric = noisy.select_dtypes(include=[np.number]).columns
    for col in numeric:
        factor = 1 + np.random.uniform(-pct, pct, size=len(noisy))
        noisy[col] = noisy[col] * factor
        if noisy[col].dtype.kind in "iu" and (noisy[col] < 0).any():
            # keep non-negative for integer types
            noisy[col] = noisy[col].clip(lower=0)
    return noisy


def inject_outliers(df: pd.DataFrame, outlier_fraction: float = 0.05) -> pd.DataFrame:
    """Make approximately `outlier_fraction` of entries extreme by flipping IQR boundaries."""
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    for col in numeric:
        series = out[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n = len(series)
        n_out = int(np.ceil(n * outlier_fraction))
        indices = np.random.choice(n, size=n_out, replace=False)
        for idx in indices:
            if np.random.rand() < 0.5:
                # assign below lower bound by a random multiple
                out.at[out.index[idx], col] = lower_bound - np.random.rand() * iqr * 3
            else:
                out.at[out.index[idx], col] = upper_bound + np.random.rand() * iqr * 3
    return out


def save_df(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved synthetic dataset to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate synthetic copies of a CSV dataset for stress testing."
    )
    parser.add_argument("input", help="path to source CSV file", nargs="?", default="CS_2.csv")
    parser.add_argument("--rows", type=int, default=1000, help="number of synthetic rows to generate")
    args = parser.parse_args()

    # read and automatically parse dates if possible
    original = pd.read_csv(args.input, parse_dates=True, infer_datetime_format=True)
    print(f"Analyzing structure of {args.input}")
    analyze_structure(original)

    n = args.rows
    base = generate_base(original, n)

    # derive prefix from input name (drop extension)
    prefix = args.input.rsplit(".", 1)[0]
    save_df(base, f"{prefix}_synthetic_normal.csv")

    noisy = add_noise(base, pct=0.2)
    save_df(noisy, f"{prefix}_synthetic_noisy.csv")

    outliered = inject_outliers(base, outlier_fraction=0.05)
    save_df(outliered, f"{prefix}_synthetic_outliers.csv")



if __name__ == "__main__":
    main()
