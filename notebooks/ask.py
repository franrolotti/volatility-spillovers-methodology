# %% [markdown]
# # Realized (Co)Variances & Semi(Co)Variances – Summary Check
#
# Input: intraday price DataFrame (rows = timestamps, columns = markets).
# Output: summary statistics of:
#   - daily realized variances/covariances (ReCov)
#   - daily positive semicovariances (ReCov⁺)
#   - daily negative semicovariances (ReCov⁻)
# for each series.


# %% Imports
import numpy as np
import pandas as pd


# %% [markdown]
# ## 0. Core helper: compute simple returns from prices
# r_{ji} = P_{ji} - P_{j,i-1}


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Simple returns r_t = P_t - P_{t-1} per column.
    prices: wide DataFrame, index = timestamps, columns = markets.
    """
    prices = prices.sort_index()
    rets = prices.diff().dropna()
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    rets["Date"] = rets.index.date
    return rets


# %% [markdown]
# ## 1. Realized covariance matrices (ReCov)
# ReCov_d = sum_{i in day d} r_i r_i'


def daily_realized_cov(rets: pd.DataFrame) -> dict:
    """
    Compute daily realized covariance matrices:
        R_d = sum_t r_{d,t} r_{d,t}'.
    rets: DataFrame of simple returns + 'Date' column.
    Returns:
        dict[date -> (N x N) DataFrame]
    """
    cov_dict = {}

    for day, grp in rets.groupby("Date"):
        R = grp.drop(columns="Date")
        cols = R.columns
        N = len(cols)

        mat = np.zeros((N, N))
        for row in R.values:
            r = row.reshape(-1, 1)  # (N x 1)
            mat += r @ r.T          # outer product

        cov_dict[pd.to_datetime(day)] = pd.DataFrame(mat, index=cols, columns=cols)

    return cov_dict


# %% [markdown]
# ## 2. Realized semicovariances (ReCov⁺, ReCov⁻)
#
# Split returns into positive/negative parts and build:
#   ReCov⁺_d  (from positive moves),
#   ReCov⁻_d  (from negative moves).


def daily_semicovariances(rets: pd.DataFrame):
    """
    Compute daily positive and negative semicovariance matrices.

    For each day d:
        - rp = max(r, 0)
        - rn = min(r, 0)
        - ReCov⁺ := sum_t (rp_t rp_t' + rp_t rn_t')
        - ReCov⁻ := sum_t (rn_t rn_t' + rn_t rp_t')

    Returns:
        pos_dict, neg_dict : dict[date -> (N x N) DataFrame]
    """
    pos_dict = {}
    neg_dict = {}

    for day, grp in rets.groupby("Date"):
        R = grp.drop(columns="Date")
        cols = R.columns
        N = len(cols)

        cov_p = np.zeros((N, N))
        cov_n = np.zeros((N, N))
        m_plus = np.zeros((N, N))
        m_minus = np.zeros((N, N))

        for row in R.values:
            rp = np.clip(row, 0, None)     # positive part
            rn = np.clip(row, None, 0)     # negative part

            cov_p += np.outer(rp, rp)
            cov_n += np.outer(rn, rn)
            m_plus += np.outer(rp, rn)
            m_minus += np.outer(rn, rp)

        mat_pos = cov_p + m_plus
        mat_neg = cov_n + m_minus

        date = pd.to_datetime(day)
        pos_dict[date] = pd.DataFrame(mat_pos, index=cols, columns=cols)
        neg_dict[date] = pd.DataFrame(mat_neg, index=cols, columns=cols)

    return pos_dict, neg_dict


# %% [markdown]
# ## 3. Convert daily matrices → vech time series
# One column per variance/covariance series (FR, ES, PT, FR-ES, FR-PT, ES-PT, ...)


def cov_dict_to_vech_df(cov_dict: dict) -> pd.DataFrame:
    """
    Flatten a dict of daily covariance matrices into a DataFrame where
    each column is a variance or covariance series (vech of each matrix).

    Ordering:
        [Var(m1), Var(m2), ..., Var(mN),
         Cov(m2,m1), Cov(m3,m1), ..., Cov(mN,m1),
         Cov(m3,m2), ..., Cov(mN,m2), ...]
    """
    # Take markets and fix an ordering
    example_mat = next(iter(cov_dict.values()))
    markets = list(example_mat.columns)
    markets_sorted = sorted(markets)

    # Build column labels
    labels = []
    for i, mi in enumerate(markets_sorted):
        labels.append(mi)  # variance
        for j in range(i):
            mj = markets_sorted[j]
            labels.append(f"{mi}-{mj}")  # covariance

    # Build rows (days)
    rows = []
    dates = []

    for date in sorted(cov_dict.keys()):
        mat = cov_dict[date].loc[markets_sorted, markets_sorted].values
        vec = []

        for i in range(len(markets_sorted)):
            # variance
            vec.append(mat[i, i])
            # covariances with previous markets
            for j in range(i):
                vec.append(mat[i, j])

        rows.append(vec)
        dates.append(pd.to_datetime(date))

    vech_df = pd.DataFrame(rows, index=pd.to_datetime(dates), columns=labels)
    return vech_df


# %% [markdown]
# ## 4. Summary statistics helper (per series)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary stats per column:
        Mean, SD, Min, Q1, Median, Q3, Max
    """
    desc = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    out = desc.rename(
        columns={
            "mean": "Mean",
            "std": "SD",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Max",
        }
    )[["Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"]]
    return out


# %% [markdown]
# ## 5. Main wrapper: from prices → summaries for ReCov, ReCov⁺, ReCov⁻


def realized_summary_from_prices(prices: pd.DataFrame):
    """
    Full pipeline:
        prices → simple returns → daily ReCov / ReCov⁺ / ReCov⁻ →
        vech time series → summary statistics.

    Input:
        prices: wide DataFrame with intraday prices.
    Returns:
        cov_df, semi_pos_df, semi_neg_df,
        cov_summary, semi_pos_summary, semi_neg_summary,
        comparison (multi-panel table)
    """
    # 1) returns
    rets = compute_simple_returns(prices)

    # 2) daily realized covariance
    daily_cov = daily_realized_cov(rets)

    # 3) daily semicovariances
    daily_pos, daily_neg = daily_semicovariances(rets)

    # 4) vech time series
    cov_df = cov_dict_to_vech_df(daily_cov)
    semi_pos_df = cov_dict_to_vech_df(daily_pos)
    semi_neg_df = cov_dict_to_vech_df(daily_neg)

    # 5) summaries per series
    cov_summary = summary_table(cov_df)
    semi_pos_summary = summary_table(semi_pos_df)
    semi_neg_summary = summary_table(semi_neg_df)

    # 6) side-by-side comparison: for each series, compare ReCov / ReCov⁺ / ReCov⁻
    comparison = pd.concat(
        {"ReCov": cov_summary, "ReCov_plus": semi_pos_summary, "ReCov_minus": semi_neg_summary},
        axis=1,
    )

    return (
        cov_df,
        semi_pos_df,
        semi_neg_df,
        cov_summary,
        semi_pos_summary,
        semi_neg_summary,
        comparison,
    )


# %% [markdown]
# ## 6. Example usage
# Suppose you already have a wide price DataFrame `prices`:
#   - index: intraday timestamps
#   - columns: ["BZN|ES", "BZN|FR", "BZN|PT"]
#
# For example:
#
#   df_raw = pd.read_parquet("parquet_files/filtered_data.parquet")
#   prices = (
#       df_raw
#       .sort_values(["Area", "Start DateTime"])
#       .pivot(index="Start DateTime", columns="Area", values="Day-ahead Price (EUR/MWh)")
#       .sort_index()
#   )
#
# Then:


# Uncomment and adapt to your data:
# cov_df, semi_pos_df, semi_neg_df, cov_summary, semi_pos_summary, semi_neg_summary, comparison = (
#     realized_summary_from_prices(prices)
# )
#
# print("=== Realized variances & covariances (ReCov) – summary ===")
# print(cov_summary)
#
# print("\n=== Positive semicovariances (ReCov⁺) – summary ===")
# print(semi_pos_summary)
#
# print("\n=== Negative semicovariances (ReCov⁻) – summary ===")
# print(semi_neg_summary)
#
# print("\n=== Side-by-side comparison (ReCov / ReCov⁺ / ReCov⁻) ===")
# print(comparison)
