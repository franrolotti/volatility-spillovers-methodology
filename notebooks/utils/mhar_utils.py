# mhar_utils.py
import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import KFold   

SEED = 12345


# mhar_utils.py
CV_KFOLD = KFold(n_splits=5, shuffle=True, random_state=SEED)
LASSO_OPTS = dict(
    cv         = CV_KFOLD,
    alphas     = np.logspace(-4, 1, 60),   # full 60-point path
    max_iter   = 1_000_000,
    tol        = 1e-10,
    random_state = SEED,
    n_jobs     = -1                        # parallel across folds
)

FAST_LASSO_OPTS = dict(
    cv       = KFold(n_splits=3, shuffle=True, random_state=SEED),
    alphas   = np.logspace(-3, 0, 12),   # 12 alphas instead of 60
    max_iter = 200_000,                  # still generous
    tol      = 1e-6,
    random_state = SEED,
    n_jobs   = 1                       
)



# ──────────────────────────────
#  Lag construction
# ──────────────────────────────
def create_mhar_lags(df: pd.DataFrame) -> pd.DataFrame:
    lag1 = df.shift(1).add_suffix("_lag1")
    wavg = df.rolling(7 , min_periods=7).mean().shift(1).add_suffix("_wavg")
    mavg = df.rolling(30, min_periods=30).mean().shift(1).add_suffix("_mavg")
    return pd.concat([df, lag1, wavg, mavg], axis=1).dropna()

# ──────────────────────────────
#  Model fitting
# ──────────────────────────────
def fit_mhar_lasso(X, Y, cv=None):
    model = MultiTaskLassoCV(**LASSO_OPTS).fit(X, Y)
    model.fit(X, Y)
    Bd, Bw, Bm = np.split(model.coef_, 3, axis=1)
    phi1 = Bd + Bw/7 + Bm/30
    return phi1, model

# ──────────────────────────────
#  FEVD / spillover
# ──────────────────────────────
def gvd(A_list, Sigma):
    H = len(A_list)
    N = Sigma.shape[0]
    inv_diag = np.diag(1/np.diag(Sigma))
    theta = np.zeros((N, N))
    for i in range(N):
        ei = np.eye(1, N, i).ravel()
        denom = sum(ei @ A @ Sigma @ A.T @ ei for A in A_list)
        for j in range(N):
            ej = np.eye(1, N, j).ravel()
            numer = sum((ei @ A @ Sigma @ ej) ** 2 for A in A_list)
            theta[i, j] = inv_diag[j, j] * numer / denom
    theta_norm = theta / theta.sum(axis=1, keepdims=True)
    return theta, theta_norm

def spillover_metrics(theta_norm):
    N = theta_norm.shape[0]
    tsi = 100 * (theta_norm.sum() - np.trace(theta_norm)) / N
    spill_mat = theta_norm * 100
    to_   = spill_mat.sum(axis=0) - np.diag(spill_mat)
    from_ = spill_mat.sum(axis=1) - np.diag(spill_mat)
    net_  = to_ - from_
    return tsi, to_, from_, net_



__all__ = [
    "SEED", "LASSO_OPTS", "FAST_LASSO_OPTS",
    "create_mhar_lags", "fit_mhar_lasso",
    "gvd", "spillover_metrics"
]