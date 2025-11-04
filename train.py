# compare_models_pipeline.py
# Compares MultiTaskElasticNet, Ridge, LinearRegression, DecisionTreeRegressor, and RRR
# Requirements: numpy, pandas, scikit-learn, scipy
# Usage: python compare_models_pipeline.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression, MultiTaskElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ================= User settings =================
csv_path = 'freesurfer_wide_with_deltas.csv'   # <- update if needed
subject_col = 'subj'                           # <- subject id column in wide CSV
outer_splits = 5
inner_splits = 3
ridge_alphas = [0.01, 0.1, 1.0, 10.0]
enet_alphas = [0.01, 0.1, 1.0]
enet_l1 = [0.2, 0.5, 0.8]
dtr_max_depths = [2, 4, 6, 8, None]           # None => no max depth
rrr_ranks  = [1, 2, 3, 5, 10, 20]
random_state = 42
output_prefix = 'compare_models'
# =================================================

def safe_pearsonr(a, b):
    try:
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return pearsonr(a, b)[0]
    except Exception:
        return np.nan

# ---------------- Load & detect columns ----------------
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at {csv_path}")

df = pd.read_csv(csv_path)
print("Loaded", csv_path, "shape:", df.shape)

if subject_col not in df.columns:
    candidates = [c for c in df.columns if c.lower() in ('subj','subject','id')]
    if len(candidates) == 1:
        subject_col = candidates[0]
        print("Using detected subject column:", subject_col)
    else:
        raise ValueError(f"Subject column '{subject_col}' not found. Candidates: {candidates}")

df[subject_col] = df[subject_col].astype(str)
df = df.set_index(subject_col)

cols = df.columns.tolist()
baseline_cols = [c for c in cols if c.endswith('_baseline')]
pct6_cols  = [c for c in cols if c.endswith('_pctchg_6m')]
pct12_cols = [c for c in cols if c.endswith('_pctchg_12m')]

if len(baseline_cols) == 0:
    raise ValueError("No baseline columns detected. Ensure baseline columns end with '_baseline'.")
if len(pct6_cols) + len(pct12_cols) == 0:
    raise ValueError("No pct-change target columns detected. Ensure they end with '_pctchg_6m' or '_pctchg_12m'.")

target_cols = pct6_cols + pct12_cols
print(f"Using {len(baseline_cols)} baseline features and {len(target_cols)} targets.")

X_df = df[baseline_cols].copy()
Y_df = df[target_cols].copy()

mask = X_df.notna().all(axis=1) & Y_df.notna().all(axis=1)
n_before = len(df); n_after = mask.sum()
X_df = X_df.loc[mask]; Y_df = Y_df.loc[mask]
print(f"Dropped {n_before - n_after} subjects due to NaNs. Remaining: {n_after}")
if n_after < 10:
    print("Warning: small sample size after cleaning; results may be unstable")

X = X_df.values
Y = Y_df.values
subjects = X_df.index.to_numpy()
Y_names = Y_df.columns.tolist()

print("X shape:", X.shape, "Y shape:", Y.shape)

# ---------------- Cross-val scaffolding ----------------
outer_cv = GroupKFold(n_splits=outer_splits)

# Storage
model_names = ['Ridge','ElasticNet','LinearOLS','DecisionTree','RRR']
fold_results = {m: [] for m in model_names}
all_preds = {m: [] for m in model_names}
all_truth = {m: [] for m in model_names}

fold_id = 0
for train_idx, test_idx in outer_cv.split(X, Y, groups=subjects):
    fold_id += 1
    print(f"\n=== Outer fold {fold_id}/{outer_splits} ===")
    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]
    subj_tr, subj_te = subjects[train_idx], subjects[test_idx]

    # scale X/Y using training data only
    x_scaler = StandardScaler().fit(X_tr)
    X_tr_s = x_scaler.transform(X_tr)
    X_te_s = x_scaler.transform(X_te)
    y_scaler = StandardScaler().fit(Y_tr)
    Y_tr_s = y_scaler.transform(Y_tr)

    # ---------- Ridge (grid-search) ----------
    from sklearn.model_selection import GridSearchCV
    ridge = Ridge(random_state=random_state, max_iter=5000)
    inner_cv = GroupKFold(n_splits=inner_splits)
    gs_ridge = GridSearchCV(ridge, {'alpha': ridge_alphas}, cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1, refit=True)
    gs_ridge.fit(X_tr_s, Y_tr_s)
    best_ridge = gs_ridge.best_estimator_
    best_ridge.fit(X_tr_s, Y_tr_s)
    Y_pred_ridge_s = best_ridge.predict(X_te_s)
    Y_pred_ridge = y_scaler.inverse_transform(Y_pred_ridge_s)
    all_preds['Ridge'].append(pd.DataFrame(Y_pred_ridge, index=subj_te, columns=Y_names))
    all_truth['Ridge'].append(pd.DataFrame(Y_te, index=subj_te, columns=Y_names))
    # metrics
    maes, r2s, prs = [], [], []
    for j in range(Y.shape[1]):
        y_t = Y_te[:, j]; y_p = Y_pred_ridge[:, j]
        maes.append(mean_absolute_error(y_t, y_p))
        r2s.append(r2_score(y_t, y_p) if np.var(y_t) > 0 else np.nan)
        prs.append(safe_pearsonr(y_t, y_p))
    df_fold = pd.DataFrame({'roi':Y_names, 'mae':maes, 'r2':r2s, 'pearson_r':prs})
    df_fold['fold'] = fold_id
    fold_results['Ridge'].append(df_fold)
    print(f"Ridge fold mean MAE: {np.nanmean(maes):.4f}, mean Pearson r: {np.nanmean(prs):.4f}")

    # ---------- ElasticNet (MultiTask) ----------
    gs_enet = GridSearchCV(MultiTaskElasticNet(max_iter=5000, random_state=random_state),
                           {'alpha': enet_alphas, 'l1_ratio': enet_l1},
                           cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1, refit=True)
    gs_enet.fit(X_tr_s, Y_tr_s)
    best_enet = gs_enet.best_estimator_
    best_enet.fit(X_tr_s, Y_tr_s)
    Y_pred_enet_s = best_enet.predict(X_te_s)
    Y_pred_enet = y_scaler.inverse_transform(Y_pred_enet_s)
    all_preds['ElasticNet'].append(pd.DataFrame(Y_pred_enet, index=subj_te, columns=Y_names))
    all_truth['ElasticNet'].append(pd.DataFrame(Y_te, index=subj_te, columns=Y_names))
    maes, r2s, prs = [], [], []
    for j in range(Y.shape[1]):
        y_t = Y_te[:, j]; y_p = Y_pred_enet[:, j]
        maes.append(mean_absolute_error(y_t, y_p))
        r2s.append(r2_score(y_t, y_p) if np.var(y_t) > 0 else np.nan)
        prs.append(safe_pearsonr(y_t, y_p))
    df_fold = pd.DataFrame({'roi':Y_names, 'mae':maes, 'r2':r2s, 'pearson_r':prs})
    df_fold['fold'] = fold_id
    fold_results['ElasticNet'].append(df_fold)
    print(f"ElasticNet fold mean MAE: {np.nanmean(maes):.4f}, mean Pearson r: {np.nanmean(prs):.4f}")

    # ---------- Linear OLS (no tuning) ----------
    ols = LinearRegression(fit_intercept=False)  # fit_intercept False because we scaled
    ols.fit(X_tr_s, Y_tr_s)
    Y_pred_ols_s = ols.predict(X_te_s)
    Y_pred_ols = y_scaler.inverse_transform(Y_pred_ols_s)
    all_preds['LinearOLS'].append(pd.DataFrame(Y_pred_ols, index=subj_te, columns=Y_names))
    all_truth['LinearOLS'].append(pd.DataFrame(Y_te, index=subj_te, columns=Y_names))
    maes, r2s, prs = [], [], []
    for j in range(Y.shape[1]):
        y_t = Y_te[:, j]; y_p = Y_pred_ols[:, j]
        maes.append(mean_absolute_error(y_t, y_p))
        r2s.append(r2_score(y_t, y_p) if np.var(y_t) > 0 else np.nan)
        prs.append(safe_pearsonr(y_t, y_p))
    df_fold = pd.DataFrame({'roi':Y_names, 'mae':maes, 'r2':r2s, 'pearson_r':prs})
    df_fold['fold'] = fold_id
    fold_results['LinearOLS'].append(df_fold)
    print(f"LinearOLS fold mean MAE: {np.nanmean(maes):.4f}, mean Pearson r: {np.nanmean(prs):.4f}")

    # ---------- Decision Tree Regressor (tune max_depth) ----------
    gs_tree = GridSearchCV(DecisionTreeRegressor(random_state=random_state), {'max_depth': dtr_max_depths},
                           cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1, refit=True)
    # DecisionTreeRegressor supports multi-output directly
    gs_tree.fit(X_tr_s, Y_tr_s)
    best_tree = gs_tree.best_estimator_
    best_tree.fit(X_tr_s, Y_tr_s)
    Y_pred_tree_s = best_tree.predict(X_te_s)
    Y_pred_tree = y_scaler.inverse_transform(Y_pred_tree_s)
    all_preds['DecisionTree'].append(pd.DataFrame(Y_pred_tree, index=subj_te, columns=Y_names))
    all_truth['DecisionTree'].append(pd.DataFrame(Y_te, index=subj_te, columns=Y_names))
    maes, r2s, prs = [], [], []
    for j in range(Y.shape[1]):
        y_t = Y_te[:, j]; y_p = Y_pred_tree[:, j]
        maes.append(mean_absolute_error(y_t, y_p))
        r2s.append(r2_score(y_t, y_p) if np.var(y_t) > 0 else np.nan)
        prs.append(safe_pearsonr(y_t, y_p))
    df_fold = pd.DataFrame({'roi':Y_names, 'mae':maes, 'r2':r2s, 'pearson_r':prs})
    df_fold['fold'] = fold_id
    fold_results['DecisionTree'].append(df_fold)
    print(f"DecisionTree fold mean MAE: {np.nanmean(maes):.4f}, mean Pearson r: {np.nanmean(prs):.4f}")

    # ---------- Reduced-Rank Regression (RRR) ----------
    inner_cv_gen = GroupKFold(n_splits=inner_splits)
    best_rank = None
    best_score = -np.inf
    for rank in rrr_ranks:
        inner_scores = []
        for i_tr, i_val in inner_cv_gen.split(X_tr_s, Y_tr_s, groups=subj_tr):
            X_i_tr, X_i_val = X_tr_s[i_tr], X_tr_s[i_val]
            Y_i_tr, Y_i_val = Y_tr_s[i_tr], Y_tr_s[i_val]
            ols_inner = LinearRegression(fit_intercept=False).fit(X_i_tr, Y_i_tr)
            B_hat = ols_inner.coef_.T  # p x q
            U, S, Vt = np.linalg.svd(B_hat, full_matrices=False)
            r_use = min(rank, U.shape[1])
            U_r = U[:, :r_use]; S_r = S[:r_use]; Vt_r = Vt[:r_use, :]
            B_r = (U_r * S_r) @ Vt_r
            Y_val_hat = X_i_val @ B_r
            inner_scores.append(-mean_absolute_error(Y_i_val, Y_val_hat))
        mean_inner = np.mean(inner_scores)
        if mean_inner > best_score:
            best_score = mean_inner
            best_rank = rank
    print("RRR best rank:", best_rank, "inner score:", best_score)

    # train final RRR on full training
    ols_full = LinearRegression(fit_intercept=False).fit(X_tr_s, Y_tr_s)
    B_hat_full = ols_full.coef_.T  # p x q
    U, S, Vt = np.linalg.svd(B_hat_full, full_matrices=False)
    r_use = min(best_rank, U.shape[1])
    U_r = U[:, :r_use]; S_r = S[:r_use]; Vt_r = Vt[:r_use, :]
    B_r_full = (U_r * S_r) @ Vt_r
    Y_pred_rrr_s = X_te_s @ B_r_full
    Y_pred_rrr = y_scaler.inverse_transform(Y_pred_rrr_s)
    all_preds['RRR'].append(pd.DataFrame(Y_pred_rrr, index=subj_te, columns=Y_names))
    all_truth['RRR'].append(pd.DataFrame(Y_te, index=subj_te, columns=Y_names))
    maes, r2s, prs = [], [], []
    for j in range(Y.shape[1]):
        y_t = Y_te[:, j]; y_p = Y_pred_rrr[:, j]
        maes.append(mean_absolute_error(y_t, y_p))
        r2s.append(r2_score(y_t, y_p) if np.var(y_t) > 0 else np.nan)
        prs.append(safe_pearsonr(y_t, y_p))
    df_fold = pd.DataFrame({'roi':Y_names, 'mae':maes, 'r2':r2s, 'pearson_r':prs})
    df_fold['fold'] = fold_id
    fold_results['RRR'].append(df_fold)
    print(f"RRR fold mean MAE: {np.nanmean(maes):.4f}, mean Pearson r: {np.nanmean(prs):.4f}")

# ---------------- Aggregate and save results ----------------
for m in model_names:
    if len(fold_results[m])==0:
        continue
    res_df = pd.concat(fold_results[m], ignore_index=True)
    mean_df = res_df.groupby('roi').agg({'mae':'mean','r2':'mean','pearson_r':'mean'}).reset_index()
    res_df.to_csv(f'{output_prefix}_{m}_per_fold_metrics.csv', index=False)
    mean_df.to_csv(f'{output_prefix}_{m}_mean_metrics.csv', index=False)
    preds_concat = pd.concat(all_preds[m]).sort_index()
    truth_concat = pd.concat(all_truth[m]).sort_index()
    preds_concat.to_csv(f'{output_prefix}_{m}_all_preds.csv')
    truth_concat.to_csv(f'{output_prefix}_{m}_all_truth.csv')
    print(f"\nSaved {m} results. Top 5 ROIs by Pearson r:")
    print(mean_df.sort_values('pearson_r', ascending=False).head(5)[['roi','pearson_r','mae','r2']])

print("\nAll done. Files saved with prefix:", output_prefix)
