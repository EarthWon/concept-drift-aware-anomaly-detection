from scipy.spatial.distance import mahalanobis
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Metrics
# -------------------------
def point_adjust_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    adjusted = y_pred.copy()

    in_anom = False
    start = 0
    for i, v in enumerate(y_true):
        if v == 1 and not in_anom:
            in_anom = True
            start = i
        elif v == 0 and in_anom:
            if y_pred[start:i].any():
                adjusted[start:i] = 1
            in_anom = False

    if in_anom and y_pred[start:].any():
        adjusted[start:] = 1

    return f1_score(y_true, adjusted, zero_division=0)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "point_adjust_f1": float(point_adjust_f1(y_true, y_pred)),
    }


def better_than(a, b):
    if b is None:
        return True
    for k in ["f1", "precision"]:
        if a[k] > b[k]:
            return True
        if a[k] < b[k]:
            return False
    return False


# -------------------------
# Features / Thresholds
# -------------------------
def extract_feature(win, clip_percentile=90):
    q_high = np.percentile(win, clip_percentile, axis=0)
    # win = np.clip(win, a_min=None, a_max=q_high)
    q_low = np.percentile(win, 100 - clip_percentile, axis=0)
    win = np.clip(win, q_low, q_high)

    mean_f = win.mean(axis=0)
    var_f = win.var(axis=0)
    abs_f = np.abs(win).mean(axis=0)
    cov_mean = float(np.mean(np.cov(win.T)))
    return np.concatenate([mean_f, var_f, abs_f, [cov_mean]])

def build_cd_baseline(bfeats, r=0.0):
    mu = bfeats.mean(axis=0)
    cov = np.cov(bfeats.T)
    cov_inv = np.linalg.pinv(cov)
    D_base = np.array([mahalanobis(f, mu, cov_inv) for f in bfeats])
    cd_th = float(np.quantile(D_base, 0.99) * (1.0 + r))
    return mu, cov_inv, cd_th

def build_ad_thresholds(base_err, k_low, k_high):
    loc = np.median(base_err)
    scale = np.std(base_err, ddof=1) + 1e-12
    return loc - k_low * scale, loc + k_high * scale

def win_slice(i, window):
    s = i * window
    e = (i + 1) * window
    return s, e


# -------------------------
# Precompute
# -------------------------
def precompute_for_tuning(df_raw, actual_cols, pred_cols, window, n_baseline, clip_percentile=95):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["error_score"] = np.sqrt((df[actual_cols].values - df[pred_cols].values) ** 2).mean(axis=1)

    N = len(df)
    N_trim = (N // window) * window
    df = df.iloc[:N_trim].copy().reset_index(drop=True)

    N = len(df)
    T = N // window

    X_act = df[actual_cols].to_numpy()
    err = df["error_score"].to_numpy()

    feats = np.stack(
        [extract_feature(X_act[slice(*win_slice(i, window))], clip_percentile=clip_percentile) for i in range(T)],
        axis=0,
    )
    return df, err, feats, T


# -------------------------
# Pipeline (CD+AD)
# -------------------------
def run_pipeline_with_params(
    df, err, feats, T,
    window=64,
    n_baseline=50,
    consec_k=10,
    r=3.0,
    k_high=10.0,
    k_low=2.0,
    ad_ratio_th=0.01,
    abnomal=None,
    ad_consec_n_default=2,
    ad_consec_map=None,
    cd_pad_pre=0,
    cd_pad_post=0,
    skip_ad_in_cd=True,
):
    if ad_consec_map is None:
        ad_consec_map = {}
    ad_consec_n = int(ad_consec_map.get(abnomal, ad_consec_n_default))

    N = len(df)
    cd_flag_win = np.zeros(T, dtype=int)
    ad_flag_win = np.zeros(T, dtype=int)
    ad_flag_pt = np.zeros(N, dtype=int)

    baseline_segments, cd_segments = [], []
    cd_score_win = np.full(T, np.nan, float)
    cd_th_win = np.full(T, np.nan, float)

    tau_low_pt = np.full(N, np.nan, float)
    tau_high_pt = np.full(N, np.nan, float)

    t = df["timestamp"].to_numpy()
    t_win = np.array([t[i * window] for i in range(T)])

    # init baseline
    base_start, base_end = 0, n_baseline
    baseline_segments.append((base_start, base_end))

    mu, cov_inv, cd_th = build_cd_baseline(feats[base_start:base_end], r=r)
    base_err = err[base_start * window: base_end * window]
    tau_low, tau_high = build_ad_thresholds(base_err, k_low, k_high)

    baseline_ready = True
    building_new_baseline = False
    new_base_start_idx = None
    baseline_buffer = []
    consec_cnt_cd = 0

    # record baseline logs
    for i in range(base_start, base_end):
        D_i = mahalanobis(feats[i], mu, cov_inv)
        cd_score_win[i] = float(D_i)
        cd_th_win[i] = float(cd_th)
        s, e = win_slice(i, window)
        tau_low_pt[s:e] = tau_low
        tau_high_pt[s:e] = tau_high

    # AD consecutive state (window-level)
    ad_cnt_win = 0
    ad_run_start_win = None

    for i in range(n_baseline, T):
        # ---- build new baseline mode
        if building_new_baseline:
            if (new_base_start_idx is not None) and (i < new_base_start_idx):
                s, e = win_slice(i, window)
                tau_low_pt[s:e] = tau_low
                tau_high_pt[s:e] = tau_high
                continue

            baseline_buffer.append(feats[i])
            s, e = win_slice(i, window)
            tau_low_pt[s:e] = tau_low
            tau_high_pt[s:e] = tau_high

            if len(baseline_buffer) >= n_baseline:
                bfeats = np.array(baseline_buffer[:n_baseline])
                mu, cov_inv, cd_th = build_cd_baseline(bfeats, r=r)

                base_w_start = new_base_start_idx
                base_w_end = new_base_start_idx + n_baseline
                baseline_segments.append((base_w_start, base_w_end))

                base_err = err[base_w_start * window: base_w_end * window]
                tau_low, tau_high = build_ad_thresholds(base_err, k_low, k_high)

                for j in range(base_w_start, min(base_w_end, T)):
                    D_j = mahalanobis(feats[j], mu, cov_inv)
                    cd_score_win[j] = float(D_j)
                    cd_th_win[j] = float(cd_th)
                    sj, ej = win_slice(j, window)
                    tau_low_pt[sj:ej] = tau_low
                    tau_high_pt[sj:ej] = tau_high

                baseline_ready = True
                building_new_baseline = False
                baseline_buffer = []
                consec_cnt_cd = 0
                ad_cnt_win = 0
                ad_run_start_win = None
            continue

        if baseline_ready:
            D_i = mahalanobis(feats[i], mu, cov_inv)
            cd_score_win[i] = float(D_i)
            cd_th_win[i] = float(cd_th)

            consec_cnt_cd = consec_cnt_cd + 1 if D_i > cd_th else 0

            if consec_cnt_cd >= consec_k:
                start_cd = i - consec_k + 1
                end_cd = i + 1
                start_ext = max(0, start_cd - int(cd_pad_pre))
                end_ext = min(T, end_cd + int(cd_pad_post))

                cd_segments.append((start_ext, end_ext))
                cd_flag_win[start_ext:end_ext] = 1

                baseline_ready = False
                building_new_baseline = True
                new_base_start_idx = end_ext
                baseline_buffer = []
                consec_cnt_cd = 0
                ad_cnt_win = 0
                ad_run_start_win = None

                s, e = win_slice(i, window)
                tau_low_pt[s:e] = tau_low
                tau_high_pt[s:e] = tau_high
                continue

            if skip_ad_in_cd and (cd_flag_win[i] == 1):
                s, e = win_slice(i, window)
                tau_low_pt[s:e] = tau_low
                tau_high_pt[s:e] = tau_high
                ad_cnt_win = 0
                ad_run_start_win = None
                continue

            if D_i > cd_th:
                s, e = win_slice(i, window)
                tau_low_pt[s:e] = tau_low
                tau_high_pt[s:e] = tau_high
                ad_cnt_win = 0
                ad_run_start_win = None
                continue

            # ---- AD
            s, e = win_slice(i, window)
            tau_low_pt[s:e] = tau_low
            tau_high_pt[s:e] = tau_high

            w_err = err[s:e]
            pt_anom = (w_err > tau_high) | (w_err < tau_low)
            ratio = float(pt_anom.mean())

            if abnomal == "hitting":
                ad_flag_pt[s:e] = pt_anom.astype(int)
            else:
                is_ad_win = (ratio >= ad_ratio_th)
                if is_ad_win:
                    if ad_cnt_win == 0:
                        ad_run_start_win = i
                    ad_cnt_win += 1
                    if ad_cnt_win >= ad_consec_n:
                        ad_flag_win[ad_run_start_win:i + 1] = 1
                else:
                    ad_cnt_win = 0
                    ad_run_start_win = None

    # finalize
    df_out = df.copy()
    df_out["cd_flag"] = np.repeat(cd_flag_win, window)[:N]

    if abnomal == "hitting":
        if skip_ad_in_cd:
            cd_mask_pt = df_out["cd_flag"].astype(bool).to_numpy()
            ad_flag_pt[cd_mask_pt] = 0
        df_out["ad_flag"] = ad_flag_pt
    else:
        if skip_ad_in_cd:
            ad_flag_win[cd_flag_win.astype(bool)] = 0
        df_out["ad_flag"] = np.repeat(ad_flag_win, window)[:N]

    df_win_out = pd.DataFrame({
        "timestamp_start": t_win,
        "cd_score": cd_score_win,
        "cd_th": cd_th_win,
        "cd_flag": cd_flag_win.astype(int),
    })

    return df_out, df_win_out, baseline_segments, cd_segments, tau_low_pt, tau_high_pt


# -------------------------
# Visualization (single figure)
# -------------------------
def visualize_cd_ad(
    df, df_win, baseline_segments, tau_low_pt, tau_high_pt,
    window=64, title_prefix="", save=None, *, titled=False,
):
    def shade(ax, x, mask, alpha, label=None, color=None):
        y0, y1 = ax.get_ylim()
        ax.fill_between(x, y0, y1, where=mask, alpha=alpha, label=label, color=color)
        ax.set_ylim(y0, y1)

    t = df["timestamp"].to_numpy()
    ad_mask = df["ad_flag"].astype(bool).to_numpy()

    sig_act = df["accel_x"].to_numpy()
    sig_pred = df["accel_x_pred"].to_numpy()
    err_sig = df["error_score"].to_numpy()

    y_min = min(sig_act.min(), sig_pred.min())
    y_max = max(sig_act.max(), sig_pred.max())
    m = 0.05 * (y_max - y_min + 1e-12)
    y_min, y_max = y_min - m, y_max + m

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=False)

    ax = axes[0]
    ax.plot(t, sig_act, lw=1, label="accel_x (actual)")
    ax.set_ylim(y_min, y_max)
    if "label" in df.columns:
        gt = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(bool).to_numpy()
        shade(ax, t, gt, 0.3, "GT", "black")
    ax.set_title(f"1) Actual + GT")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(t, sig_pred, lw=1, label="accel_x (pred)")
    ax.set_ylim(y_min, y_max)
    shade(ax, t, ad_mask, 0.25, "AD (pred)", "red")
    ax.set_title(f"2) Prediction + AD")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(t, err_sig, lw=1, label="error_score")
    ax.plot(t, tau_high_pt, "--", lw=1.2, label="tau_high")
    ax.plot(t, tau_low_pt, "--", lw=1.2, label="tau_low")
    ax.relim(); ax.autoscale_view()
    shade(ax, t, ad_mask, 0.25, "AD (pred)", "red")
    ax.set_title(f"3) Error score + dynamic thresholds")
    ax.legend(loc="upper right")

    ax = axes[3]
    t_win = df_win["timestamp_start"].to_numpy()
    cd_score = df_win["cd_score"].to_numpy()
    cd_th = df_win["cd_th"].to_numpy()
    cd_mask = df_win["cd_flag"].astype(bool).to_numpy()

    ax.plot(t_win, cd_score, lw=1, label="Mahalanobis D")
    ax.plot(t_win, cd_th, "--", lw=1, label="CD threshold")

    base_mask = np.zeros(len(df_win), dtype=bool)
    for ws, we in baseline_segments:
        base_mask[ws:min(we, len(base_mask))] = True

    shade(ax, t_win, base_mask, 0.15, "Baseline")
    shade(ax, t_win, cd_mask, 0.25, "CD flagged")
    ax.set_title(f"4) Concept Drift")
    ax.legend(loc="upper right")

    for ax in axes:
        ax.set_xlabel("timestamp")

    if titled:
        fig.suptitle(title_prefix, fontsize=9)
    plt.tight_layout()
    if save:
        d = os.path.dirname(save)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight" if titled else None)
        plt.close(fig)
    else:
        plt.show()
