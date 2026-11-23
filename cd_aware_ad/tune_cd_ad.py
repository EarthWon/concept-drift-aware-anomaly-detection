import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from concept_drift_anomaly_detection import (
    better_than,
    compute_metrics,
    precompute_for_tuning,
    run_pipeline_with_params,
    visualize_cd_ad,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODES = ("zeroshot", "full_finetuned")
MODELS = ("chronos2", "timesfm", "moment", "lagllama", "chronos_t5_large")


def tune(
    data_root=RESULTS_DIR / "inference_df",
    out_csv_dir=RESULTS_DIR / "concept_drift",
    out_fig_root=RESULTS_DIR / "figures",
    *,
    modes=MODES,
    models=MODELS,
    cd_pad_pre=20,
    r_list=(7,),
):
    data_root = Path(data_root)
    actual_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
    pred_cols = [c + "_pred" for c in actual_cols]

    # fixed hypers
    window = 64
    n_baseline = 50
    consec_k = 10

    # abnormal-wise AD consecutive requirement
    ad_consec_map = {
        "instop": 5,
        "grabbing": 3,
        "hitting": 1
        }

    # CD extension (window units)
    cd_pad_post = 0
    skip_ad_in_cd = True

    # tuning ranges
    kh_vals = np.arange(7, 15.1, 0.1)
    kl_vals = np.arange(0.1, 7.1, 0.1)
    ratio_list = np.arange(0.1, 0.9, 0.1)

    for mode in modes:
        total_best_params = []

        for model in models:
            for abnor in ad_consec_map:
                p = (
                    data_root / mode / f"{abnor}_all_scenario"
                    / f"{model}_3st_output_with_cd_label.csv"
                )
                if not p.is_file():
                    p = p.with_name(f"{model}_3st_output.csv")
                df_raw = pd.read_csv(p)

                df_base, err, feats, T = precompute_for_tuning(
                    df_raw, actual_cols, pred_cols, window, n_baseline, clip_percentile=95
                )
                y_true = df_base["label"].to_numpy().astype(int)

                best_metric = None
                best_params = None  # (kh, kl, ratio, r)

                ratio_iter = ratio_list if abnor != "hitting" else [float(ratio_list[0])]
                ad_consec_n_used = int(ad_consec_map.get(abnor, 2))

                for kh in tqdm(kh_vals, desc=f"{mode}-{model}-{abnor}", leave=False):
                    for kl in kl_vals:
                        for ratio_th in ratio_iter:
                            for r in r_list:
                                df_out, *_ = run_pipeline_with_params(
                                    df=df_base,
                                    err=err,
                                    feats=feats,
                                    T=T,
                                    window=window,
                                    n_baseline=n_baseline,
                                    consec_k=consec_k,
                                    r=float(r),
                                    k_high=float(kh),
                                    k_low=float(kl),
                                    ad_ratio_th=float(ratio_th),
                                    abnomal=abnor,
                                    ad_consec_n_default=2,
                                    ad_consec_map=ad_consec_map,
                                    cd_pad_pre=cd_pad_pre,
                                    cd_pad_post=cd_pad_post,
                                    skip_ad_in_cd=skip_ad_in_cd,
                                )

                                y_pred = df_out["ad_flag"].to_numpy().astype(int)
                                m = compute_metrics(y_true, y_pred)

                                if better_than(m, best_metric):
                                    best_metric = m
                                    best_params = (float(kh), float(kl), float(ratio_th), float(r))

                kh_best, kl_best, ratio_best, r_best = best_params

                # best 재실행
                df_best, df_win_best, baseline_segments_best, _, tau_low_pt_best, tau_high_pt_best = run_pipeline_with_params(
                    df=df_base,
                    err=err,
                    feats=feats,
                    T=T,
                    window=window,
                    n_baseline=n_baseline,
                    consec_k=consec_k,
                    r=r_best,
                    k_high=kh_best,
                    k_low=kl_best,
                    ad_ratio_th=ratio_best,
                    abnomal=abnor,
                    ad_consec_n_default=2,
                    ad_consec_map=ad_consec_map,
                    cd_pad_pre=cd_pad_pre,
                    cd_pad_post=cd_pad_post,
                    skip_ad_in_cd=skip_ad_in_cd,
                )

                title_prefix = (
                    f"[{mode} | {model} | {abnor}] "
                    f"kh={kh_best:.2f}, kl={kl_best:.2f}, ratio={ratio_best:.2f}, r={r_best:.2f}, "
                    f"ad_consec_n={ad_consec_n_used}, cd_pad_pre={cd_pad_pre}, cd_pad_post={cd_pad_post}, skip_ad_in_cd={skip_ad_in_cd} | "
                )

                fig_path = (
                    f"{out_fig_root}/{mode}/"
                    f"{model}_{abnor}_ver2.png"
                )

                visualize_cd_ad(
                    df=df_best,
                    df_win=df_win_best,
                    baseline_segments=baseline_segments_best,
                    tau_low_pt=tau_low_pt_best,
                    tau_high_pt=tau_high_pt_best,
                    window=window,
                    title_prefix=title_prefix,
                    save=fig_path,
                )

                row = {
                    "mode": mode,
                    "model": model,
                    "abnomal": abnor,
                    "k_high": kh_best,
                    "k_low": kl_best,
                    "ad_ratio_th": ratio_best,
                    "ad_consec_n": ad_consec_n_used,
                    "cd_pad_pre": int(cd_pad_pre),
                    "cd_pad_post": int(cd_pad_post),
                    "skip_ad_in_cd": bool(skip_ad_in_cd),
                    "r": float(r_best),
                    **best_metric,
                }
                total_best_params.append(row)

                print(
                    f"[BEST] mode={mode} | model={model} | abnor={abnor} | "
                    f"k_high={kh_best:.2f}, k_low={kl_best:.2f}, ratio={ratio_best:.2f}, r={r_best:.2f} | "
                    f"F1={best_metric['f1']:.4f}, PA-F1={best_metric['point_adjust_f1']:.4f}, "
                    f"BAcc={best_metric['balanced_accuracy']:.4f}, "
                    f"accuracy={best_metric['accuracy']:.4f}, precision={best_metric['precision']:.4f}, recall={best_metric['recall']:.4f}"
                )

        os.makedirs(out_csv_dir, exist_ok=True)
        total_best_df = pd.DataFrame(total_best_params)
        out_csv = f"{out_csv_dir}/cd_ad_best_{mode}_hyperparam_tuning_results2.csv"
        total_best_df.to_csv(out_csv, index=False)
        print(f"[SAVED] {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Tune CD+AD on forecast CSVs.")
    parser.add_argument("--data-root", type=Path, default=RESULTS_DIR / "inference_df")
    parser.add_argument("--out-csv-dir", type=Path, default=RESULTS_DIR / "concept_drift")
    parser.add_argument("--out-fig-root", type=Path, default=RESULTS_DIR / "figures")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    tune(**vars(args))


if __name__ == "__main__":
    main()
