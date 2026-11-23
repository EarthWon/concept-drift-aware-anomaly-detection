from tqdm import tqdm
import numpy as np
import torch
import timesfm

def run_timesfm(
    df,
    ckpt_path=None,
    abnormal = None,
    window=512,
    div=8
    ):
    torch.set_float32_matmul_precision("high")
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model load + compile
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model._device = torch_device
    model.model.to(torch_device)

    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=torch_device)
        model.model.load_state_dict(state, strict=False)
        print(f"Loaded finetuned model from {ckpt_path}")

    model.model.eval()

    features = [c for c in df.columns if c not in ("timestamp", "label")]
    
    C = int(window)
    H = int(C / div)
    s = H  # non-overlap

    X = [df[c].to_numpy() for c in features]
    N = len(X[0])
    F = len(X)
    out_len = N - C
    
    print(f"Start {abnormal}")
    print(f"Input Context length:    {C}")
    print(f"Output Horizon:          {H}")
    print(f"Total Data Length (N):   {N}")
    print(f"Stride:                  {s}")

    point_chunks, q10_chunks, q50_chunks, q90_chunks = [], [], [], []

    for t in tqdm(range(C, N, s)):
        rem = out_len - (t - C)
        if rem <= 0:
            break
        pred_len = H if rem >= H else rem

        inputs = [x[t - C:t] for x in X]  # list of (C,)

        point, quantile = model.forecast(horizon=pred_len, inputs=inputs)
        # point: (F, pred_len), quantile: (F, pred_len, Q)

        point = np.asarray(point)
        q = np.asarray(quantile)

        q10 = q[:, :, 0]
        q50 = q[:, :, 5]
        q90 = q[:, :, 9]

        point_chunks.append(point)
        q10_chunks.append(q10)
        q50_chunks.append(q50)
        q90_chunks.append(q90)

        if pred_len < H:
            break

    point_full = np.concatenate(point_chunks, axis=1)  # (F, out_len)
    q10_full   = np.concatenate(q10_chunks,   axis=1)
    q50_full   = np.concatenate(q50_chunks,   axis=1)
    q90_full   = np.concatenate(q90_chunks,   axis=1)

    valid_df = df.iloc[C:].copy()  # 처음 input context 제외
    for i, col in enumerate(features):
        valid_df[f"{col}_pred"]     = point_full[i]
        valid_df[f"{col}_q10_pred"] = q10_full[i]
        valid_df[f"{col}_q50_pred"] = q50_full[i]
        valid_df[f"{col}_q90_pred"] = q90_full[i]
        
    return valid_df
