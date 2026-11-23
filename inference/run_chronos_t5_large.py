from chronos import ChronosPipeline
from tqdm import tqdm

import numpy as np
import torch
import os

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_chronos_t5_large(
    df,
    ckpt_path=None,
    abnormal = None,
    window=512,
    div=8
    ):
    
    if ckpt_path is None:
        model = ChronosPipeline.from_pretrained("amazon/chronos-t5-large", device_map="cuda", torch_dtype=torch.bfloat16,)
    else:
        model = ChronosPipeline.from_pretrained(ckpt_path, device_map="cuda", torch_dtype=torch.bfloat16,)
        print(f"Loaded finetuned model from {ckpt_path}")
    
    features = [c for c in df.columns if c not in ("timestamp", "label")]
    
    C = int(window)
    H = int(C / div)
    s = H  # non-overlap

    X = [df[c].to_numpy() for c in features]
    N = len(X[0])
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
        
        inputs = [torch.tensor(x[t-C:t], dtype=torch.float32) for x in X]
        quantile = model.predict(
            inputs=inputs,
            prediction_length=pred_len,
            )
        
        point = torch.quantile(quantile, q=0.5, dim=1).cpu().numpy()
        q = quantile.cpu().numpy()  # Get the quantiles from the model output

        q10 = np.quantile(q, 0.1, axis=1)  # (F,)
        q50 = np.quantile(q, 0.5, axis=1)  # (F,)
        q90 = np.quantile(q, 0.9, axis=1)  # (F,)

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
