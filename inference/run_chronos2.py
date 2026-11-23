from chronos import Chronos2Pipeline, BaseChronosPipeline
from tqdm import tqdm
import numpy as np

def run_chronos2(
    df,
    ckpt_path=None,
    abnormal = None,
    window=512,
    div=8
    ):
    if ckpt_path is None:
        pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    else:
        pipeline = Chronos2Pipeline.from_pretrained(ckpt_path, device_map="cuda")
        print(f"Loaded finetuned model from {ckpt_path}")
        
    # pipeline = (BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    #             if ckpt_path is None else
    #             Chronos2Pipeline.from_pretrained(ckpt_path, device_map="cuda"))

    features = [c for c in df.columns if c not in ("timestamp", "label")]

    C = int(window)
    H = int(C / div)
    s = H 

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

        inputs = np.array([[x[t - C:t] for x in X]])
        qs, pt = pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=pred_len,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        pt = pt[0].cpu().numpy()          # (F, pred_len)
        qs = qs[0].cpu().numpy()          # (F, pred_len, 3)

        point_chunks.append(pt)
        q10_chunks.append(qs[:, :, 0])
        q50_chunks.append(qs[:, :, 1])
        q90_chunks.append(qs[:, :, 2])

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
