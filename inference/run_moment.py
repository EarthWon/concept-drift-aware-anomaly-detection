from tqdm import tqdm

import numpy as np
import torch
from momentfm import MOMENTPipeline
import warnings
warnings.filterwarnings("ignore")


def run_moment(
    df,
    ckpt_path=None,
    abnormal=None,
    window=512,
    div=8
    ):
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs = {
            'task_name': 'forecasting',
            'forecast_horizon': 64,
            'freeze_encoder': True,
            'freeze_embedder': True,
    }
    )
    model.init()

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(state, strict=True)
        # model.head.load_state_dict(head_state, strict=True) # 모델 헤드만 finetuning 했을 경우
        print(f"Loaded finetuned model from {ckpt_path}")

    model = model.to("cuda").float().eval()

    features = [c for c in df.columns if c not in ("timestamp", "label")]
    C = int(window)
    H = int(C / div)
    s = H 
    L = C + H

    X = df[features].to_numpy(dtype=np.float32)
    N, F = X.shape
    out_len = N - C
    
    print(f"Start {abnormal}")
    print(f"Input Context length:    {C}")
    print(f"Output Horizon:          {H}")
    print(f"Total Data Length (N):   {N}")
    print(f"Stride:                  {s}")

    chunks = []

    for t in tqdm(range(C, N, s)):
        rem = out_len - (t - C)
        if rem <= 0:
            break
        pred_len = H if rem >= H else rem

        seg = X[t - C : t - C + (C + pred_len)]  # (L, F)
        seg_tc = torch.from_numpy(seg.T).unsqueeze(0).to("cuda")  # (1, F, L)

        input_mask = torch.zeros((1, C + pred_len), dtype=torch.long, device="cuda")
        input_mask[:, :C] = 1

        with torch.no_grad():
            out = model(x_enc=seg_tc[:, :, :C], input_mask=input_mask[:, :C])
            pred = out.forecast[0, :, :pred_len].detach().cpu().numpy()  # (F, H)

        chunks.append(pred)

    point_full = np.concatenate(chunks, axis=1)  # (F, num_steps*H)

    valid_len = point_full.shape[1]
    valid_df = df.iloc[C:].copy()  # 예측 길이에 맞춰 자름

    for i, col in enumerate(features):
        valid_df[f"{col}_pred"] = point_full[i]
        
    return valid_df
