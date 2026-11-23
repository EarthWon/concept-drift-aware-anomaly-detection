from pathlib import Path
import os, sys
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
LAG_LLAMA_DIR = str(BUNDLE_DIR / "dependencies" / "lag-llama")
CKPT_PATH = os.path.join(LAG_LLAMA_DIR, "lag-llama.ckpt")
sys.path.append(LAG_LLAMA_DIR)

from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])
if not hasattr(StudentTOutput, "loss"):
    def loss(self, target, distr_args, loc=None, scale=None):
        distr = self.distribution(distr_args, loc=loc, scale=scale)
        return -distr.log_prob(target)
    StudentTOutput.loss = loss


from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator

from pandas.tseries.frequencies import to_offset
from types import ModuleType

from tqdm import tqdm

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_dummy_module(module_path: str):
    parts = module_path.split('.')
    current = ''
    parent = None
    for part in parts:
        current = current + '.' + part if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent:
                setattr(sys.modules[parent], part, module)
        parent = current
    return sys.modules[module_path]

gluonts_module = create_dummy_module('gluonts.torch.modules.loss')

class DistributionLoss:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs): 
        return 0.0
    def __getattr__(self, name): 
        return lambda *args, **kwargs: None

class NegativeLogLikelihood:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs): 
        return 0.0
    def __getattr__(self, name): 
        return lambda *args, **kwargs: None

gluonts_module.DistributionLoss = DistributionLoss
gluonts_module.NegativeLogLikelihood = NegativeLogLikelihood


def run_lagllama(
    df,
    ckpt_path=None,  # Fine-tuned 모델 디렉토리 (None이면 zero-shot)
    abnormal=None,
    window=512,
    div=8,
):
    device = torch.device("cuda:0")
    
    C = int(window)
    H = int(C / div)
    s = H  # non-overlap
    
    if ckpt_path is None:
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        lags_seq = estimator_args["lags_seq"]
        
        rope_scaling_arguments = {
            "type": "linear",
            "factor": (C + H) / estimator_args["context_length"],
        }

        estimator = LagLlamaEstimator(
            ckpt_path=CKPT_PATH,
            prediction_length=H,
            context_length=C,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            lags_seq=lags_seq,
            rope_scaling=rope_scaling_arguments,
            batch_size=3,
            num_parallel_samples=100,
            device=device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)
    else:
        predictor = PyTorchPredictor.deserialize(Path(ckpt_path), device=device)
        print(f"Loaded finetuned model from {ckpt_path}")
        

    features = [c for c in df.columns if c not in ("timestamp", "label")]

    # timestamp/label 보존
    ts_origin = df["timestamp"].copy()
    label_origin = df["label"].copy() if "label" in df.columns else None

    # Lag-Llama용: timestamp index + regular frequency로 재인덱싱
    X = df[["timestamp"] + features].copy()
    # X["timestamp"] = pd.to_datetime(X["timestamp"])
    X["timestamp"] = pd.to_datetime( # ds용
    X["timestamp"],
    format="ISO8601",
    utc=True,
    errors="coerce",
    )
    X = X.set_index("timestamp")
    X = X[features].apply(pd.to_numeric, errors="coerce").astype("float32")

    freq = pd.infer_freq(X.index)
    if freq is None:
        step = X.index.to_series().diff().dropna().median()
        freq = to_offset(step).freqstr

    new_index = pd.date_range(start=X.index.min(), periods=len(X), freq=freq)
    X = X.set_index(new_index)

    # label/timestamp도 동일 인덱스로 재설정
    ts_origin = ts_origin.set_axis(new_index)
    if label_origin is not None:
        label_origin = label_origin.set_axis(new_index)

    N, F = X.shape
    out_len = N - C

    print(f"Scenario:                {abnormal}")
    print(f"Context length (C):      {C}")
    print(f"Prediction length (H):   {H}")
    print(f"Total Data Length (N):   {N}")
    print(f"Valid output length:     {out_len}")
    print(f"Stride:                  {s}")

    # 예측 루프
    point_chunks = []

    for t in tqdm(range(C, N, s), desc="Predicting"):
        rem = out_len - (t - C)
        if rem <= 0:
            break
        pred_len = H if rem >= H else rem

        seg = X.iloc[t - C : t - C + (C + pred_len)]
        test_ds = PandasDataset(dict(seg), freq=freq)

        forecast_it, _ = make_evaluation_predictions(
            dataset=test_ds,
            predictor=predictor,
            num_samples=100,
        )
        forecasts = list(forecast_it)

        point = np.stack([f.mean[:pred_len] for f in forecasts], axis=0).astype(np.float32)
        point_chunks.append(point)

        if pred_len < H:
            break

    point_full = np.concatenate(point_chunks, axis=1)

    valid_df = X.iloc[C:].copy()
    for i, col in enumerate(features):
        valid_df[f"{col}_pred"] = point_full[i]

    valid_df["timestamp_origin"] = ts_origin.iloc[C:].values
    if label_origin is not None:
        valid_df["label"] = label_origin.iloc[C:].values

    valid_df.reset_index(names="timestamp", inplace=True)
    return valid_df
