import argparse
from importlib import import_module
from pathlib import Path

import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BUNDLE_DIR / "data" / "3st_collection"
OUTPUT_DIR = BUNDLE_DIR / "results" / "inference_df"
WEIGHTS_DIR = BUNDLE_DIR / "finetuning" / "weights"

CHECKPOINTS = {
    "chronos2": "chronos2/chronos2_full_finetuned",
    "chronos_t5_large": "chronos-t5-large/chronos-t5-large_full_finetuned",
    "timesfm": "timesfm/timesfm_full_finetuned/timesfm_full_finetuned.safetensors",
    "moment": "moment/moment_full_finetuned/moment_finetuned.pth",
    "lagllama": "lagllama/lagllama_full_finetuned",
}
SCENARIOS = ("instop", "grabbing", "hitting")
COLUMNS = (
    "timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z", "label",
)


def main():
    parser = argparse.ArgumentParser(description="Save model predictions for existing IMU scenarios.")
    parser.add_argument("--model", required=True, choices=CHECKPOINTS)
    parser.add_argument("--mode", choices=("zeroshot", "full_finetuned"), default="zeroshot")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--checkpoint", type=Path, help="Override the full-finetuned checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if args.mode == "full_finetuned":
        checkpoint = checkpoint or WEIGHTS_DIR / CHECKPOINTS[args.model]
        if not checkpoint.exists():
            parser.error(f"checkpoint does not exist: {checkpoint}")
    elif checkpoint is not None:
        parser.error("--checkpoint requires --mode full_finetuned")

    module = import_module(f"run_{args.model}")
    predict = getattr(module, f"run_{args.model}")
    for scenario in args.scenarios:
        source = args.data_dir / f"labeled_abnor_imu_{scenario}.csv"
        df = pd.read_csv(source, usecols=COLUMNS)
        predictions = predict(df=df, abnormal=scenario, ckpt_path=checkpoint)
        destination = (
            args.out_dir / args.mode / f"{scenario}_all_scenario"
            / f"{args.model}_3st_output.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(destination, index=False)
        print(f"Saved to {destination}")


if __name__ == "__main__":
    main()
