# nnUNet 5cls Test
#
# 3D 가중치: nnUNet_results/Dataset003_MBHSeg25/nnUNetTrainer__nnUNetPlans__3d_fullres
# 2D 가중치: nnUNet_results/Dataset003_MBHSeg25/nnUNetTrainer__nnUNetPlans__2d
#
# test 데이터
#   /home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet/nnUNet_test/Test1_MBHSeg25
#   /home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet/nnUNet_test/Test2_CTICH

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

# ---- nnUNet 환경변수 ----
NNUNET_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet")
os.environ["nnUNet_raw"] = str(NNUNET_ROOT / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(NNUNET_ROOT / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(NNUNET_ROOT / "nnUNet_results")

NNUNET_PREDICT_BIN = Path(sys.exec_prefix) / "bin" / "nnUNetv2_predict"

DATASET_ID = 3  # Dataset003_MBHSeg25
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"
FOLDS = (0, 1, 2, 3, 4)

CLASS_NAMES = {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"}

TEST_SETS = {
    #"MBH-Seg25": NNUNET_ROOT / "nnUNet_test" / "Test1_MBHSeg25",
    "CT-ICH": NNUNET_ROOT / "nnUNet_test" / "Test2_CTICH",
}

# nnUNetv2 configuration 이름: 3D는 "3d_fullres", 2D는 "2d" (폴더명 기준)
CONFIGS = {"3d_fullres": "3D", "2d": "2D"}

PRED_ROOT = NNUNET_ROOT / "nnUNet_predictions" / "Dataset003_MBHSeg25"

# 6, 7번 GPU만 번갈아 사용
GPU_POOL = ["6", "7"]


def run_predict(input_dir: Path, output_dir: Path, config: str, gpu_id: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(NNUNET_PREDICT_BIN),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(DATASET_ID),
        "-c", config,
        "-tr", TRAINER,
        "-p", PLANS,
        "-f", *[str(f) for f in FOLDS],
        "-device", "cuda",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"RUN (GPU {gpu_id}):", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def dice_iou(pred: np.ndarray, gt: np.ndarray, cls: int):
    p = pred == cls
    g = gt == cls
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    p_sum, g_sum = p.sum(), g.sum()
    if p_sum + g_sum == 0:
        return np.nan, np.nan
    dice = 2 * inter / (p_sum + g_sum)
    iou = inter / union if union > 0 else np.nan
    return dice, iou


def evaluate(pred_dir: Path, gt_dir: Path) -> pd.DataFrame:
    rows = []
    for gt_path in sorted(gt_dir.glob("*.nii*")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            print(f"[경고] 예측 결과 없음: {pred_path}")
            continue
        gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)
        pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)

        row = {"case": gt_path.name}
        for cls, name in CLASS_NAMES.items():
            dice, iou = dice_iou(pred, gt, cls)
            row[f"{name}_dice"] = dice
            row[f"{name}_iou"] = iou
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    results = []
    call_idx = 0

    for test_name, test_dir in TEST_SETS.items():
        images_dir = test_dir / "imagesTs"
        labels_dir = test_dir / "labelsTs"

        if not images_dir.exists():
            print(f"[건너뜀] {test_name}: imagesTs 없음 ({images_dir})")
            continue

        for config, dim_label in CONFIGS.items():
            pred_dir = PRED_ROOT / config / test_name
            gpu_id = GPU_POOL[call_idx % len(GPU_POOL)]
            call_idx += 1
            run_predict(images_dir, pred_dir, config, gpu_id)

            df_case = evaluate(pred_dir, labels_dir)
            if df_case.empty:
                print(f"[경고] {test_name}/{dim_label}: 평가할 케이스 없음")
                continue

            dice_cols = [f"{name}_dice" for name in CLASS_NAMES.values()]
            iou_cols = [f"{name}_iou" for name in CLASS_NAMES.values()]
            mean_dice = df_case[dice_cols].mean()
            mean_iou = df_case[iou_cols].mean()

            row = {"Test Dataset": test_name, "Dimension": dim_label}
            for name in CLASS_NAMES.values():
                row[name] = mean_dice[f"{name}_dice"]
            row["mDice"] = mean_dice.mean()
            row["mIoU"] = mean_iou.mean()
            results.append(row)

    if not results:
        print("결과 없음")
        return

    results_df = pd.DataFrame(results)[
        ["Test Dataset", "Dimension", "mDice", "EDH", "IPH", "IVH", "SAH", "SDH", "mIoU"]
    ]
    print("\n===== nnUNetv2 5cls 결과 =====")
    print(results_df.round(4).to_string(index=False))

    out_csv = NNUNET_ROOT / "nnUNet_predictions" / "results_5cls.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        # 기존 결과 중, 이번에 실행한 (Test Dataset, Dimension) 조합만 새 값으로 교체, 나머지는 유지
        existing_df = pd.read_csv(out_csv)
        key_cols = ["Test Dataset", "Dimension"]
        new_keys = set(map(tuple, results_df[key_cols].values.tolist()))
        existing_df = existing_df[~existing_df[key_cols].apply(tuple, axis=1).isin(new_keys)]
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(out_csv, index=False)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
