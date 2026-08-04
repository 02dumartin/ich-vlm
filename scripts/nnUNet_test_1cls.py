# nnUNet 1cls(2cls: background+ich) Test
#
# 3D 가중치: nnUNet_results/Dataset004_MBHSeg25_1cls/nnUNetTrainer__nnUNetPlans__3d_fullres
# 2D 가중치: nnUNet_results/Dataset004_MBHSeg25_1cls/nnUNetTrainer__nnUNetPlans__2d
#
# test 데이터
#   /home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet/nnUNet_test_1cls/Test1_MBHSeg25_1cls  (3D+2D)
#   /home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet/nnUNet_test_1cls/Test2_CTICH_1cls      (3D+2D)
#   SNU HE-01                                                                                      (2D만)

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

DATASET_ID = 4  # Dataset004_MBHSeg25_1cls
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"
FOLDS = (0, 1, 2, 3, 4)

FG_CLASS = 1  # ich

# 테스트셋별로 어떤 차원(config)을 시도할지 정의
# HE-01은 3D 볼륨을 만들 수 없어서 2D만 시도
TEST_SETS = {
    #"MBH-Seg25": {
    #    "dir": NNUNET_ROOT / "nnUNet_test_1cls" / "Test1_MBHSeg25_1cls",
    #    "configs": {"3d_fullres": "3D", "2d": "2D"},
    #},
    "CT-ICH": {
        "dir": NNUNET_ROOT / "nnUNet_test_1cls" / "Test2_CTICH_1cls",
        "configs": {"3d_fullres": "3D", "2d": "2D"},
    },
    
    #"SNU HE-01": {
    #    "dir": NNUNET_ROOT / "nnUNet_test_1cls" / "Test3_HE01_1cls",  # TODO: 아직 미생성
    #    "configs": {"2d": "2D"},
    #},
}

PRED_ROOT = NNUNET_ROOT / "nnUNet_predictions" / "Dataset004_MBHSeg25_1cls"

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
        dice, iou = dice_iou(pred, gt, FG_CLASS)
        rows.append({"case": gt_path.name, "dice": dice, "iou": iou})
    return pd.DataFrame(rows)


def main():
    # {test_name: {"3D Dice":..,"3D IoU":..,"2D Dice":..,"2D IoU":..}}
    table = {name: {} for name in TEST_SETS}
    call_idx = 0

    for test_name, spec in TEST_SETS.items():
        test_dir = spec["dir"]
        images_dir = test_dir / "imagesTs"
        labels_dir = test_dir / "labelsTs"

        if not test_dir.exists():
            print(f"[건너뜀] {test_name}: 폴더 없음 ({test_dir}) -> TODO: 데이터 준비 필요")
            continue
        if not images_dir.exists():
            print(f"[건너뜀] {test_name}: imagesTs 없음 ({images_dir})")
            continue

        for config, dim_label in spec["configs"].items():
            pred_dir = PRED_ROOT / config / test_name.replace(" ", "_")
            gpu_id = GPU_POOL[call_idx % len(GPU_POOL)]
            call_idx += 1
            run_predict(images_dir, pred_dir, config, gpu_id)

            df_case = evaluate(pred_dir, labels_dir)
            if df_case.empty:
                print(f"[경고] {test_name}/{dim_label}: 평가할 케이스 없음")
                continue

            table[test_name][f"{dim_label} Dice"] = df_case["dice"].mean()
            table[test_name][f"{dim_label} IoU"] = df_case["iou"].mean()

    columns = ["3D Dice", "3D IoU", "2D Dice", "2D IoU"]
    rows = []
    for test_name, spec in TEST_SETS.items():
        row = {"Test Dataset": test_name}
        available_dims = set(spec["configs"].values())
        for col in columns:
            dim = col.split(" ")[0]
            if dim not in available_dims:
                row[col] = "-"  # 애초에 이 차원으로 테스트 불가 (예: HE-01의 3D)
            else:
                row[col] = table[test_name].get(col, np.nan)
        rows.append(row)

    results_df = pd.DataFrame(rows)[["Test Dataset"] + columns]
    print("\n===== nnUNetv2 1cls(2cls) 결과 =====")
    print(results_df.round(4).to_string(index=False))

    out_csv = NNUNET_ROOT / "nnUNet_predictions" / "results_1cls.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        # 기존 결과 중, 이번에 실행하지 않은 데이터셋의 행만 남기고 이번 결과와 합침
        existing_df = pd.read_csv(out_csv)
        existing_df = existing_df[~existing_df["Test Dataset"].isin(results_df["Test Dataset"])]
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(out_csv, index=False)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
