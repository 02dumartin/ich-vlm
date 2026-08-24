# nnUNet 5cls background-threshold sensitivity 실험 (공식 지표)
#
# scripts/nnUNet_prob_5cls.py가 미리 뽑아둔 확률맵(experiments/260821_nnunet-infer-5cls-prob-mbhseg25)을
# 입력으로, 전경 확률 합(1-배경확률) threshold를 낮춰 recall(=sensitivity)을 올리는 실험.
# threshold가 낮을수록 더 관대한 기준이 되어 더 많은 voxel이 전경으로 강제 배정된다.
#
# 흐름:
#   1) nnUNetv2_find_best_configuration(-c 2d)로 2D 단독 postprocessing.pkl 확보
#      (기존 nnUNet_infer_5cls.py의 2d_pp와 동일한 pkl)
#   2) baseline: 확률맵과 함께 저장된 nnUNet 기본 argmax nii.gz에 위 pkl 적용
#   3) thr=0.5/0.4/0.3/0.2/0.1: 확률맵에 threshold 규칙 적용 -> 동일 pkl 적용 -> 평가
#      (postprocessing.pkl은 모든 variant에 동일하게 재사용 - 재검색하지 않음)
#   4) 6개 행(Dimension: nnUNet default (PP) / thr=0.5..0.1 (PP))을 results_thr_test.csv에 저장
#      (mDice/mIoU/mRecall/mPrecision + 클래스별 Dice/IoU/Precision/Recall/n_cases)
#
# GPU 불필요 (postprocessing/evaluate는 CPU 전용, 확률맵 예측은 nnUNet_prob_5cls.py에서 이미 완료)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.nnunet.test import (
    NNUNetConfig,
    add_macro_average,
    build_results_df,
    postprocess_and_evaluate,
    run_find_best_config,
    save_or_merge_csv,
    summary_to_row,
)
from src.segmentation.nnunet.threshold import apply_threshold_to_folder

NNUNET_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet")
CFG = NNUNetConfig(nnunet_root=NNUNET_ROOT, dataset_id=1, dataset_name="Dataset001_MBHSeg25")
CFG.set_env()

CLASS_NAMES = {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"}
TEST_NAME = "MBH-Seg25"
LABELS_DIR = NNUNET_ROOT / "test_data" / "Test1_MBHSeg25" / "labelsTs"

PROB_DIR = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/experiments/260821_nnunet-infer-5cls-prob-mbhseg25/output/2d/MBH-Seg25")
OUT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/experiments/260821_nnunet-infer-5cls-thr-sweep-mbhseg25/output")

THRESHOLDS = [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]


def main():
    if not any(PROB_DIR.glob("*.npz")):
        raise RuntimeError(f"확률맵이 없음: {PROB_DIR} (scripts/nnUNet_prob_5cls.py를 먼저 실행)")

    best = run_find_best_config(CFG, configs=("2d",))
    if best is None:
        raise RuntimeError("2d 단독 postprocessing.pkl을 찾지 못함 (run_find_best_config 실패)")
    pp_pkl, plans_json = best["postprocessing_pkl"], best["plans_json"]
    print(f"[postprocessing.pkl] {pp_pkl}")

    results = []

    # baseline: 확률맵과 함께 저장된 nnUNet 기본 argmax nii.gz
    baseline_pp_dir = OUT_ROOT / "baseline_pp"
    summary_path = postprocess_and_evaluate(CFG, PROB_DIR, baseline_pp_dir, LABELS_DIR, pp_pkl, plans_json)
    results.append(summary_to_row(summary_path, TEST_NAME, "nnUNet default (PP)", CLASS_NAMES))

    for thr in THRESHOLDS:
        raw_dir = OUT_ROOT / f"thr{thr}" / "raw"
        pp_dir = OUT_ROOT / f"thr{thr}" / "pp"

        apply_threshold_to_folder(PROB_DIR, raw_dir, thr, plans_json)
        summary_path = postprocess_and_evaluate(CFG, raw_dir, pp_dir, LABELS_DIR, pp_pkl, plans_json)
        results.append(summary_to_row(summary_path, TEST_NAME, f"thr={thr} (PP)", CLASS_NAMES))

    results = [add_macro_average(r, CLASS_NAMES, metrics=("Dice", "IoU", "Recall", "Precision")) for r in results]
    results_df = build_results_df(results, CLASS_NAMES, macro_metrics=("Dice", "IoU", "Recall", "Precision"))

    print("\n===== nnUNet 5cls threshold 실험 결과 =====")
    print(results_df.to_string(index=False))

    out_csv = OUT_ROOT / "results_thr_test.csv"
    save_or_merge_csv(results_df, out_csv)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
