# nnUNet 5cls background-threshold 예비 스윕 (두 가지 관점)
#
# 1) sens_spec_sweep: scripts/nnUNet_prob_5cls.py가 뽑아둔 확률맵으로, postprocessing 없이
#    순수 numpy로 threshold별 병변 유무(ICH vs 배경) sensitivity/specificity를 빠르게 훑어본다.
#    GPU 불필요. 본 실험(scripts/nnUNet_infer_thr_sweep_5cls.py, threshold=0.5/0.4/0.3/0.2/0.1)의
#    값들이 적절한 구간인지 참고하는 용도.
#    threshold=0.50 시뮬레이션 결과와, 확률맵과 함께 저장된 nnUNet 진짜 기본 argmax 결과의
#    sens/spec도 같이 계산해서 nnUNet default (argmax) 행으로 별도 비교한다 (배경확률<0.5
#    규칙은 6클래스 전체 argmax와 수학적으로 완전히 동일하지 않음).
#    결과: experiments/260821_nnunet-infer-5cls-thr-sweep-mbhseg25/output/sens_spec_sweep.csv
#
# 2) 2cls_relabel_eval: nnUNet_infer_thr_sweep_5cls.py가 만든 6개 postprocessing 적용 결과
#    (nnUNet default + thr=0.5/0.4/0.3/0.2/0.1)를 relabel_multiclass_to_binary로 이진화한 뒤
#    nnUNetv2_evaluate_folder(공식 CLI)로 2cls GT와 비교한다. notebook/nnUNet_result.ipynb
#    섹션 7(5 cls to 2 cls evaluation)과 동일한 패턴.
#    결과: experiments/260821_nnunet-infer-5cls-to-2cls-sweep/output/pred_2cls_from_5cls/{variant}/*.nii.gz
#          experiments/260821_nnunet-infer-5cls-to-2cls-sweep/output/results.csv
#    GPU 불필요 (relabel/evaluate 전부 CPU 전용)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.nnunet.relabel import relabel_multiclass_to_binary
from src.segmentation.nnunet.test import (
    NNUNetConfig,
    build_results_df,
    run_evaluate_folder,
    save_or_merge_csv,
    summary_to_row,
)
from src.segmentation.nnunet.threshold import (
    compute_sens_spec,
    load_case_probabilities,
    load_gt_array,
    threshold_binary,
)

NNUNET_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet")

LABELS_DIR = NNUNET_ROOT / "test_data" / "Test1_MBHSeg25" / "labelsTs"
PROB_DIR = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/experiments/260821_nnunet-infer-5cls-prob-mbhseg25/output/2d/MBH-Seg25")
SENS_SPEC_OUT_CSV = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/experiments/260821_nnunet-infer-5cls-thr-sweep-mbhseg25/output/sens_spec_sweep.csv")
SENS_SPEC_THRESHOLDS = np.arange(0.1, 0.6, 0.05).round(2)

CFG_2CLS = NNUNetConfig(nnunet_root=NNUNET_ROOT, dataset_id=2, dataset_name="Dataset002_MBHSeg25_1cls")
CFG_2CLS.set_env()
CLASS_NAMES_2CLS = {1: "ICH"}
GT_2CLS_DIR = NNUNET_ROOT / "test_data_1cls" / "Test1_MBHSeg25_1cls" / "labelsTs"

THR_SWEEP_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/experiments/260821_nnunet-infer-5cls-thr-sweep-mbhseg25/output")
RELABEL_OUT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/experiments/260821_nnunet-infer-5cls-to-2cls-sweep/output")
PRED_2CLS_ROOT = RELABEL_OUT_ROOT / "pred_2cls_from_5cls"
RELABEL_OUT_CSV = RELABEL_OUT_ROOT / "results.csv"

# (5cls PP 결과 폴더, Dimension에 쓸 variant 이름) — nnUNet_infer_thr_sweep_5cls.py 출력과 매칭
VARIANTS = [
    (THR_SWEEP_ROOT / "baseline_pp", "nnUNet default"),
    (THR_SWEEP_ROOT / "thr0.5" / "pp", "thr=0.5"),
    (THR_SWEEP_ROOT / "thr0.4" / "pp", "thr=0.4"),
    (THR_SWEEP_ROOT / "thr0.3" / "pp", "thr=0.3"),
    (THR_SWEEP_ROOT / "thr0.2" / "pp", "thr=0.2"),
    (THR_SWEEP_ROOT / "thr0.1" / "pp", "thr=0.1"),
    (THR_SWEEP_ROOT / "thr0.09" / "pp", "thr=0.09"),
    (THR_SWEEP_ROOT / "thr0.08" / "pp", "thr=0.08"),
    (THR_SWEEP_ROOT / "thr0.07" / "pp", "thr=0.07"),
    (THR_SWEEP_ROOT / "thr0.06" / "pp", "thr=0.06"),
    (THR_SWEEP_ROOT / "thr0.05" / "pp", "thr=0.05"),
    (THR_SWEEP_ROOT / "thr0.04" / "pp", "thr=0.04"),
    (THR_SWEEP_ROOT / "thr0.03" / "pp", "thr=0.03"),
    (THR_SWEEP_ROOT / "thr0.02" / "pp", "thr=0.02"),
    (THR_SWEEP_ROOT / "thr0.01" / "pp", "thr=0.01"),
    (THR_SWEEP_ROOT / "thr0.009" / "pp", "thr=0.009"),
    (THR_SWEEP_ROOT / "thr0.008" / "pp", "thr=0.008"),
    (THR_SWEEP_ROOT / "thr0.007" / "pp", "thr=0.007"),
    (THR_SWEEP_ROOT / "thr0.006" / "pp", "thr=0.006"),
    (THR_SWEEP_ROOT / "thr0.005" / "pp", "thr=0.005"),
    (THR_SWEEP_ROOT / "thr0.004" / "pp", "thr=0.004"),
    (THR_SWEEP_ROOT / "thr0.003" / "pp", "thr=0.003"),
    (THR_SWEEP_ROOT / "thr0.002" / "pp", "thr=0.002"),
    (THR_SWEEP_ROOT / "thr0.001" / "pp", "thr=0.001"),
]


def run_sens_spec_sweep():
    case_ids = sorted(p.stem for p in PROB_DIR.glob("*.npz"))
    if not case_ids:
        raise RuntimeError(f"확률맵이 없음: {PROB_DIR} (scripts/nnUNet_prob_5cls.py를 먼저 실행)")
    print(f"[sens_spec_sweep] 대상 케이스 수: {len(case_ids)}")

    gt_cache = {cid: load_gt_array(LABELS_DIR / f"{cid}.nii.gz") > 0 for cid in case_ids}
    prob_cache = {cid: load_case_probabilities(PROB_DIR, cid)[0] for cid in case_ids}

    rows = []
    for thr in SENS_SPEC_THRESHOLDS:
        sens_list, spec_list = [], []
        for cid in case_ids:
            pred = threshold_binary(prob_cache[cid], threshold=float(thr))
            sens, spec = compute_sens_spec(pred, gt_cache[cid])
            sens_list.append(sens)
            spec_list.append(spec)
        rows.append({
            "threshold": f"{thr:.2f}",
            "sensitivity": float(np.nanmean(sens_list)),
            "specificity": float(np.nanmean(spec_list)),
        })
        print(f"thr={thr:.2f}  sensitivity={rows[-1]['sensitivity']:.4f}  specificity={rows[-1]['specificity']:.4f}")

    # nnUNet 진짜 기본 argmax(threshold 규칙 아님)의 sens/spec — thr=0.50 시뮬레이션과 비교용
    sens_list, spec_list = [], []
    for cid in case_ids:
        pred_argmax = (prob_cache[cid].argmax(0) > 0).astype(np.uint8)
        sens, spec = compute_sens_spec(pred_argmax, gt_cache[cid])
        sens_list.append(sens)
        spec_list.append(spec)
    rows.append({
        "threshold": "nnUNet default (argmax)",
        "sensitivity": float(np.nanmean(sens_list)),
        "specificity": float(np.nanmean(spec_list)),
    })
    print(f"nnUNet default (argmax)  sensitivity={rows[-1]['sensitivity']:.4f}  specificity={rows[-1]['specificity']:.4f}")

    df = pd.DataFrame(rows)
    SENS_SPEC_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SENS_SPEC_OUT_CSV, index=False)
    print(f"[sens_spec_sweep] 저장: {SENS_SPEC_OUT_CSV}")
    print(df.to_string(index=False))


def run_2cls_relabel_eval():
    results = []
    for pred_5cls_dir, variant_name in VARIANTS:
        if not pred_5cls_dir.exists():
            print(f"[2cls_relabel_eval] [건너뜀] 없음: {pred_5cls_dir}")
            continue

        pred_2cls_dir = PRED_2CLS_ROOT / variant_name.replace("=", "").replace(".", "")
        relabel_multiclass_to_binary(pred_5cls_dir, pred_2cls_dir)

        summary_path = run_evaluate_folder(CFG_2CLS, GT_2CLS_DIR, pred_2cls_dir)
        row = summary_to_row(summary_path, "MBH-Seg25", f"2D (PP, 5cls->2cls, {variant_name})", CLASS_NAMES_2CLS)
        results.append(row)

    if not results:
        print("[2cls_relabel_eval] 결과 없음")
        return

    results_df = build_results_df(results, CLASS_NAMES_2CLS, macro_metrics=())
    print("\n===== [2cls_relabel_eval] 5cls->2cls threshold sweep 결과 =====")
    print(results_df.to_string(index=False))

    save_or_merge_csv(results_df, RELABEL_OUT_CSV)
    print(f"[2cls_relabel_eval] 저장: {RELABEL_OUT_CSV}")


def main():
    run_sens_spec_sweep()
    run_2cls_relabel_eval()


if __name__ == "__main__":
    main()
