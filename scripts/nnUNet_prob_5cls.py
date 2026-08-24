# nnUNet 5cls 확률맵(softmax probabilities) 저장
#
# 2D 가중치(5-fold 앙상블): nnUNet_results/Dataset001_MBHSeg25/nnUNetTrainer__nnUNetPlans__2d
# 테스트 데이터: nnUNet/test_data/Test1_MBHSeg25 (imagesTs)
#
# --save_probabilities로 예측하면 케이스마다 기본 argmax .nii.gz(공짜 부산물) + 6채널
# 확률맵 .npz(키 probabilities) + write_seg에 필요한 properties_dict .pkl이 함께 저장된다.
# 이 확률맵은 scripts/nnUNet_infer_thr_sweep_5cls.py, scripts/nnUNet_infer_thr_sweep_relabel.py에서
# threshold 실험 입력으로 재사용한다.
#
# 결과 저장 위치: experiments/260821_nnunet-infer-5cls-prob-mbhseg25/output/2d/MBH-Seg25
# GPU: 7번 우선, 실패(OOM 등) 시 5번으로 재시도

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.nnunet.test import NNUNetConfig, run_predict

NNUNET_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet")
CFG = NNUNetConfig(nnunet_root=NNUNET_ROOT, dataset_id=1, dataset_name="Dataset001_MBHSeg25")
CFG.set_env()

IMAGES_DIR = NNUNET_ROOT / "test_data" / "Test1_MBHSeg25" / "imagesTs"
OUTPUT_DIR = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/experiments/260821_nnunet-infer-5cls-prob-mbhseg25/output/2d/MBH-Seg25")

GPU_PRIMARY = "7"
GPU_FALLBACK = "5"


def expected_case_ids() -> list[str]:
    return sorted(p.name.replace("_0000.nii.gz", "") for p in IMAGES_DIR.glob("*_0000.nii.gz"))


def already_done(case_ids: list[str]) -> bool:
    if not OUTPUT_DIR.exists():
        return False
    return all((OUTPUT_DIR / f"{cid}.npz").exists() for cid in case_ids)


def main():
    case_ids = expected_case_ids()
    print(f"대상 케이스 수: {len(case_ids)}")

    if already_done(case_ids):
        print(f"[건너뜀] 이미 완료됨: {OUTPUT_DIR}")
        return

    try:
        run_predict(CFG, IMAGES_DIR, OUTPUT_DIR, config="2d", gpu_id=GPU_PRIMARY, save_probabilities=True)
    except subprocess.CalledProcessError as e:
        print(f"[GPU {GPU_PRIMARY} 실패, GPU {GPU_FALLBACK}로 재시도] {e}")
        run_predict(CFG, IMAGES_DIR, OUTPUT_DIR, config="2d", gpu_id=GPU_FALLBACK, save_probabilities=True)

    n_npz = len(list(OUTPUT_DIR.glob("*.npz")))
    n_nii = len(list(OUTPUT_DIR.glob("*.nii.gz")))
    print(f"완료: npz {n_npz}개, nii.gz {n_nii}개 -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
