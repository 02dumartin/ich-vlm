"""전경 확률 임계값 기반 재분류(re-argmax) 유틸.

nnUNet 기본 판정은 6채널 softmax(배경 포함)에서 argmax를 그대로 쓴다. 
민감도(recall)를 올리기 위해, 전경 확률 합(1-배경확률)이 threshold보다 큰 voxel만 무조건 전경 클래스 중
argmax로 강제 배정하는 규칙을 적용한다. threshold가 낮을수록 더 관대하게 전경으로 강제
배정되어(배경확률 < 1-threshold) recall은 오르고 FP도 늘어난다.
threshold=0.5는 배경확률<0.5와 동일하며, 6채널 전체 argmax와 근사하지만 완전히 같지는
않다(배경확률이 0.5보다 낮아도 6개 채널 중 개별 최댓값이면 argmax는 여전히 배경을 고를
수 있음).

--save_probabilities로 예측하면 원본 이미지 공간으로 이미 resample된 확률맵(.npz, 키
probabilities, shape (C,z,y,x), C=6)과 write_seg에 필요한 properties_dict(.pkl)가 같이
저장된다. 이 모듈은 그 두 파일을 읽어 threshold 규칙을 적용한 뒤 nnUNet 자체 writer로
다시 nii.gz를 저장한다.
"""

import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def apply_threshold_to_probabilities(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """
    probabilities: (C, z, y, x), 채널 0=background. 
    전경 확률 합(1-배경확률)이 threshold보다 큰(=배경확률이 1-threshold보다 작은) voxel만 
    전경 클래스(1..C-1) 중 argmax로 강제 배정, 나머지는 배경(0). 
    threshold가 낮을수록 더 관대해져 전경이 늘어난다.
    """
    bg_prob = probabilities[0]
    fg = probabilities[1:].argmax(0) + 1
    seg = np.where(bg_prob < (1 - threshold), fg, 0)
    return seg.astype(np.uint8)


def load_case_probabilities(prob_dir: Path, case_id: str) -> tuple[np.ndarray, dict]:
    """case_id.npz, case_id.pkl을 읽어 (probabilities, properties_dict) 반환."""
    probabilities = np.load(prob_dir / f"{case_id}.npz")["probabilities"]
    properties = load_pickle(str(prob_dir / f"{case_id}.pkl"))
    return probabilities, properties


def apply_threshold_to_folder(prob_dir: Path, out_dir: Path, threshold: float,
                               plans_json: Path, file_ending: str = ".nii.gz",
                               skip_existing: bool = True):
    """prob_dir의 모든 case(.npz/.pkl 쌍)에 threshold 규칙을 적용해 
    out_dir에 nii.gz로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plans_manager = PlansManager(str(plans_json))
    rw = plans_manager.image_reader_writer_class()

    case_ids = sorted(p.stem for p in prob_dir.glob("*.npz"))
    for case_id in case_ids:
        out_path = out_dir / f"{case_id}{file_ending}"
        if skip_existing and out_path.exists():
            continue
        probabilities, properties = load_case_probabilities(prob_dir, case_id)
        seg = apply_threshold_to_probabilities(probabilities, threshold)
        rw.write_seg(seg, str(out_path), properties)

    # nnUNetv2_apply_postprocessing/evaluate_folder는 pred 폴더 안에 dataset.json이 이미
    # 복사돼 있을 것을 기대한다 (nnUNetv2_predict가 원래 자동으로 복사해주는 파일).
    # prob_dir(nnUNetv2_predict 출력)에 이미 있는 사본을 그대로 가져옴.
    for meta_name in ("dataset.json", "plans.json"):
        src = prob_dir / meta_name
        if src.exists():
            shutil.copy(src, out_dir / meta_name)

    print(f"[threshold={threshold}] {prob_dir} -> {out_dir} 완료 ({len(case_ids)} cases)")


def threshold_binary(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """5클래스 threshold 세그멘테이션을 만든 뒤 병변 유무(전경>0) 이진 배열로 축약."""
    return (apply_threshold_to_probabilities(probabilities, threshold) > 0).astype(np.uint8)


def compute_sens_spec(pred_binary: np.ndarray, gt_binary: np.ndarray) -> tuple[float, float]:
    """voxel 단위 binary confusion으로 sensitivity(=recall), specificity 계산.
    gt_binary에 병변이 전혀 없으면 sensitivity는 정의 불가(NaN)로, 배경 voxel이 전혀 없으면
    specificity도 NaN으로 반환한다."""
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)

    tp = np.logical_and(pred_binary, gt_binary).sum()
    fn = np.logical_and(~pred_binary, gt_binary).sum()
    tn = np.logical_and(~pred_binary, ~gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return float(sensitivity), float(specificity)


def load_gt_array(gt_path: Path) -> np.ndarray:
    """GT label nii.gz를 nnUNet의 write_seg/probabilities와 동일한 배열 축 순서(z,y,x)로 로드."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
