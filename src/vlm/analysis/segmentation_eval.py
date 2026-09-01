"""
1) 저장
    VLM 판정(vlm1_is_lesion, vlm2_lesion_type)을 nnUNet 5cls 예측 볼륨에 반영해 nii.gz로 저장
2) 평가
    nnUNet 공식 evaluate_folder(nnUNetv2_evaluate_folder)로 평가
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.measure import label as sklabel
from skimage.measure import regionprops

from src.segmentation.nnunet.test import (
    NNUNetConfig,
    add_macro_average,
    run_evaluate_folder,
    summary_to_row,
)

NNUNET_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet")
CFG_5CLS = NNUNetConfig(nnunet_root=NNUNET_ROOT, dataset_id=1, dataset_name="Dataset001_MBHSeg25")
CFG_2CLS = NNUNetConfig(nnunet_root=NNUNET_ROOT, dataset_id=2, dataset_name="Dataset002_MBHSeg25_1cls")
# 2cls GT는 5cls GT(nnUNet/test_data/Test1_MBHSeg25/labelsTs, 값 0~5)와 다른 파일 -
# nnUNetv2_evaluate_folder는 라벨 "1"만 보고 채점하므로, 진짜 이진(0/1) GT를 따로 써야 한다.
GT_DIR_2CLS = NNUNET_ROOT / "test_data_1cls" / "Test1_MBHSeg25_1cls" / "labelsTs"
CLASS_NAMES_5CLS = {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"}
CLASS_NAMES_2CLS = {1: "lesion"}
MACRO_METRICS = ("Dice", "IoU", "Recall", "Precision")
TEST_NAME = "MBH-Seg25"


def _reorient_like(array: np.ndarray, reference_img: "sitk.Image", original_orientation: str) -> "sitk.Image":
    """LPS 좌표계(dataset_prepare가 bbox/slice_idx를 계산한 것과 동일한 재정렬) 배열을
    reference_img(LPS)의 spacing/origin/direction으로 감싼 뒤, 원본 파일의 orientation으로 되돌린다."""
    img = sitk.GetImageFromArray(array)
    img.CopyInformation(reference_img)
    return sitk.DICOMOrient(img, original_orientation)


def write_binarized_volume(case_id: str, pred_5cls_dir: Path, out_path: Path) -> None:
    """
    VLM 보정 없이 5cls pred를 그대로 이진화(0/1)만 해서 저장
    - 2cls baseline(VLM 개입 전)용.
    LPS 재정렬 없이 원본 orientation 그대로 다뤄도 됨(전체를 균일하게 thresholding하는 것뿐이라
    bbox/slice_idx 좌표계와 무관)."""
    pred_path = Path(pred_5cls_dir) / f"{case_id}.nii.gz"
    pred_img = sitk.ReadImage(str(pred_path))
    binary = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)
    out_img = sitk.GetImageFromArray(binary)
    out_img.CopyInformation(pred_img)
    sitk.WriteImage(out_img, str(out_path))


def write_corrected_volume(case_id: str, records_for_case: list[dict], pred_5cls_dir: Path,
                            out_path: Path, class_names: dict = None) -> None:
    """
    VLM 판정을 실제 5cls 예측 볼륨에 반영해 nii.gz로 저장.
    class_names가 None이면 2cls(vlm1_is_lesion만 반영, 나머지는 이진 1로 유지) 
    - subtype이 아직 없는 단계(vlm#1 is_lesion)에서 씀.
    class_names를 주면 5cls(vlm1_is_lesion=False면 제거, True면 vlm2_lesion_type 클래스로 재라벨링) 
    - subtype까지 돈 단계(vlm#2)에서 씀.
    
    dataset 만들 때와 동일하게 클래스별 connected component로 인스턴스를 다시 뽑아 
    bbox로 픽셀을 특정하고, LPS 좌표계에서 보정한 뒤 원본 orientation으로 되돌려 저장
    """
    pred_path = Path(pred_5cls_dir) / f"{case_id}.nii.gz"
    original_img = sitk.ReadImage(str(pred_path))
    original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(original_img.GetDirection())
    lps_img = sitk.DICOMOrient(original_img, "LPS")
    pred_lps = sitk.GetArrayFromImage(lps_img).astype(np.uint8)

    name_to_class = {v: k for k, v in class_names.items()} if class_names else None
    corrected = pred_lps.copy() if class_names else (pred_lps > 0).astype(np.uint8)

    by_slice = {}
    for r in records_for_case:
        by_slice.setdefault(r["slice_idx"], []).append(r)

    for z, recs in by_slice.items():
        slice_pred = pred_lps[z]
        bbox_to_mask = {}
        for cls_id in np.unique(slice_pred):
            if cls_id == 0:
                continue
            labeled = sklabel(slice_pred == cls_id)
            for region in regionprops(labeled):
                bbox_to_mask[tuple(region.bbox)] = (labeled == region.label)

        for r in recs:
            inst_mask = bbox_to_mask.get(tuple(r["bbox"]))
            if inst_mask is None:
                print(f"[경고] bbox 매칭 실패: {r['id']}")
                continue
            if not r["vlm1_is_lesion"]:
                corrected[z][inst_mask] = 0
            elif name_to_class is not None:
                corrected[z][inst_mask] = name_to_class[r["vlm2_lesion_type"]]
            # 2cls(name_to_class is None)면서 vlm1_is_lesion=True면 이미 1이라 손댈 것 없음

    final_img = _reorient_like(corrected, lps_img, original_orientation)
    sitk.WriteImage(final_img, str(out_path))


def _evaluate_dir(cfg: NNUNetConfig, gt_dir: Path, pred_dir: Path, class_names: dict, row_label: str,
                   skip_existing: bool = True) -> dict:
    cfg.set_env()
    summary_path = pred_dir / "summary.json"
    if not (skip_existing and summary_path.exists()):
        summary_path = run_evaluate_folder(cfg, gt_dir, pred_dir)
    row = summary_to_row(summary_path, TEST_NAME, row_label, class_names)
    if len(class_names) > 1:
        row = add_macro_average(row, class_names, metrics=MACRO_METRICS)
    return row


def _run_segmentation_eval_2cls(records: list[dict], by_case: dict, pred_5cls_dir: Path, out_dir: Path,
                                 all_case_ids: list[str]) -> dict:
    """vlm1_is_lesion만 반영한 2cls(배경/병변) 보정. vlm2_lesion_type 유무와 무관하게 항상 계산 가능.

    baseline_2cls는 crop_mode(=out_dir)와 무관하게 항상 같은 값이라 out_dir의 부모(실험
    output/ 루트) 아래 _baseline_2cls/에 캐싱해서 크롭 모드마다 다시 계산하지 않는다."""
    baseline_dir = out_dir.parent / "_baseline_2cls"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    if not (baseline_dir / "summary.json").exists():
        for cid in all_case_ids:
            write_binarized_volume(cid, pred_5cls_dir, baseline_dir / f"{cid}.nii.gz")
    baseline = _evaluate_dir(CFG_2CLS, GT_DIR_2CLS, baseline_dir, CLASS_NAMES_2CLS, "baseline_2cls")

    corrected_dir = out_dir / "full_corrected_nii_2cls"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    for cid in all_case_ids:
        write_corrected_volume(cid, by_case.get(cid, []), pred_5cls_dir, corrected_dir / f"{cid}.nii.gz",
                                class_names=None)
    vlm_corrected = _evaluate_dir(CFG_2CLS, GT_DIR_2CLS, corrected_dir, CLASS_NAMES_2CLS, "vlm_corrected_2cls",
                                   skip_existing=False)

    print(f"[2cls] baseline Dice: {baseline.get('lesion_Dice'):.4f}  |  "
          f"VLM 보정 후: {vlm_corrected.get('lesion_Dice'):.4f}")
    return {"n_cases": len(all_case_ids), "baseline": baseline, "vlm_corrected": vlm_corrected}


def _run_segmentation_eval_5cls(records: list[dict], by_case: dict, pred_5cls_dir: Path, gt_5cls_dir: Path,
                                 out_dir: Path, all_case_ids: list[str]) -> dict:
    """
    vlm2_lesion_type까지 반영한 5cls 보정. 
    subtype 스테이지가 아직 안 돌았으면(records에 vlm2_lesion_type이 하나도 없으면) None을 반환

    baseline_5cls는 nnUNet_infer_thr_sweep_5cls.py가 이미 만들어둔 pred_5cls_dir/summary.json을
    그대로 재사용한다(재계산 없음 - Row3 baseline과 완전히 동일한 값이어야 하므로)."""
    if not any("vlm2_lesion_type" in r for r in records):
        return None

    baseline = _evaluate_dir(CFG_5CLS, gt_5cls_dir, Path(pred_5cls_dir), CLASS_NAMES_5CLS, "baseline_5cls")

    corrected_dir = out_dir / "full_corrected_nii_5cls"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    for cid in all_case_ids:
        write_corrected_volume(cid, by_case.get(cid, []), pred_5cls_dir, corrected_dir / f"{cid}.nii.gz",
                                class_names=CLASS_NAMES_5CLS)
    vlm_corrected = _evaluate_dir(CFG_5CLS, gt_5cls_dir, corrected_dir, CLASS_NAMES_5CLS, "vlm_corrected_5cls",
                                   skip_existing=False)

    print(f"[5cls] baseline mDice: {baseline.get('mDice'):.4f}  |  "
          f"VLM 보정 후: {vlm_corrected.get('mDice'):.4f}")
    return {"n_cases": len(all_case_ids), "baseline": baseline, "vlm_corrected": vlm_corrected}


def run_segmentation_eval(records: list[dict], pred_5cls_dir: Path, gt_5cls_dir: Path, out_dir: Path) -> dict:
    """
    baseline vs VLM 보정 후를 nnUNet 공식 evaluate_folder로 비교해 반환. 
    2cls(is_lesion만 반영)와 5cls(subtype까지 반영)를 항상 둘 다 계산.
    반환값을 그대로 {tag}_summary.json의 "segmentation" 섹션에 포함
    """
    out_dir = Path(out_dir)

    by_case = {}
    for r in records:
        by_case.setdefault(r["case_id"], []).append(r)

    # nnUNetv2_evaluate_folder는 gt_dir의 모든 파일이 pred_dir에도 있어야 한다 - VLM 데이터셋에
    # 인스턴스가 하나도 없던 케이스(레코드 자체가 없음)도 pred_5cls_dir 전체 파일 목록 기준으로
    # 빠짐없이 써야 한다(그런 케이스는 고칠 인스턴스가 없으니 원본 그대로/이진화만 저장됨).
    all_case_ids = sorted(p.stem.replace(".nii", "") for p in Path(pred_5cls_dir).glob("*.nii.gz"))

    return {
        "2cls": _run_segmentation_eval_2cls(records, by_case, pred_5cls_dir, out_dir, all_case_ids),
        "5cls": _run_segmentation_eval_5cls(records, by_case, pred_5cls_dir, gt_5cls_dir, out_dir, all_case_ids),
    }
