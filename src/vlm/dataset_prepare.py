"""
nnUNet 5cls 예측과 5cls GT를 인스턴스 단위로 매칭해 VLM(Qwen) 입력용 데이터셋 준비

pipeline:
    semantic mask -> instance mask/bbox
    -> pred 인스턴스별 bbox overlay + crop 4종 이미지 저장
    -> pred 인스턴스와 GT 인스턴스를 픽셀 겹침으로 매칭
    -> dataset.json 샘플 기록

dataset.json 샘플 형식 — 필드는 데이터 계보 순서(gt -> segmentation pred -> vlm#1 pred ->
vlm#2 pred)로 엔티티 접두어를 붙여 정리한다. vlm1_is_lesion/vlm2_lesion_type은 이 스크립트가
아니라 vlm_is_lesion.py/vlm_lesion_subtype.py가 나중에 채운다(dataset_prepare 시점엔 없음):
    {
        "id": "case_id_slice_idx_inst_id",
        "case_id": "case_id",
        "slice_idx": slice_idx,
        "bbox": [min_row, min_col, max_row, max_col],
        "area_px": area_px,
        "gt_is_lesion": gt_is_lesion,
        "gt_lesion_type_id": gt_lesion_type_id,
        "gt_lesion_type": gt_lesion_type,
        "seg_lesion_type_id": seg_lesion_type_id,   # nnUNet(Reader#1) 원래 판정, GT/VLM과 무관
        "seg_lesion_type": seg_lesion_type,
        "seg_lesion_type_correct": seg_lesion_type_correct,
        "images": {
            "overlay": "images/overlay/{id}.png",
            "crop_only": "images/crop_only/{id}.png",
            "crop_min": "images/crop_min/{id}.png",
            "crop_min_bbox": "images/crop_min_bbox/{id}.png",
            "crop_min_contour": "images/crop_min_contour/{id}.png",
        }
    }
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from src.utils import CLASS_NAMES, apply_window, load_lps_array
from src.vlm.image_crop import CROP_MODES, crop_image, draw_single_bbox_overlay


def semantic_to_instance(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """클래스별 connected component로 instance 분리. class_id는 원본 semantic 값 유지."""
    instance_mask = np.zeros_like(mask, dtype=np.int32)
    regions_info = []
    next_id = 1
    for cls_id in np.unique(mask):
        if cls_id == 0:
            continue
        labeled = label(mask == cls_id)
        for region in regionprops(labeled):
            instance_mask[labeled == region.label] = next_id
            regions_info.append({
                "instance_id": next_id,
                "class_id": int(cls_id),
                "bbox": region.bbox,
                "area_px": int(region.area),
                "eccentricity": float(region.eccentricity),
            })
            next_id += 1
    return instance_mask, regions_info


def match_pred_to_gt(pred_mask: np.ndarray, pred_regions: list[dict],
                      gt_mask: np.ndarray, gt_regions: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    pred 인스턴스마다 GT(클래스 무관하게 합친 마스크)와 픽셀 겹침을 확인.
    - 안 겹치면: is_lesion=False, gt_class_id=None (FP)
    - 겹치면: is_lesion=True, 가장 많이 겹친 GT 인스턴스의 클래스를 gt_class_id로 기록
    fn_only: 어떤 pred와도 안 겹친 GT 인스턴스(통계용, 샘플엔 미포함)
    """
    gt_class_lookup = {r["instance_id"]: r["class_id"] for r in gt_regions}

    matched = []
    matched_gt_ids = set()

    for pred in pred_regions:
        pred_pixels = pred_mask == pred["instance_id"]
        overlap_ids = np.unique(gt_mask[pred_pixels])
        overlap_ids = overlap_ids[overlap_ids > 0]

        if len(overlap_ids) == 0:
            matched.append({**pred, "is_lesion": False, "gt_class_id": None})
            continue

        overlap_counts = {gid: int(np.sum((gt_mask == gid) & pred_pixels)) for gid in overlap_ids}
        best_gt_id = max(overlap_counts, key=overlap_counts.get)
        matched_gt_ids.add(best_gt_id)
        matched.append({**pred, "is_lesion": True, "gt_class_id": gt_class_lookup[best_gt_id]})

    fn_only = [r for r in gt_regions if r["instance_id"] not in matched_gt_ids]
    return matched, fn_only


def build_slice_samples(case_id: str, slice_idx: int, image_2d: np.ndarray,
                         pred_2d: np.ndarray, gt_2d: np.ndarray, out_dir: Path,
                         class_names: dict = CLASS_NAMES) -> tuple[list[dict], int]:
    """
    한 슬라이스의 5cls pred/gt를 인스턴스 매칭해 샘플 목록을 만들고,
    pred 인스턴스별 overlay + crop 4종 이미지를 out_dir/images/{타입}/ 아래에 저장한다.
    반환: (samples, fn_only 개수)
    """
    windowed = apply_window(image_2d)

    pred_mask, pred_regions = semantic_to_instance(pred_2d)
    gt_mask, gt_regions = semantic_to_instance(gt_2d)

    if not pred_regions:
        return [], len(gt_regions)

    matched, fn_only = match_pred_to_gt(pred_mask, pred_regions, gt_mask, gt_regions)

    samples = []
    for region in matched:
        sample_id = f"{case_id}_s{slice_idx:03d}_inst{region['instance_id']}"
        instance_mask = pred_mask == region["instance_id"]

        overlay_img = draw_single_bbox_overlay(windowed, region["bbox"])
        overlay_path = out_dir / "images" / "overlay" / f"{sample_id}.png"
        overlay_img.save(overlay_path)

        images = {"overlay": str(overlay_path)}
        for mode in CROP_MODES:
            crop_img = crop_image(windowed, region["bbox"], mode=mode, instance_mask=instance_mask)
            crop_path = out_dir / "images" / mode / f"{sample_id}.png"
            crop_img.save(crop_path)
            images[mode] = str(crop_path)

        seg_lesion_type_id = region["class_id"]
        gt_lesion_type_id = region["gt_class_id"]

        samples.append({
            "id": sample_id,
            "case_id": case_id,
            "slice_idx": slice_idx,
            "bbox": list(region["bbox"]),
            "area_px": region["area_px"],
            "gt_is_lesion": region["is_lesion"],
            "gt_lesion_type_id": gt_lesion_type_id,
            "gt_lesion_type": class_names[gt_lesion_type_id] if gt_lesion_type_id else None,
            "seg_lesion_type_id": seg_lesion_type_id,
            "seg_lesion_type": class_names[seg_lesion_type_id],
            "seg_lesion_type_correct": (seg_lesion_type_id == gt_lesion_type_id) if region["is_lesion"] else None,
            "images": images,
        })

    return samples, len(fn_only)


def build_dataset(pred_dir: Path, gt_dir: Path, images_dir: Path, out_dir: Path,
                   class_names: dict = CLASS_NAMES) -> tuple[list[dict], list[dict]]:
    """
    pred_dir의 케이스마다 gt_dir/images_dir에서 짝을 찾아 build_slice_samples를 슬라이스별로 실행.
    반환: (전체 샘플 목록, FN 발생 슬라이스 로그)
    """
    out_dir = Path(out_dir)
    for mode in ("overlay", *CROP_MODES):
        (out_dir / "images" / mode).mkdir(parents=True, exist_ok=True)

    all_samples = []
    fn_log = []

    pred_cases = sorted(p.stem.replace(".nii", "") for p in Path(pred_dir).glob("*.nii.gz"))
    print(f"전체 케이스 수: {len(pred_cases)}")

    for case_id in pred_cases:
        pred_path = Path(pred_dir) / f"{case_id}.nii.gz"
        gt_path = Path(gt_dir) / f"{case_id}.nii.gz"
        img_path = Path(images_dir) / f"{case_id}_0000.nii.gz"

        if not gt_path.exists() or not img_path.exists():
            print(f"[건너뜀] {case_id}: gt 또는 원본 이미지 없음")
            continue

        pred_vol = load_lps_array(pred_path).astype(np.uint8)
        gt_vol = load_lps_array(gt_path).astype(np.uint8)
        img_vol = load_lps_array(img_path)

        for z in range(pred_vol.shape[0]):
            pred_2d, gt_2d, img_2d = pred_vol[z], gt_vol[z], img_vol[z]
            if pred_2d.sum() == 0 and gt_2d.sum() == 0:
                continue

            samples, n_fn = build_slice_samples(case_id, z, img_2d, pred_2d, gt_2d, out_dir, class_names)
            all_samples.extend(samples)
            if n_fn > 0:
                fn_log.append({"case_id": case_id, "slice_idx": z, "n_fn": n_fn})

    print(f"총 샘플 수: {len(all_samples)}, FN 발생 슬라이스 수: {len(fn_log)}")
    return all_samples, fn_log


def save_dataset(all_samples: list[dict], fn_log: list[dict], out_dir: Path, dataset_filename: str) -> None:
    out_dir = Path(out_dir)
    dataset_path = out_dir / dataset_filename
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    fn_log_path = out_dir / "fn_log.csv"
    pd.DataFrame(fn_log).to_csv(fn_log_path, index=False)

    print(f"저장 완료: {dataset_path} ({len(all_samples)}개 샘플)")
    print(f"FN 로그: {fn_log_path}")
