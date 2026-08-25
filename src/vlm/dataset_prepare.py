"""
nnUNet 5cls 예측과 5cls GT를 인스턴스 단위로 매칭해 VLM(Qwen) 입력용 데이터셋 준비

pipeline:
    semantic mask -> instance mask/bbox 
    -> pred 인스턴스별 bbox overlay + crop 이미지 저장
    -> pred 인스턴스와 GT 인스턴스를 픽셀 겹침으로 매칭 
    -> dataset.json 샘플 기록
    
dataset.json 샘플 형식:
    {
        "id": "case_id_slice_idx_inst_id",
        "case_id": "case_id",
        "slice_idx": slice_idx,
        "bbox": [min_row, min_col, max_row, max_col],
        "area_px": area_px,
        "pred_class_id": pred_class_id,
        "pred_class_name": pred_class_name,
        "is_lesion": is_lesion,
        "gt_class_id": gt_class_id,
        "gt_class_name": gt_class_name,
        "class_correct": class_correct,
        "images": {
            "overlay": "overlay.png",
            "cropped": "cropped.png",
        }
    }
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops

from src.utils import CLASS_NAMES, apply_window, load_lps_array


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


def draw_single_bbox_overlay(image_2d: np.ndarray, bbox: tuple, color: tuple = (255, 0, 0)) -> Image.Image:
    """이 인스턴스의 pred bbox 하나만 그린다 - GT는 표시하지 않음."""
    rgb = np.stack([image_2d] * 3, axis=-1).astype(np.uint8)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    min_row, min_col, max_row, max_col = bbox
    draw.rectangle([min_col, min_row, max_col, max_row], outline=color, width=2)
    return pil_img


def crop_zoomed(image_2d: np.ndarray, bbox: tuple, margin: int = 10) -> Image.Image:
    min_row, min_col, max_row, max_col = bbox
    h, w = image_2d.shape
    r0, r1 = max(0, min_row - margin), min(h, max_row + margin)
    c0, c1 = max(0, min_col - margin), min(w, max_col + margin)
    return Image.fromarray(image_2d[r0:r1, c0:c1].astype(np.uint8))


def build_slice_samples(case_id: str, slice_idx: int, image_2d: np.ndarray,
                         pred_2d: np.ndarray, gt_2d: np.ndarray, out_dir: Path,
                         class_names: dict = CLASS_NAMES) -> tuple[list[dict], int]:
    """
    한 슬라이스의 5cls pred/gt를 인스턴스 매칭해 샘플 목록을 만들고,
    pred 인스턴스별 bbox overlay/crop 이미지를 out_dir/images 아래에 저장한다.
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
        overlay_img = draw_single_bbox_overlay(windowed, region["bbox"])
        overlay_path = out_dir / "images" / f"{case_id}_s{slice_idx:03d}_inst{region['instance_id']}_overlay.png"
        overlay_img.save(overlay_path)

        crop_img = crop_zoomed(windowed, region["bbox"])
        crop_path = out_dir / "images" / f"{case_id}_s{slice_idx:03d}_inst{region['instance_id']}_crop.png"
        crop_img.save(crop_path)

        pred_class_id = region["class_id"]
        gt_class_id = region["gt_class_id"]

        samples.append({
            "id": f"{case_id}_s{slice_idx:03d}_inst{region['instance_id']}",
            "case_id": case_id,
            "slice_idx": slice_idx,
            "bbox": list(region["bbox"]),
            "area_px": region["area_px"],
            "pred_class_id": pred_class_id,
            "pred_class_name": class_names[pred_class_id],
            "is_lesion": region["is_lesion"],
            "gt_class_id": gt_class_id,
            "gt_class_name": class_names[gt_class_id] if gt_class_id else None,
            "class_correct": (pred_class_id == gt_class_id) if region["is_lesion"] else None,
            "images": {
                "overlay": str(overlay_path),
                "cropped": str(crop_path),
            },
        })

    return samples, len(fn_only)


def build_dataset(pred_dir: Path, gt_dir: Path, images_dir: Path, out_dir: Path,
                   class_names: dict = CLASS_NAMES) -> tuple[list[dict], list[dict]]:
    """
    pred_dir의 케이스마다 gt_dir/images_dir에서 짝을 찾아 build_slice_samples를 슬라이스별로 실행.
    반환: (전체 샘플 목록, FN 발생 슬라이스 로그)
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

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
