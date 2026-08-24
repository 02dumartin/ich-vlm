"""스캔(케이스) 단위 평가 CSV 생성.

nnUNet_infer_5cls.py/nnUNet_infer_2cls.py가 만든 
summary.json의 `metric_per_case`
(케이스 하나하나의 volume 전체 기준 raw metric)를 그대로 행으로 풀어서 스캔별 CSV 생성

volume 전체를 GT와 비교
scripts/nnUNet_slice_eval.py
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def get_n_slices(nifti_path: Path) -> int:
    """z축(3번째 축) 슬라이스 수. NIfTI 관례상 axis order가 (X, Y, Z)라 가정"""
    img = nib.load(str(nifti_path))
    return int(img.shape[2])


def case_id_from_path(path: Path) -> str:
    return path.name.replace(".nii.gz", "").replace(".nii", "")


def extract_class_metrics(case: dict, class_names: dict) -> dict:
    """
    nnUNet_predictions/.../summary.json의
    metric_per_case의 한 항목에서 클래스별 Dice/IoU/Precision/Recall/TP/FP/FN/n_ref 추출
    """
    per_class = {}
    for cls, name in class_names.items():
        m = case["metrics"].get(str(cls))
        if m is None:
            per_class[name] = {"dice": float("nan"), "iou": float("nan"),
                                "precision": float("nan"), "recall": float("nan"),
                                "tp": 0, "fp": 0, "fn": 0, "n_ref": 0}
            continue

        dice = m.get("Dice")
        dice = float(dice) if dice is not None and not (isinstance(dice, float) and np.isnan(dice)) else float("nan")
        iou = m.get("IoU")
        iou = float(iou) if iou is not None and not (isinstance(iou, float) and np.isnan(iou)) else float("nan")

        tp = m.get("TP", 0) or 0
        fp = m.get("FP", 0) or 0
        fn = m.get("FN", 0) or 0

        precision = m.get("Precision")
        if precision is None and m.get("TP") is not None and m.get("FP") is not None:
            precision = tp / (tp + fp) if (tp + fp) > 0 else None
        precision = float(precision) if precision is not None else float("nan")

        recall = m.get("Recall")
        if recall is None and m.get("TP") is not None and m.get("FN") is not None:
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
        recall = float(recall) if recall is not None else float("nan")

        n_ref = m.get("n_ref", 0) or 0

        per_class[name] = {"dice": dice, "iou": iou, "precision": precision, "recall": recall,
                            "tp": tp, "fp": fp, "fn": fn, "n_ref": n_ref}
    return per_class


def build_scan_df(summary_path: Path, test_name: str, row_label: str, class_names: dict,
                   multiclass: bool) -> pd.DataFrame:
    """
    multiclass=True면 mDice/mIoU + 클래스별 Dice/IoU/n_ref 컬럼(5cls류)
    multiclass=False면 클래스가 1개뿐이라는 뜻이므로 Dice/IoU/Precision/Recall을 접두어 없이 평평하게(2cls류)
    nnUNet_predictions/.../summary.json의 metric_per_case를 행으로 풀어서 스캔별 CSV 생성
    """
    with open(summary_path) as f:
        data = json.load(f)

    rows = []
    for case in data["metric_per_case"]:
        ref_path = Path(case["reference_file"])
        case_id = case_id_from_path(Path(case["prediction_file"]))
        n_slice = get_n_slices(ref_path)
        per_class = extract_class_metrics(case, class_names)

        row = {"Test Dataset": test_name, "Dimension": row_label, "case_id": case_id, "n_slice": n_slice}

        if multiclass:
            dice_vals = [v["dice"] for v in per_class.values()]
            iou_vals = [v["iou"] for v in per_class.values()]
            row["mDice"] = float(np.nanmean(dice_vals)) if dice_vals else float("nan")
            row["mIoU"] = float(np.nanmean(iou_vals)) if iou_vals else float("nan")
            for name, v in per_class.items():
                row[f"{name}_Dice"] = v["dice"]
                row[f"{name}_IoU"] = v["iou"]
                row[f"{name}_TP"] = v["tp"]
                row[f"{name}_FP"] = v["fp"]
                row[f"{name}_FN"] = v["fn"]
                row[f"{name}_n_ref"] = v["n_ref"]
        else:
            (_, v), = per_class.items()
            row["Dice"] = v["dice"]
            row["IoU"] = v["iou"]
            row["Precision"] = v["precision"]
            row["Recall"] = v["recall"]
            row["TP"] = v["tp"]
            row["FP"] = v["fp"]
            row["FN"] = v["fn"]

        rows.append(row)

    return pd.DataFrame(rows)


def compute_slice_class_metrics(pred_slice: np.ndarray, gt_slice: np.ndarray, class_names: dict) -> dict:
    """
    2D 슬라이스 한 장에서 클래스별 TP/FP/FN(픽셀 카운트)과 그로부터 계산한 Dice/IoU 반환.
    n_ref=0이고 예측도 없으면(TP=FP=FN=0) Dice/IoU는 NaN(정의 불가), 그 외에는
    Dice=2TP/(2TP+FP+FN), IoU=TP/(TP+FP+FN) — nnUNet summary.json과 동일한 컨벤션.
    """
    per_class = {}
    for cls, name in class_names.items():
        gt_mask = gt_slice == cls
        pred_mask = pred_slice == cls

        tp = int(np.logical_and(gt_mask, pred_mask).sum())
        fp = int(np.logical_and(~gt_mask, pred_mask).sum())
        fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
        n_ref = int(gt_mask.sum())

        denom_dice = 2 * tp + fp + fn
        dice = float("nan") if denom_dice == 0 else 2 * tp / denom_dice
        denom_iou = tp + fp + fn
        iou = float("nan") if denom_iou == 0 else tp / denom_iou

        per_class[name] = {"tp": tp, "fp": fp, "fn": fn, "n_ref": n_ref, "dice": dice, "iou": iou}

    return per_class


def classify_slice_error_type(agg_tp: int, agg_fp: int, agg_fn: int) -> str:
    """클래스 무관 집계 TP/FP/FN으로 슬라이스 에러 유형 분류.
    TN(둘 다 없음) / TP_clean(FP·FN 없이 완전히 맞음) / FP_only / FN_only / mixed(부분 겹침·클래스 혼동 등)."""
    if agg_tp == 0 and agg_fp == 0 and agg_fn == 0:
        return "TN"
    if agg_tp > 0 and agg_fp == 0 and agg_fn == 0:
        return "TP_clean"
    if agg_fp > 0 and agg_tp == 0 and agg_fn == 0:
        return "FP_only"
    if agg_fn > 0 and agg_tp == 0 and agg_fp == 0:
        return "FN_only"
    return "mixed"


def build_scan_analysis_df(summary_path: Path, thr: str, class_names: dict) -> pd.DataFrame:
    """
    스캔(볼륨) 단위 threshold 분석용 DataFrame.
    summary.json(threshold 하나에 대응) metric_per_case에서 클래스별
    Dice/IoU/Precision/Recall/TP/FP/FN/n_ref + thr/gt_has_lesion/n_classes_present까지 포함.
    build_scan_df와 달리 Precision/Recall도 남기고, threshold sweep 비교를 위해 thr 컬럼을 붙인다.
    """
    with open(summary_path) as f:
        data = json.load(f)

    rows = []
    for case in data["metric_per_case"]:
        ref_path = Path(case["reference_file"])
        case_id = case_id_from_path(Path(case["prediction_file"]))
        n_slice = get_n_slices(ref_path)
        per_class = extract_class_metrics(case, class_names)

        dice_vals = [v["dice"] for v in per_class.values()]
        iou_vals = [v["iou"] for v in per_class.values()]
        n_classes_present = sum(1 for v in per_class.values() if v["n_ref"] > 0)

        row = {
            "case_id": case_id,
            "thr": thr,
            "n_slice": n_slice,
            "mDice": float(np.nanmean(dice_vals)) if dice_vals else float("nan"),
            "mIoU": float(np.nanmean(iou_vals)) if iou_vals else float("nan"),
            "gt_has_lesion": n_classes_present > 0,
            "n_classes_present": n_classes_present,
        }
        for name, v in per_class.items():
            row[f"{name}_Dice"] = v["dice"]
            row[f"{name}_IoU"] = v["iou"]
            row[f"{name}_Precision"] = v["precision"]
            row[f"{name}_Recall"] = v["recall"]
            row[f"{name}_TP"] = v["tp"]
            row[f"{name}_FP"] = v["fp"]
            row[f"{name}_FN"] = v["fn"]
            row[f"{name}_n_ref"] = v["n_ref"]

        rows.append(row)

    return pd.DataFrame(rows)


def build_slice_analysis_df(pred_dir: Path, gt_dir: Path, thr: str, class_names: dict) -> pd.DataFrame:
    """
    슬라이스 단위 threshold 분석용 DataFrame.
    summary.json에는 슬라이스 단위 지표가 없어서, pred/GT nii.gz를 직접 읽어
    z-slice마다 클래스별 TP/FP/FN/Dice/IoU + gt/pred_has_lesion, 전경 픽셀 수, error_type,
    slice_pos(0~1 정규화 z 위치)까지 계산한다. pred_dir의 각 케이스에 대해 gt_dir에서
    동일 case_id의 GT를 찾아 짝짓는다(없으면 건너뜀).
    """
    rows = []
    pred_paths = sorted(Path(pred_dir).glob("*.nii.gz"))

    for pred_path in pred_paths:
        case_id = case_id_from_path(pred_path)
        gt_path = Path(gt_dir) / f"{case_id}.nii.gz"
        if not gt_path.exists():
            continue

        pred_vol = np.asarray(nib.load(str(pred_path)).dataobj).astype(np.uint8)
        gt_vol = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.uint8)
        n_slice = pred_vol.shape[2]

        for z in range(n_slice):
            pred_slice = pred_vol[:, :, z]
            gt_slice = gt_vol[:, :, z]

            per_class = compute_slice_class_metrics(pred_slice, gt_slice, class_names)
            agg_tp = sum(v["tp"] for v in per_class.values())
            agg_fp = sum(v["fp"] for v in per_class.values())
            agg_fn = sum(v["fn"] for v in per_class.values())

            dice_vals = [v["dice"] for v in per_class.values()]
            iou_vals = [v["iou"] for v in per_class.values()]
            gt_area = int((gt_slice > 0).sum())
            pred_area = int((pred_slice > 0).sum())

            row = {
                "case_id": case_id,
                "slice_idx": z,
                "thr": thr,
                "slice_pos": z / (n_slice - 1) if n_slice > 1 else 0.0,
                "mDice": float(np.nanmean(dice_vals)) if dice_vals else float("nan"),
                "mIoU": float(np.nanmean(iou_vals)) if iou_vals else float("nan"),
                "gt_has_lesion": gt_area > 0,
                "pred_has_lesion": pred_area > 0,
                "gt_area_px": gt_area,
                "pred_area_px": pred_area,
                "error_type": classify_slice_error_type(agg_tp, agg_fp, agg_fn),
            }
            for name, v in per_class.items():
                row[f"{name}_Dice"] = v["dice"]
                row[f"{name}_IoU"] = v["iou"]
                row[f"{name}_TP"] = v["tp"]
                row[f"{name}_FP"] = v["fp"]
                row[f"{name}_FN"] = v["fn"]

            rows.append(row)

    return pd.DataFrame(rows)


def compute_slice_binary_metrics(pred_slice: np.ndarray, gt_slice: np.ndarray) -> dict:
    """클래스 무관, 병변 유무(전경 vs 배경) 기준 슬라이스 단위 TP/FP/FN.
    클래스가 틀려도 위치가 겹치면 TP로 센다 - 클래스 정확도보다 병변 존재 자체를 더 중요하게 볼 때 사용.
    (compute_slice_class_metrics는 클래스별로 엄격히 맞춰야 TP라, 클래스만 틀린 픽셀은 FN+FP로 이중 집계된다.)"""
    gt_fg = gt_slice > 0
    pred_fg = pred_slice > 0
    tp = int(np.logical_and(gt_fg, pred_fg).sum())
    fp = int(np.logical_and(~gt_fg, pred_fg).sum())
    fn = int(np.logical_and(gt_fg, ~pred_fg).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def build_slice_binary_df(pred_dir: Path, gt_dir: Path, thr: str) -> pd.DataFrame:
    """
    슬라이스 단위, 클래스 무관 이진(병변 유무) TP/FP/FN과 recall/fp_ratio를 담은 DataFrame.
    recall = tp/(tp+fn) (GT에 병변이 없으면 NaN), fp_ratio = fp/(tp+fp) (예측 전경이 없으면 NaN).
    """
    rows = []
    pred_paths = sorted(Path(pred_dir).glob("*.nii.gz"))

    for pred_path in pred_paths:
        case_id = case_id_from_path(pred_path)
        gt_path = Path(gt_dir) / f"{case_id}.nii.gz"
        if not gt_path.exists():
            continue

        pred_vol = np.asarray(nib.load(str(pred_path)).dataobj).astype(np.uint8)
        gt_vol = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.uint8)
        n_slice = pred_vol.shape[2]

        for z in range(n_slice):
            m = compute_slice_binary_metrics(pred_vol[:, :, z], gt_vol[:, :, z])
            tp, fp, fn = m["tp"], m["fp"], m["fn"]
            rows.append({
                "case_id": case_id,
                "slice_idx": z,
                "thr": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
                "fp_ratio": fp / (tp + fp) if (tp + fp) > 0 else float("nan"),
                "gt_area_px": tp + fn,
                "pred_area_px": tp + fp,
            })

    return pd.DataFrame(rows)
