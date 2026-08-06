"""스캔(케이스) 단위 평가 CSV 생성.

nnUNet_test_5cls.py/nnUNet_test_1cls.py가 만든 summary.json의 `metric_per_case`
(케이스 하나하나의 volume 전체 기준 raw metric)를 그대로 행으로 풀어서 스캔별 CSV 생서

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
    """metric_per_case의 한 항목에서 클래스별 Dice/IoU/Precision/Recall/n_ref 가져오기"""
    per_class = {}
    for cls, name in class_names.items():
        m = case["metrics"].get(str(cls))
        if m is None:
            per_class[name] = {"dice": float("nan"), "iou": float("nan"),
                                "precision": float("nan"), "recall": float("nan"), "n_ref": 0}
            continue

        dice = m.get("Dice")
        dice = float(dice) if dice is not None and not (isinstance(dice, float) and np.isnan(dice)) else float("nan")
        iou = m.get("IoU")
        iou = float(iou) if iou is not None and not (isinstance(iou, float) and np.isnan(iou)) else float("nan")

        precision = m.get("Precision")
        if precision is None and m.get("TP") is not None and m.get("FP") is not None:
            tp, fp = m["TP"], m["FP"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else None
        precision = float(precision) if precision is not None else float("nan")

        recall = m.get("Recall")
        if recall is None and m.get("TP") is not None and m.get("FN") is not None:
            tp, fn = m["TP"], m["FN"]
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
        recall = float(recall) if recall is not None else float("nan")

        n_ref = m.get("n_ref", 0) or 0

        per_class[name] = {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "n_ref": n_ref}
    return per_class


def build_scan_df(summary_path: Path, test_name: str, row_label: str, class_names: dict,
                   multiclass: bool) -> pd.DataFrame:
    """multiclass=True면 mDice/mIoU + 클래스별 Dice/IoU/n_ref 컬럼(5cls류)
    multiclass=False면 클래스가 1개뿐이라는 뜻이므로 Dice/IoU/Precision/Recall을 접두어 없이 평평하게(1cls류)."""
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
                row[f"{name}_n_ref"] = v["n_ref"]
        else:
            (_, v), = per_class.items()
            row["Dice"] = v["dice"]
            row["IoU"] = v["iou"]
            row["Precision"] = v["precision"]
            row["Recall"] = v["recall"]

        rows.append(row)

    return pd.DataFrame(rows)
