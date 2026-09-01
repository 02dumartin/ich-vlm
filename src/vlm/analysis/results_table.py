"""
crop_mode별 full_summary.json/full_results.json을 읽어 results_table.md에 들어갈 두 표를
markdown으로 만든다:

1. 지표 표: crop_mode x (is lesion/lesion type) 단계별 Segmentation mDice/mIoU/mPrecision/mRecall +
   Detection Acc/Precision/Recall. mIoU/mPrecision 모두 새 계산 없이 full_summary.json에서 바로
   뽑는다 - segmentation_eval.py의 MACRO_METRICS = ("Dice", "IoU", "Recall", "Precision")에
   이미 다 들어있어서(2cls는 lesion_IoU/lesion_Precision, 5cls는 summary_to_row가
   add_macro_average로 만든 mIoU/mPrecision을 그대로 씀).
2. FP/TP 제거율 표: full_results.json의 인스턴스 단위 gt_is_lesion/vlm1_is_lesion을 직접 세서,
   진짜 FP(gt_is_lesion=False)와 진짜 병변(gt_is_lesion=True) 각각을 VLM이 얼마나 지웠는지
   비율로 보여준다. precision 하나로는 "잘 골라냈다"와 "그냥 다 아니라고 했다"를 구분 못 해서
   (예: 260826 crop_min_contour에서 FP 97.7%/진짜 병변 82.4%를 둘 다 비슷하게 지움 - precision은
   높게 나오지만 실제로는 판별력이 낮은 것) 이 표로 직접 대조한다.

사용 예:
    python scripts/vlm_results_table.py \\
        --exp-dir experiments/260826_vlm-is-lesion-crop-sweep-mbhseg25-5cls_v01
"""

import json
from pathlib import Path

from src.vlm.image_crop import CROP_MODES


def _metrics_row(summary: dict) -> dict:
    """full_summary.json 하나에서 is lesion(2cls)/lesion type(5cls) 두 행을 뽑는다."""
    g = summary["detect"]["VLM_vs_GT"]
    seg = summary["segmentation"]
    s2 = seg["2cls"]["vlm_corrected"] if seg.get("2cls") else {}
    s5 = seg["5cls"]["vlm_corrected"] if seg.get("5cls") else {}

    is_lesion = {
        "seg_dice": s2.get("lesion_Dice"), "seg_iou": s2.get("lesion_IoU"),
        "seg_precision": s2.get("lesion_Precision"), "seg_recall": s2.get("lesion_Recall"),
        "det_acc": g.get("accuracy"), "det_precision": g.get("precision"), "det_recall": g.get("recall"),
    }
    lesion_type = {
        "seg_dice": s5.get("mDice"), "seg_iou": s5.get("mIoU"),
        "seg_precision": s5.get("mPrecision"), "seg_recall": s5.get("mRecall"),
        "det_acc": g.get("lesion_type_accuracy"), "det_precision": g.get("lesion_type_precision"),
        "det_recall": g.get("lesion_type_recall"),
    } if seg.get("5cls") else None
    return {"is_lesion": is_lesion, "lesion_type": lesion_type}


def _removal_rates(records: list[dict]) -> dict:
    """gt_is_lesion=False(진짜 FP)/True(진짜 병변) 각각에서 VLM이 지운 비율."""
    gt_false = [r for r in records if not r["gt_is_lesion"]]
    gt_true = [r for r in records if r["gt_is_lesion"]]

    def _rate(group):
        removed = sum(1 for r in group if not r["vlm1_is_lesion"])
        n = len(group)
        return {"n": n, "removed": removed, "removed_pct": removed / n if n else float("nan")}

    return {"fp": _rate(gt_false), "tp": _rate(gt_true)}


def build_tables(exp_dir: Path, crop_modes: tuple = CROP_MODES, tag: str = "full") -> dict:
    """crop_mode별로 metrics_row/removal_rates를 계산해 반환. results_table.md 작성 시 참고용."""
    result = {}
    for mode in crop_modes:
        out_dir = exp_dir / "output" / mode
        summary = json.loads((out_dir / f"{tag}_summary.json").read_text())
        records = json.loads((out_dir / f"{tag}_results.json").read_text())
        result[mode] = {"metrics": _metrics_row(summary), "removal": _removal_rates(records)}
    return result


def render_metrics_table(tables: dict) -> str:
    lines = ["| crop_mode | 단계 | Segmentation mDice | Segmentation mIoU | Segmentation mPrecision | "
             "Segmentation mRecall | Detection Acc | Detection Precision | Detection Recall |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mode, t in tables.items():
        for stage_key, stage_label in [("is_lesion", "is lesion (2cls)"), ("lesion_type", "lesion type (5cls)")]:
            row = t["metrics"][stage_key]
            if row is None:
                continue
            lines.append(
                f"| {mode} | {stage_label} | {row['seg_dice']:.3f} | {row['seg_iou']:.3f} | "
                f"{row['seg_precision']:.3f} | {row['seg_recall']:.3f} | {row['det_acc']:.3f} | "
                f"{row['det_precision']:.3f} | {row['det_recall']:.3f} |"
            )
    return "\n".join(lines)


def render_removal_rate_table(tables: dict) -> str:
    lines = ["| crop_mode | 진짜 FP 개수 | VLM이 지운 FP | FP 제거율 | 진짜 병변 개수 | "
             "VLM이 지운 병변 | 병변 오제거율 |",
             "|---|---|---|---|---|---|---|"]
    for mode, t in tables.items():
        fp, tp = t["removal"]["fp"], t["removal"]["tp"]
        lines.append(
            f"| {mode} | {fp['n']} | {fp['removed']} | {fp['removed_pct']:.1%} | "
            f"{tp['n']} | {tp['removed']} | {tp['removed_pct']:.1%} |"
        )
    return "\n".join(lines)
