"""
VLM 분류 결과 평가: GT/pred(nnUNet=seg)/VLM 세 판정 중 어느 조합이든 같은 함수로 비교한다.
vlm_is_lesion.py + vlm_lesion_subtype.py가 채운 단일 결과 JSON(GT와 VLM 예측이 이미
같은 레코드에 다 들어있음)을 그대로 받는다 — 더 이상 별도 GT 파일과 join할 필요 없음.
"""

import numpy as np
import pandas as pd
from PIL import Image


def with_nnunet_is_lesion(records: list[dict]) -> pd.DataFrame:
    """nnUNet(Reader#1=seg) 예측 인스턴스는 전부 '병변'이라고 주장한 것이므로,
    seg vs GT/VLM 비교용으로 상수 컬럼 seg_is_lesion=True를 추가 (저장되는 필드는 아님)."""
    df = pd.DataFrame(records)
    df["seg_is_lesion"] = True
    return df


def _bool_confusion(df: pd.DataFrame, subj_col: str, ref_col: str) -> dict:
    """subj_col/ref_col(둘 다 bool)의 2x2 confusion을 {ref_value: {subj_value: count}} 형태로.
    예: {"True": {"True": tp, "False": fn}, "False": {"True": fp, "False": tn}}"""
    confusion = pd.crosstab(df[ref_col], df[subj_col])
    confusion.index = confusion.index.astype(str)
    confusion.columns = confusion.columns.astype(str)
    return confusion.to_dict(orient="index")


def _class_confusion(df: pd.DataFrame, subj_col: str, ref_col: str) -> dict:
    """subj_col/ref_col(범주형, 예: lesion_type)의 NxN confusion을 {ref_value: {subj_value: count}} 형태로."""
    confusion = pd.crosstab(df[ref_col], df[subj_col])
    return confusion.to_dict(orient="index")


def _precision_recall_binary(confusion: dict) -> tuple[float, float]:
    """_bool_confusion 결과({"True":{"True":tp,"False":fn}, "False":{"True":fp,"False":tn}})에서
    True를 양성으로 두고 precision/recall 계산. 해당 셀이 없으면(예: seg_is_lesion처럼 한쪽 값이
    상수라 크로스탭에 한 축만 있는 경우) 0으로 취급."""
    tp = confusion.get("True", {}).get("True", 0)
    fn = confusion.get("True", {}).get("False", 0)
    fp = confusion.get("False", {}).get("True", 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def _macro_precision_recall_class(confusion: dict) -> tuple[float, float]:
    """_class_confusion 결과(NxN, {ref_value: {subj_value: count}})에서 클래스별 precision/recall을
    구해 macro-average(클래스 단순 평균)한다."""
    classes = sorted(set(confusion.keys()) | {c for row in confusion.values() for c in row.keys()})
    precisions, recalls = [], []
    for c in classes:
        tp = confusion.get(c, {}).get(c, 0)
        row_total = sum(confusion.get(c, {}).values())  # ref=c인 전체(=recall 분모)
        col_total = sum(confusion.get(r, {}).get(c, 0) for r in confusion)  # subj=c인 전체(=precision 분모)
        if col_total > 0:
            precisions.append(tp / col_total)
        if row_total > 0:
            recalls.append(tp / row_total)
    return float(np.mean(precisions)) if precisions else float("nan"), \
        float(np.mean(recalls)) if recalls else float("nan")


def run_detection_eval(records: list[dict]) -> dict:
    """VLM_vs_GT / VLM_vs_pred / GT_vs_pred 세 조합의 is_lesion(T/F) confusion + accuracy를 계산.
    vlm2_lesion_type이 있으면(subtype 단계까지 돈 경우) 같은 세 조합에 lesion_type confusion +
    accuracy도 추가한다. 반환값을 그대로 {tag}_summary.json의 "detect" 섹션에 넣는다."""
    df = with_nnunet_is_lesion(records)
    has_subtype = "vlm2_lesion_type" in df.columns

    pairs = [
        ("VLM_vs_GT", "vlm1_is_lesion", "gt_is_lesion", "vlm2_lesion_type", "gt_lesion_type"),
        ("VLM_vs_pred", "vlm1_is_lesion", "seg_is_lesion", "vlm2_lesion_type", "seg_lesion_type"),
        ("GT_vs_pred", "gt_is_lesion", "seg_is_lesion", "gt_lesion_type", "seg_lesion_type"),
    ]
    detect = {}
    for pair_name, subj_col, ref_col, subj_cls_col, ref_cls_col in pairs:
        accuracy = float((df[subj_col] == df[ref_col]).mean())
        confusion = _bool_confusion(df, subj_col, ref_col)
        precision, recall = _precision_recall_binary(confusion)
        entry = {"n": len(df), "accuracy": accuracy, "precision": precision, "recall": recall,
                 "confusion": confusion}
        print(f"[{pair_name}] is_lesion acc={accuracy:.4f} precision={precision:.4f} recall={recall:.4f}  (n={len(df)})")

        if has_subtype:
            agree = df[df[ref_col] & df[subj_col]]
            if not agree.empty:
                cls_acc = float((agree[subj_cls_col] == agree[ref_cls_col]).mean())
                cls_confusion = _class_confusion(agree, subj_cls_col, ref_cls_col)
                cls_precision, cls_recall = _macro_precision_recall_class(cls_confusion)
                entry["lesion_type_n"] = len(agree)
                entry["lesion_type_accuracy"] = cls_acc
                entry["lesion_type_precision"] = cls_precision
                entry["lesion_type_recall"] = cls_recall
                entry["lesion_type_confusion"] = cls_confusion
                print(f"[{pair_name}] lesion_type acc={cls_acc:.4f} precision(macro)={cls_precision:.4f} "
                      f"recall(macro)={cls_recall:.4f}  (n={len(agree)})")

        detect[pair_name] = entry

    return detect


def show_vlm_errors(records: list[dict], max_display: int = None) -> pd.DataFrame:
    """GT와 다른 VLM 판정만 골라서 반환. max_display를 주면 crop 이미지까지 출력(노트북 전용).
    vlm2_lesion_type이 없으면(subtype 단계를 안 돌린 경우) is_lesion만으로 오답을 가른다."""
    df = pd.DataFrame(records)

    if "vlm2_lesion_type" in df.columns:
        gt_label = np.where(df["gt_is_lesion"], df["gt_lesion_type"], "NONE")
        vlm_label = np.where(df["vlm1_is_lesion"], df["vlm2_lesion_type"], "NONE")
    else:
        gt_label = df["gt_is_lesion"].astype(str)
        vlm_label = df["vlm1_is_lesion"].astype(str)
    is_wrong = gt_label != vlm_label
    errors = df[is_wrong].copy()
    errors["gt_label"] = gt_label[is_wrong]
    errors["vlm_label"] = vlm_label[is_wrong]

    print(f"오답(vs GT) {len(errors)}개 / 전체 {len(df)}개")

    if max_display:
        for _, r in errors.head(max_display).iterrows():
            print(f"\n{r['id']}  Reader1(seg)={r['seg_lesion_type']}  GT={r['gt_label']}  VLM={r['vlm_label']}")
            Image.open(r["images"]["crop_min"]).show()

    return errors
