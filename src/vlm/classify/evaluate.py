"""
VLM 분류 결과 평가: GT/pred(nnUNet)/VLM 세 판정 중 어느 조합이든 같은 함수로 비교한다.
vlm_classifier_5cls.ipynb의 compare_judgment/summarize_vlm_5cls/show_vlm_errors 로직 이식.
"""

import numpy as np
import pandas as pd
from PIL import Image


def with_ground_truth(pred_records: list[dict], gt_by_id: dict) -> list[dict]:
    """id로 5cls_dataset.json(GT) 레코드와 join해서 검증 전용 뷰 생성."""
    rows = []
    for r in pred_records:
        gt = gt_by_id[r["id"]]
        rows.append({
            **r,
            "gt_is_lesion": gt["is_lesion"],
            "gt_class_id": gt["gt_class_id"],
            "gt_class_name": gt["gt_class_name"],
            "nnunet_correct": gt["class_correct"],  # 참고용: nnUNet 자체가 맞았는지
        })
    return rows


def compare_judgment(df: pd.DataFrame, subj_is_lesion_col: str, ref_is_lesion_col: str,
                      subj_label: str, ref_label: str,
                      subj_class_col: str = None, ref_class_col: str = None) -> float:
    """
    subj_*를 ref_*(기준)과 비교. VLM/GT/Pred 어느 조합이든 이 함수 하나로 통일해서 쓴다.
    subj_class_col/ref_class_col을 둘 다 주면 병변 유무 판정에 더해 class 일치율/confusion까지 계산
    (is_lesion 단계만 비교할 땐 생략).
    """
    detect_acc = (df[subj_is_lesion_col] == df[ref_is_lesion_col]).mean()
    print(f"[{subj_label} vs {ref_label}] 병변 유무 판정 일치율: {detect_acc:.4f}  (n={len(df)})")

    if subj_class_col and ref_class_col:
        agree = df[df[ref_is_lesion_col] & df[subj_is_lesion_col]]
        if not agree.empty:
            cls_acc = (agree[subj_class_col] == agree[ref_class_col]).mean()
            print(f"[{subj_label} vs {ref_label}] 클래스 일치율(둘 다 병변으로 본 것 중): {cls_acc:.4f}  (n={len(agree)})")

            confusion = pd.crosstab(agree[ref_class_col], agree[subj_class_col],
                                     rownames=[ref_label], colnames=[subj_label])
            per_class = agree.groupby(ref_class_col).apply(lambda g: (g[subj_class_col] == g.name).mean())
            print(f"[{subj_label} vs {ref_label}] 클래스별 정확도:")
            print(per_class.round(4))
            return detect_acc, confusion

    return detect_acc, None


def show_vlm_errors(pred_records: list[dict], gt_by_id: dict, max_display: int = None) -> pd.DataFrame:
    """GT와 다른 VLM 판정만 골라서 반환. max_display를 주면 crop 이미지까지 출력(노트북 전용)."""
    df = pd.DataFrame(with_ground_truth(pred_records, gt_by_id))

    gt_label = np.where(df["gt_is_lesion"], df["gt_class_name"], "NONE")
    vlm_label = np.where(df["vlm_is_lesion"], df.get("vlm_lesion_type", "NONE"), "NONE")
    is_wrong = gt_label != vlm_label
    errors = df[is_wrong].copy()
    errors["gt_label"] = gt_label[is_wrong]
    errors["vlm_label"] = vlm_label[is_wrong]

    print(f"오답(vs GT) {len(errors)}개 / 전체 {len(df)}개")

    if max_display:
        for _, r in errors.head(max_display).iterrows():
            print(f"\n{r['id']}  Reader1(pred)={r['pred_class_name']}  GT={r['gt_label']}  VLM={r['vlm_label']}")
            Image.open(r["images"]["cropped"]).show()

    return errors
