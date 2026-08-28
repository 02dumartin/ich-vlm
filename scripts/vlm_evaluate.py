# vlm_is_lesion.py + vlm_lesion_subtype.py가 채운 단일 결과 JSON 하나로 평가:
#   1) VLM_vs_GT / VLM_vs_pred / GT_vs_pred is_lesion(T/F) confusion + accuracy
#      (subtype까지 돌았으면 같은 세 조합에 lesion_type confusion + accuracy도 추가)
#   2) GT와 다른 VLM 오답 목록 ({tag}_errors.csv)
#   3) (선택) VLM 보정을 nnUNet 5cls 예측 볼륨에 반영해 nii.gz로 저장하고 nnUNet 공식
#      evaluate_folder로 재평가(mDice/mIoU/mRecall/mPrecision) - subtype 없으면 2cls, 있으면 5cls
# 위 1)+3)을 합쳐 {tag}_summary.json 하나로 저장한다.
#
# 사용 예:
#   python scripts/vlm_evaluate.py \
#     --results-json /path/to/output/full_results.json \
#     --out-dir /path/to/output --tag full \
#     --pred-5cls-dir /path/to/nnUNet_predictions/.../MBH-Seg25 \
#     --gt-5cls-dir /path/to/test_data/Test1_MBHSeg25/labelsTs

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.classify.evaluate import run_detection_eval, show_vlm_errors
from src.vlm.classify.segmentation_eval import run_segmentation_eval


def parse_arguments():
    parser = argparse.ArgumentParser(description="VLM 분류 결과 평가 (detect confusion/accuracy + 2cls Dice)")
    parser.add_argument("--results-json", required=True, type=Path,
                         help="vlm_is_lesion.py + vlm_lesion_subtype.py가 채운 단일 결과 JSON")
    parser.add_argument("--out-dir", required=True, type=Path, help="summary/오답 저장 위치")
    parser.add_argument("--tag", default="full", help="출력 파일명 접두어")
    parser.add_argument("--pred-5cls-dir", type=Path, default=None, help="nnUNet 5cls 예측 nii.gz 폴더 (2cls Dice 재계산용)")
    parser.add_argument("--gt-5cls-dir", type=Path, default=None, help="5cls GT label nii.gz 폴더 (2cls Dice 재계산용)")
    parser.add_argument("--skip-seg-eval", action="store_true", help="2cls Dice 재계산을 건너뜀")
    return parser.parse_args()


def main():
    args = parse_arguments()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results_json, encoding="utf-8") as f:
        records = json.load(f)

    summary = {"detect": run_detection_eval(records)}

    errors = show_vlm_errors(records)
    errors.drop(columns=["images"]).to_csv(args.out_dir / f"{args.tag}_errors.csv", index=False)
    print(f"저장: {args.out_dir / f'{args.tag}_errors.csv'} ({len(errors)}개)")

    if not args.skip_seg_eval:
        if args.pred_5cls_dir is None or args.gt_5cls_dir is None:
            print("[안내] --pred-5cls-dir/--gt-5cls-dir이 없어 2cls Dice 재계산을 건너뜁니다 (--skip-seg-eval로 명시 가능)")
        else:
            summary["segmentation"] = run_segmentation_eval(records, args.pred_5cls_dir, args.gt_5cls_dir, args.out_dir)

    summary_json = args.out_dir / f"{args.tag}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"저장: {summary_json}")


if __name__ == "__main__":
    main()
