# GT(5cls) / nnUNet 예측(5cls, thr0.001, pp) / VLM reader 최종 결과(5cls, full_corrected_nii_5cls)
# 를 케이스별로, GT/예측/VLM reader 중 하나라도 마스크가 있는 모든 z 슬라이스에 대해
# 1x3 그리드 PNG로 비교 오버레이한다(완전히 빈 슬라이스는 제외).
# vlm-nii-dir을 바꿔 끼우면 어느 실험/crop_mode의 VLM 결과든 동일하게 그릴 수 있다.
#
# 사용 예:
#   python scripts/vlm_reader_overlay.py \
#     --images-dir nnUNet/nnUNet_raw/Dataset001_MBHSeg25/imagesTs \
#     --gt-dir nnUNet/test_data/Test1_MBHSeg25/labelsTs \
#     --pred-dir exp/260821_nnunet-infer-5cls-thr-sweep-mbhseg25/output/thr0.001/pp \
#     --vlm-nii-dir exp/260831_vlm-is-lesion-crop-sweep-mbhseg25-5cls/output/crop_min_contour/full_corrected_nii_5cls \
#     --vlm-label crop_min_contour_zeroshot \
#     --out-dir exp/260831_vlm-is-lesion-crop-sweep-mbhseg25-5cls/output/crop_min_contour/reader_overlay

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.analysis.overlay import (
    load_reader_case_volumes,
    plot_slice_reader_grid,
    slices_with_any_mask,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="GT/nnUNet 예측/VLM reader 결과 1x3 오버레이 생성")
    parser.add_argument("--images-dir", required=True, type=Path, help="원본 CT(imagesTs, *_0000.nii.gz)")
    parser.add_argument("--gt-dir", required=True, type=Path, help="5cls GT labelsTs")
    parser.add_argument("--pred-dir", required=True, type=Path, help="nnUNet 예측(5cls, thr0.001, pp)")
    parser.add_argument("--vlm-nii-dir", required=True, type=Path, help="VLM reader 최종 결과(full_corrected_nii_5cls)")
    parser.add_argument("--vlm-label", required=True, help="[0,2] 패널 제목에 붙일 VLM reader 구분 이름")
    parser.add_argument("--out-dir", required=True, type=Path, help="PNG 저장 경로")
    parser.add_argument("--case-ids", nargs="*", default=None, help="지정 시 이 케이스들만 생성(미지정이면 전체)")
    return parser.parse_args()


def match_case_ids(gt_dir: Path, pred_dir: Path, vlm_nii_dir: Path) -> list[str]:
    gt_cases = {p.stem.replace(".nii", "") for p in gt_dir.glob("*.nii.gz")}
    pred_cases = {p.stem.replace(".nii", "") for p in pred_dir.glob("*.nii.gz")}
    vlm_cases = {p.stem.replace(".nii", "") for p in vlm_nii_dir.glob("*.nii.gz")}
    case_ids = sorted(gt_cases & pred_cases & vlm_cases)
    missing = sorted(gt_cases - (pred_cases & vlm_cases))
    if missing:
        print(f"예측/VLM 결과 없어서 제외된 케이스: {len(missing)}개 (예: {missing[:5]})")
    return case_ids


def main():
    args = parse_arguments()
    case_ids = match_case_ids(args.gt_dir, args.pred_dir, args.vlm_nii_dir)
    if args.case_ids:
        wanted = set(args.case_ids)
        selected = [(i, c) for i, c in enumerate(case_ids) if c in wanted]
    else:
        selected = list(enumerate(case_ids))

    print(f"매칭 케이스 {len(case_ids)}개 중 {len(selected)}개 생성")

    total_images = 0
    for case_idx, case_id in selected:
        img_path = args.images_dir / f"{case_id}_0000.nii.gz"
        gt_path = args.gt_dir / f"{case_id}.nii.gz"
        pred_path = args.pred_dir / f"{case_id}.nii.gz"
        vlm_path = args.vlm_nii_dir / f"{case_id}.nii.gz"

        volumes = load_reader_case_volumes(img_path, gt_path, pred_path, vlm_path)
        slice_indices = slices_with_any_mask(volumes)
        if not slice_indices:
            print(f"[{case_idx}] {case_id} -> 마스크 있는 슬라이스 없음, 건너뜀")
            continue

        for slice_idx in slice_indices:
            out_path = plot_slice_reader_grid(
                case_id=case_id,
                case_idx=case_idx,
                slice_idx=slice_idx,
                img_path=img_path,
                gt_path=gt_path,
                pred_path=pred_path,
                vlm_path=vlm_path,
                vlm_label=args.vlm_label,
                out_dir=args.out_dir,
                volumes=volumes,
            )
            total_images += 1
        print(f"[{case_idx}] {case_id} -> 슬라이스 {len(slice_indices)}장 생성")

    print(f"총 {total_images}장 생성 완료")


if __name__ == "__main__":
    main()
