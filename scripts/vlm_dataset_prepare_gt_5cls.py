# nnUNet 예측이 아니라 5cls GT 라벨 자체에서 VLM 학습/평가용 데이터셋 생성.
# Dataset001_MBHSeg25의 imagesTr/labelsTr, imagesTs/labelsTs로 train/test 데이터셋을
# 각각 만들 때 씀 - 예측이 없으니 seg_lesion_type 같은 예측 파생 필드가 없고,
# crop은 crop_min_contour 하나만(overlay + crop_min_contour 두 이미지).
#
# 슬라이스별 GT 인스턴스 bbox overlay, crop_min_contour 이미지를 --out-dir/images 아래에
# 저장하고 5cls_dataset.json을 --out-dir에 저장한다.
#
# 사용 예:
#   python scripts/vlm_dataset_prepare_gt_5cls.py \
#     --gt-dir nnUNet/nnUNet_raw/Dataset001_MBHSeg25/labelsTr \
#     --images-dir nnUNet/nnUNet_raw/Dataset001_MBHSeg25/imagesTr \
#     --out-dir vlm_datasets/mbh-seg25-5cls/train
#
#   python scripts/vlm_dataset_prepare_gt_5cls.py \
#     --gt-dir nnUNet/nnUNet_raw/Dataset001_MBHSeg25/labelsTs \
#     --images-dir nnUNet/nnUNet_raw/Dataset001_MBHSeg25/imagesTs \
#     --out-dir vlm_datasets/mbh-seg25-5cls/test

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.preprocessing.dataset_prepare import build_gt_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="5cls GT 라벨만으로 VLM 데이터셋 생성 (예측 없음)")
    parser.add_argument("--gt-dir", required=True, type=Path, help="5cls GT label nii.gz 폴더")
    parser.add_argument("--images-dir", required=True, type=Path, help="5cls 원본 이미지(_0000.nii.gz) 폴더")
    parser.add_argument("--out-dir", required=True, type=Path, help="images/, 5cls_dataset.json 저장 위치")
    return parser.parse_args()


def main():
    args = parse_arguments()
    all_samples = build_gt_dataset(args.gt_dir, args.images_dir, args.out_dir)

    dataset_path = args.out_dir / "5cls_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {dataset_path} ({len(all_samples)}개 샘플)")


if __name__ == "__main__":
    main()
