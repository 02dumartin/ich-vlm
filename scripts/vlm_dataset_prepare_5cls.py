# nnUNet 5cls 예측(Dataset001_MBHSeg25)과 5cls GT를 매칭해 VLM(Qwen) 입력용 데이터셋 생성.
#
# 슬라이스별 pred 인스턴스 bbox overlay, crop 이미지를 --out-dir/images 아래에 저장하고
# 5cls_dataset.json(bbox, class_id 등), fn_log.csv(매칭 안 된 GT 통계)를 --out-dir에 저장한다.
#
# 사용 예:
#   python scripts/vlm_dataset_prepare_5cls.py \
#     --pred-dir /path/to/nnUNet_predictions/Dataset001_MBHSeg25/2d_pp/MBH-Seg25 \
#     --gt-dir /path/to/test_data/Test1_MBHSeg25/labelsTs \
#     --images-dir /path/to/test_data/Test1_MBHSeg25/imagesTs \
#     --out-dir /path/to/experiments/260818_vlm_mbhseg_5cls_dataset


import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.dataset_prepare import build_dataset, save_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="nnUNet 5cls 예측 vs 5cls GT VLM 데이터셋 생성")
    parser.add_argument("--pred-dir", required=True, type=Path, help="5cls 예측 nii.gz 폴더")
    parser.add_argument("--gt-dir", required=True, type=Path, help="5cls GT label nii.gz 폴더")
    parser.add_argument("--images-dir", required=True, type=Path, help="5cls GT 원본 이미지(_0000.nii.gz) 폴더")
    parser.add_argument("--out-dir", required=True, type=Path, help="images/, 5cls_dataset.json, fn_log.csv 저장 위치")
    return parser.parse_args()


def main():
    args = parse_arguments()
    all_samples, fn_log = build_dataset(args.pred_dir, args.gt_dir, args.images_dir, args.out_dir)
    save_dataset(all_samples, fn_log, args.out_dir, "5cls_dataset.json")


if __name__ == "__main__":
    main()
