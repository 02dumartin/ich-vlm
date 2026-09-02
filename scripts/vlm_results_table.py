# crop sweep 실험 폴더 하나의 full_summary.json/full_results.json을 모아 results_table.md에
# 붙여넣을 지표 표(mDice/mPrecision/mRecall + Detection Acc/Precision/Recall)와
# FP/진짜 병변 제거율 표를 markdown으로 출력한다. 계산 로직은 src/vlm/analysis/results_table.py.
#
# 사용 예:
#   python scripts/vlm_results_table.py \
#     --exp-dir experiments/260826_vlm-is-lesion-crop-sweep-mbhseg25-5cls_v01

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.analysis.results_table import build_tables, render_metrics_table, render_removal_rate_table
from src.vlm.preprocessing.image_crop import CROP_MODES


def parse_arguments():
    parser = argparse.ArgumentParser(description="crop sweep 실험 결과를 results_table.md용 markdown 표로 출력")
    parser.add_argument("--exp-dir", required=True, type=Path, help="experiments/{실험명}_v{NN} 폴더")
    parser.add_argument("--crop-modes", nargs="+", default=list(CROP_MODES), choices=list(CROP_MODES))
    parser.add_argument("--tag", default="full")
    return parser.parse_args()


def main():
    args = parse_arguments()
    tables = build_tables(args.exp_dir, crop_modes=tuple(args.crop_modes), tag=args.tag)

    print("## 지표 표\n")
    print(render_metrics_table(tables))
    print("\n## FP/진짜 병변 제거율 표\n")
    print(render_removal_rate_table(tables))


if __name__ == "__main__":
    main()
