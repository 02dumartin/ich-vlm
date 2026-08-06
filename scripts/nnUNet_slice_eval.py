# nnUNet 스캔 단위 평가 CSV
#
# nnUNet_test_5cls.py / nnUNet_test_1cls.py가 만들어 둔 summary.json들을 읽어서,
# (테스트셋 전체 평균이 아니라) 케이스 하나하나의 volume 전체 기준 Dice/IoU를 행으로 풀어낸다.
# z축 슬라이스 수(n_slice)도 같이 붙여서, 노트북에서 "주목할 만한 스캔"을 골라내는 데 쓴다.
#
# 사용 예:
#   python scripts/nnUNet_slice_eval.py --classes 5cls
#   python scripts/nnUNet_slice_eval.py --classes 1cls --dimension 2d --dataset CT-ICH

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.nnunet.slice_eval import build_scan_df

NNUNET_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet")

DATASET_SPECS = {
    "5cls": {
        "dataset_name": "Dataset001_MBHSeg25",
        "class_names": {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"},
        "multiclass": True,
    },
    "1cls": {
        "dataset_name": "Dataset002_MBHSeg25_1cls",
        "class_names": {1: "ICH"},
        "multiclass": False,
    },
}

# summary.json에 매핑되는 Dimension 라벨 (nnUNet_test_*.py가 만드는 폴더명과 동일해야 함)
DIM_LABELS = {
    "3d_fullres": "3D only",
    "2d": "2D only",
    "3d_fullres_pp": "3D only (PP)",
    "2d_pp": "2D only (PP)",
    "best_config": "best_config",
    "best_config_pp": "best_config (PP)",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="nnUNet 스캔 단위 Dice/IoU 평가 CSV 생성")
    parser.add_argument("--classes", choices=list(DATASET_SPECS.keys()), default="5cls",
                         help="5cls(Dataset001) 또는 1cls(Dataset002) 구분")
    parser.add_argument("--dimension", choices=list(DIM_LABELS.keys()) + ["all"], default="all",
                         help="어떤 config 폴더의 summary.json을 읽을지")
    parser.add_argument("--dataset", default="all",
                         help="테스트셋 이름 (예: CT-ICH, MBH-Seg25). 기본은 전부")
    parser.add_argument("--output-dir", default="../nnUNet/nnUNet_predictions",
                         help="CSV 저장 위치 (스크립트 기준 상대경로)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    spec = DATASET_SPECS[args.classes]
    pred_root = NNUNET_ROOT / "nnUNet_predictions" / spec["dataset_name"]

    dims = list(DIM_LABELS.keys()) if args.dimension == "all" else [args.dimension]

    dfs = []
    for dim in dims:
        dim_dir = pred_root / dim
        if not dim_dir.exists():
            continue
        for test_dir in sorted(dim_dir.iterdir()):
            if not test_dir.is_dir():
                continue
            test_name = test_dir.name.replace("_", " ")
            if args.dataset != "all" and test_name != args.dataset:
                continue
            summary_path = test_dir / "summary.json"
            if not summary_path.exists():
                print(f"[건너뜀] summary.json 없음: {summary_path}")
                continue
            print(f"[로드] {summary_path}")
            dfs.append(build_scan_df(summary_path, test_name, DIM_LABELS[dim], spec["class_names"], spec["multiclass"]))

    if not dfs:
        print("결과 없음")
        return

    result_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    result_df.insert(0, "Classes", args.classes)  # 5cls/1cls 구분 (여러 CSV를 합쳐도 구분 가능하도록)

    out_csv = Path(__file__).resolve().parent / args.output_dir / f"scan_eval_{args.classes}.csv"
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv)  # 판다스 기본 행 번호(RangeIndex) 포함해서 저장
    print(f"\n저장: {out_csv} ({len(result_df)}행)")


if __name__ == "__main__":
    main()
