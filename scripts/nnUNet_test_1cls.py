# nnUNet 1cls(2cls: background+ich) Test
#
# 3D 가중치: nnUNet_results/Dataset002_MBHSeg25_1cls/nnUNetTrainer__nnUNetPlans__3d_fullres
# 2D 가중치: nnUNet_results/Dataset002_MBHSeg25_1cls/nnUNetTrainer__nnUNetPlans__2d
#
# test 데이터
#   /home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet/test_data_1cls/Test1_MBHSeg25_1cls    (3D+2D)
#   /home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet/test_data_1cls/Test2_CTICH_1cls       (3D+2D)
#   SNU HE-01                                                                                     (2D만)
#
# 평가는 nnUNetv2_evaluate_folder가 생성하는 summary.json 기준(nnU-Net 표준 지표)
#
# 흐름:
#   1) nnUNetv2_find_best_configuration 실행, inference_information.json(공식 결정 파일)에서
#      best_config(단일 or ensemble) + postprocessing.pkl 확보
#   2) 테스트셋별로 spec["configs"]에 정의된 차원(들) 예측 (postprocessing 미적용)
#   3) best_config가 그 테스트셋이 지원하는 차원을 전부 포함할 때만 best_config 예측 + PP 적용본 추가
#
# 공통 함수(예측/평가/집계)는 src/segmentation/nnunet/test.py 참고. 여기엔 이 데이터셋 전용 설정과
# 실행 순서만 남긴다. (1cls는 클래스가 ICH 하나뿐이라 mDice/mIoU macro-average를 추가하지 않는다 —
# ICH_Dice/ICH_IoU 자체가 이미 "전체" 값이라 중복이기 때문)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.nnunet.test import (
    NNUNetConfig,
    best_config_predict_and_evaluate,
    build_results_df,
    postprocess_and_evaluate,
    predict_and_evaluate,
    run_find_best_config,
    save_or_merge_csv,
    summary_to_row,
)

NNUNET_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet")
CFG = NNUNetConfig(nnunet_root=NNUNET_ROOT, dataset_id=2, dataset_name="Dataset002_MBHSeg25_1cls")
CFG.set_env()

CLASS_NAMES = {1: "ICH"}  # 1cls(전경) = ICH 유무

# 테스트셋별로 어떤 차원(config)을 시도할지 정의
# HE-01은 3D 볼륨을 만들 수 없어서 2D만 시도
TEST_SETS = {
    "MBH-Seg25": {
        "dir": NNUNET_ROOT / "test_data_1cls" / "Test1_MBHSeg25_1cls",
        "configs": {"3d_fullres": "3D only", "2d": "2D only"},
    },
    "CT-ICH": {
        "dir": NNUNET_ROOT / "test_data_1cls" / "Test2_CTICH_1cls",
        "configs": {"3d_fullres": "3D only", "2d": "2D only"},
    },
    #"SNU HE-01": {
    #    "dir": NNUNET_ROOT / "test_data_1cls" / "Test3_HE01_1cls",
    #    "configs": {"2d": "2D only"},
    #},
}

PRED_ROOT = NNUNET_ROOT / "nnUNet_predictions" / CFG.dataset_name

# 6번 GPU만 사용
GPU_POOL = ["6"]


def main():
    # 이 데이터셋에서 실제로 쓰이는 config 종류(모든 테스트셋의 configs 합집합) 각각에 대해
    # 단독 postprocessing 여부를 따로 구한다. best_config가 3D(또는 앙상블)로 뽑히더라도
    # 2D 자신의 PP 적용 결과를 놓치지 않기 위함.
    all_configs = sorted({config for spec in TEST_SETS.values() for config in spec["configs"]})
    best_by_config = {config: run_find_best_config(CFG, configs=(config,)) for config in all_configs}
    for config, best in best_by_config.items():
        if best is not None:
            print(f"[{config} 단독 best] postprocessing={best['postprocessing_pkl']}")

    best_config = run_find_best_config(CFG, configs=tuple(all_configs))
    if best_config is not None:
        print(f"[best_config] {best_config['configs']}")

    results = []
    call_idx = 0

    for test_name, spec in TEST_SETS.items():
        test_dir = spec["dir"]
        images_dir = test_dir / "imagesTs"
        labels_dir = test_dir / "labelsTs"

        if not test_dir.exists():
            print(f"[건너뜀] {test_name}: 폴더 없음 ({test_dir}) -> TODO: 데이터 준비 필요")
            continue
        if not images_dir.exists():
            print(f"[건너뜀] {test_name}: imagesTs 없음 ({images_dir})")
            continue

        safe_name = test_name.replace(" ", "_")
        for config, dim_label in spec["configs"].items():
            pred_dir = PRED_ROOT / config / safe_name
            gpu_id = GPU_POOL[call_idx % len(GPU_POOL)]
            call_idx += 1
            summary_path = predict_and_evaluate(CFG, images_dir, pred_dir, labels_dir, config, gpu_id)
            results.append(summary_to_row(summary_path, test_name, dim_label, CLASS_NAMES))

            # 이 config 자신의 postprocessing 적용 버전
            best_single = best_by_config.get(config)
            if best_single is not None:
                pp_dir = PRED_ROOT / f"{config}_pp" / safe_name
                pp_summary_path = postprocess_and_evaluate(
                    CFG, pred_dir, pp_dir, labels_dir,
                    best_single["postprocessing_pkl"], best_single["plans_json"],
                )
                results.append(summary_to_row(pp_summary_path, test_name, f"{dim_label} (PP)", CLASS_NAMES))

        # best_config는 이 테스트셋이 필요한 차원(config)을 전부 지원할 때만 실행
        # (예: HE-01은 2D만 지원하므로 best_config가 3D/ensemble이면 건너뜀)
        if best_config is not None and set(best_config["configs"]).issubset(spec["configs"].keys()):
            pred_dir = PRED_ROOT / "best_config" / safe_name
            gpu_id = GPU_POOL[call_idx % len(GPU_POOL)]
            call_idx += 1
            summary_path = best_config_predict_and_evaluate(
                CFG, best_config["configs"], images_dir, pred_dir, labels_dir, gpu_id
            )
            results.append(summary_to_row(summary_path, test_name, "best_config", CLASS_NAMES))

            pp_dir = PRED_ROOT / "best_config_pp" / safe_name
            pp_summary_path = postprocess_and_evaluate(
                CFG, pred_dir, pp_dir, labels_dir,
                best_config["postprocessing_pkl"], best_config["plans_json"],
            )
            results.append(summary_to_row(pp_summary_path, test_name, "best_config (PP)", CLASS_NAMES))
        elif best_config is not None:
            print(f"[건너뜀] {test_name}: best_config({best_config['configs']})에 필요한 차원 없음")

    if not results:
        print("결과 없음")
        return

    results_df = build_results_df(results, CLASS_NAMES, macro_metrics=())
    print("\n===== nnUNetv2 1cls(2cls) 결과 =====")
    print(results_df.to_string(index=False))

    out_csv = NNUNET_ROOT / "nnUNet_predictions" / "results_1cls_all.csv"
    save_or_merge_csv(results_df, out_csv)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
