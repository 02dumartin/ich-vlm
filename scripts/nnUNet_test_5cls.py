# nnUNet 5cls Test
#
# 3D 가중치: nnUNet_results/Dataset003_MBHSeg25/nnUNetTrainer__nnUNetPlans__3d_fullres
# 2D 가중치: nnUNet_results/Dataset003_MBHSeg25/nnUNetTrainer__nnUNetPlans__2d
#
# test 데이터
#   /home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet/test_data/Test1_MBHSeg25
#   /home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet/test_data/Test2_CTICH
#
# 흐름:
#   1) nnUNetv2_find_best_configuration 실행, 결과인 inference_information.json(공식 결정 파일:
#      postprocessing까지 반영된 최종 승자)을 읽어 best_config(단일 or ensemble) + postprocessing.pkl 확보
#   2) 성공하면: best_config 예측 + best_config에 postprocessing 적용까지 2행, 2D/3D 각각 예측(PP 미적용) 2행
#      실패하면: 2D/3D 각각 예측만 수행
#   3) 모든 예측은 nnUNetv2_evaluate_folder로 summary.json을 생성해 평가 (nnU-Net 표준 지표 사용)
#   4) results_5cls_all.csv에 Dimension 열을 "best_config" / "best_config (PP)" / "2D only" / "3D only"로 저장 (소수점 3자리)
#
# 공통 함수(예측/평가/집계)는 src/segmentation/nnunet/test.py 참고. 여기엔 이 데이터셋 전용 설정과
# 실행 순서만 남긴다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.nnunet.test import (
    NNUNetConfig,
    add_macro_average,
    best_config_predict_and_evaluate,
    build_results_df,
    postprocess_and_evaluate,
    predict_and_evaluate,
    run_find_best_config,
    save_or_merge_csv,
    summary_to_row,
)

NNUNET_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/nnUNet")
CFG = NNUNetConfig(nnunet_root=NNUNET_ROOT, dataset_id=1, dataset_name="Dataset001_MBHSeg25")
CFG.set_env()

CLASS_NAMES = {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"}

# 데이터 2개 사용
TEST_SETS = {
    "MBH-Seg25": NNUNET_ROOT / "test_data" / "Test1_MBHSeg25",
    "CT-ICH": NNUNET_ROOT / "test_data" / "Test2_CTICH",
}

# nnUNetv2 configuration 이름: 3D는 "3d_fullres", 2D는 "2d" (폴더명 기준)
CONFIGS = {"3d_fullres": "3D only", "2d": "2D only"}

PRED_ROOT = NNUNET_ROOT / "nnUNet_predictions" / CFG.dataset_name

# 6번 GPU만 사용
GPU_POOL = ["6"]


def main():
    # config별(2D 단독/3D 단독) postprocessing 여부 + 전체(앙상블 포함) 최적 조합을 따로 구한다.
    # best_config가 3D(또는 앙상블)로 뽑히더라도 2D 자신의 PP 적용 결과를 놓치지 않기 위함.
    best_by_config = {
        config: run_find_best_config(CFG, configs=(config,)) for config in CONFIGS
    }
    for config, best in best_by_config.items():
        if best is not None:
            print(f"[{config} 단독 best] postprocessing={best['postprocessing_pkl']}")

    best_config = run_find_best_config(CFG, configs=tuple(CONFIGS.keys()))
    if best_config is not None:
        print(f"[best_config] {best_config['configs']}")

    results = []
    call_idx = 0

    for test_name, test_dir in TEST_SETS.items():
        images_dir = test_dir / "imagesTs"
        labels_dir = test_dir / "labelsTs"

        if not images_dir.exists():
            print(f"[건너뜀] {test_name}: imagesTs 없음 ({images_dir})")
            continue

        # 2D / 3D 각각 (find_best_config 성공 여부와 무관하게 항상 실행, postprocessing은 적용 안 함)
        for config, dim_label in CONFIGS.items():
            pred_dir = PRED_ROOT / config / test_name
            gpu_id = GPU_POOL[call_idx % len(GPU_POOL)]
            call_idx += 1
            summary_path = predict_and_evaluate(CFG, images_dir, pred_dir, labels_dir, config, gpu_id)
            results.append(summary_to_row(summary_path, test_name, dim_label, CLASS_NAMES))

            # 이 config 자신의 postprocessing 적용 버전
            best_single = best_by_config.get(config)
            if best_single is not None:
                pp_dir = PRED_ROOT / f"{config}_pp" / test_name
                pp_summary_path = postprocess_and_evaluate(
                    CFG, pred_dir, pp_dir, labels_dir,
                    best_single["postprocessing_pkl"], best_single["plans_json"],
                )
                results.append(summary_to_row(pp_summary_path, test_name, f"{dim_label} (PP)", CLASS_NAMES))

        # best_config 결과 (find_best_config이 성공했을 때만 추가) + postprocessing 적용 버전
        if best_config is not None:
            pred_dir = PRED_ROOT / "best_config" / test_name
            gpu_id = GPU_POOL[call_idx % len(GPU_POOL)]
            call_idx += 1
            summary_path = best_config_predict_and_evaluate(
                CFG, best_config["configs"], images_dir, pred_dir, labels_dir, gpu_id
            )
            results.append(summary_to_row(summary_path, test_name, "best_config", CLASS_NAMES))

            pp_dir = PRED_ROOT / "best_config_pp" / test_name
            pp_summary_path = postprocess_and_evaluate(
                CFG, pred_dir, pp_dir, labels_dir,
                best_config["postprocessing_pkl"], best_config["plans_json"],
            )
            results.append(summary_to_row(pp_summary_path, test_name, "best_config (PP)", CLASS_NAMES))

    if not results:
        print("결과 없음")
        return

    results = [add_macro_average(r, CLASS_NAMES, metrics=("Dice", "IoU")) for r in results]
    results_df = build_results_df(results, CLASS_NAMES, macro_metrics=("Dice", "IoU"))
    print("\n===== nnUNetv2 5cls 결과 =====")
    print(results_df.to_string(index=False))

    out_csv = NNUNET_ROOT / "nnUNet_predictions" / "results_5cls_all.csv"
    save_or_merge_csv(results_df, out_csv)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
