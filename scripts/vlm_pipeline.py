# YAML config 하나로 is_lesion -> subtype -> evaluate를 순서대로 실행하는 오케스트레이터.
# 각 스테이지는 서브프로세스가 아니라 vlm_is_lesion.py/vlm_lesion_subtype.py/vlm_evaluate.py가
# 쓰는 함수를 그대로 재사용한다. dataset_json(읽기 전용)을 결과_json으로 복사하며 시작해서,
# 이후 스테이지들이 같은 결과_json 위에 컬럼을 계속 채워나간다.
#
# 사용 예:
#   python scripts/vlm_pipeline.py --config configs/vlm_pipeline/thr0.001.yaml

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.classify.client import GEN_CONFIGS, get_client
from src.vlm.classify.evaluate import run_detection_eval, show_vlm_errors
from src.vlm.classify.is_lesion_prompts import ICHIsLesionResult
from src.vlm.classify.is_lesion_prompts import build_messages as build_is_lesion_messages
from src.vlm.classify.pipeline import classify_dataset_records
from src.vlm.classify.segmentation_eval import run_segmentation_eval
from src.vlm.classify.subtype_prompts import ICHSubtypeResult
from src.vlm.classify.subtype_prompts import build_messages as build_subtype_messages

ALL_STAGES = ("is_lesion", "subtype", "evaluate")


def parse_arguments():
    parser = argparse.ArgumentParser(description="vlm_is_lesion + vlm_lesion_subtype + vlm_evaluate를 config로 체이닝")
    parser.add_argument("--config", required=True, type=Path, help="configs/vlm_pipeline/*.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("stages", list(ALL_STAGES))
    cfg.setdefault("tag", "full")
    # subtype 설정이 비어있으면 is_lesion 설정(base_url/model/gen_config/crop_mode/max_workers)을 상속
    cfg["subtype"] = {**cfg.get("is_lesion", {}), **cfg.get("subtype", {})}
    return cfg


def run_is_lesion_stage(cfg: dict, results_json: Path) -> None:
    stage_cfg = cfg["is_lesion"]
    with open(cfg["dataset_json"], encoding="utf-8") as f:
        records = json.load(f)

    client = get_client(stage_cfg.get("base_url", "http://localhost:8891/v1"))
    results = classify_dataset_records(
        records,
        build_messages_fn=build_is_lesion_messages,
        schema_cls=ICHIsLesionResult,
        client=client,
        model_name=stage_cfg.get("model", "Qwen/Qwen3.5-27B"),
        gen_config=GEN_CONFIGS[stage_cfg.get("gen_config", "non_thinking")],
        fallback={"is_lesion": False},
        crop_mode=stage_cfg.get("crop_mode", "crop_min"),
        max_workers=stage_cfg.get("max_workers", 8),
    )
    for r in results:
        r["vlm1_is_lesion"] = r.pop("is_lesion")

    results_json.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[is_lesion] 저장: {results_json} ({len(results)}개)")


def run_subtype_stage(cfg: dict, results_json: Path) -> None:
    stage_cfg = cfg["subtype"]
    with open(results_json, encoding="utf-8") as f:
        records = json.load(f)

    lesion_records = [r for r in records if r["vlm1_is_lesion"]]
    client = get_client(stage_cfg.get("base_url", "http://localhost:8891/v1"))
    subtype_results = classify_dataset_records(
        lesion_records,
        build_messages_fn=build_subtype_messages,
        schema_cls=ICHSubtypeResult,
        client=client,
        model_name=stage_cfg.get("model", "Qwen/Qwen3.5-27B"),
        gen_config=GEN_CONFIGS[stage_cfg.get("gen_config", "non_thinking")],
        fallback={"lesion_type": "NONE"},
        crop_mode=stage_cfg.get("crop_mode", "crop_min"),
        max_workers=stage_cfg.get("max_workers", 8),
    )
    subtype_by_id = {r["id"]: r["lesion_type"] for r in subtype_results}
    merged = [{**r, "vlm2_lesion_type": subtype_by_id.get(r["id"], "NONE")} for r in records]

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[subtype] 저장: {results_json} ({len(merged)}개)")


def run_evaluate_stage(cfg: dict, results_json: Path) -> None:
    stage_cfg = cfg["evaluate"]
    out_dir = Path(cfg["results_json"]).parent
    tag = cfg["tag"]

    with open(results_json, encoding="utf-8") as f:
        records = json.load(f)

    summary = {"detect": run_detection_eval(records)}

    errors = show_vlm_errors(records)
    errors.drop(columns=["images"]).to_csv(out_dir / f"{tag}_errors.csv", index=False)

    if not stage_cfg.get("skip_seg_eval", False):
        summary["segmentation"] = run_segmentation_eval(records, stage_cfg["pred_5cls_dir"], stage_cfg["gt_5cls_dir"], out_dir)

    with open(out_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[evaluate] 저장: {out_dir / f'{tag}_summary.json'}")


STAGE_RUNNERS = {
    "is_lesion": run_is_lesion_stage,
    "subtype": run_subtype_stage,
    "evaluate": run_evaluate_stage,
}


def main():
    args = parse_arguments()
    cfg = load_config(args.config)
    results_json = Path(cfg["results_json"])

    for stage in cfg["stages"]:
        print(f"\n########## stage: {stage} ##########")
        STAGE_RUNNERS[stage](cfg, results_json)


if __name__ == "__main__":
    main()
