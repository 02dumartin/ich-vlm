# 2단계: vlm_is_lesion.py 결과 중 is_lesion=True인 인스턴스만 대상으로
# VLM(Reader#2)이 subtype(EDH/IPH/IVH/SAH/SDH)을 분류한다.
# is_lesion=False였던 인스턴스는 vlm_lesion_type="NONE"으로 그대로 통과시켜,
# 두 단계 결과를 합친 최종 예측 파일(vlm_is_lesion/vlm_lesion_type)을 만든다.
#
# 사용 예:
#   python scripts/vlm_lesion_subtype.py \
#     --is-lesion-json /path/to/output/pilot_is_lesion.json \
#     --out-dir /path/to/output --tag pilot

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.classify.client import GEN_CONFIGS, get_client
from src.vlm.classify.pipeline import classify_dataset_records
from src.vlm.classify.subtype_prompts import ICHSubtypeResult, build_messages


def parse_arguments():
    parser = argparse.ArgumentParser(description="VLM 2단계: is_lesion=True 인스턴스의 subtype 분류")
    parser.add_argument("--is-lesion-json", required=True, type=Path, help="vlm_is_lesion.py가 만든 {tag}_is_lesion.json")
    parser.add_argument("--out-dir", required=True, type=Path, help="{tag}_predictions.json 저장 위치")
    parser.add_argument("--tag", default="full", help="출력 파일명 접두어 (예: pilot, full)")
    parser.add_argument("--base-url", default="http://localhost:8891/v1", help="vLLM OpenAI 호환 서버 주소")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B", help="vLLM에 올라간 모델 이름")
    parser.add_argument("--gen-config", default="non_thinking", choices=list(GEN_CONFIGS.keys()))
    parser.add_argument("--max-workers", type=int, default=8, help="슬라이스 그룹 내 병렬 요청 수")
    return parser.parse_args()


def run_lesion_subtype(is_lesion_records: list[dict], client, model_name: str, gen_config: dict,
                        max_workers: int = 8) -> list[dict]:
    lesion_records = [r for r in is_lesion_records if r["is_lesion"]]
    print(f"=== {len(lesion_records)}/{len(is_lesion_records)}개(is_lesion=True) subtype 분류 시작 ===")

    subtype_results = classify_dataset_records(
        lesion_records,
        build_messages_fn=build_messages,
        schema_cls=ICHSubtypeResult,
        client=client,
        model_name=model_name,
        gen_config=gen_config,
        fallback={"lesion_type": "NONE"},
        max_workers=max_workers,
    )
    subtype_by_id = {r["id"]: r["lesion_type"] for r in subtype_results}

    return [
        {
            **{k: v for k, v in r.items() if k != "is_lesion"},
            "vlm_is_lesion": r["is_lesion"],
            "vlm_lesion_type": subtype_by_id.get(r["id"], "NONE"),
        }
        for r in is_lesion_records
    ]


def main():
    args = parse_arguments()

    with open(args.is_lesion_json, encoding="utf-8") as f:
        is_lesion_records = json.load(f)

    client = get_client(args.base_url)
    merged = run_lesion_subtype(
        is_lesion_records, client, args.model, GEN_CONFIGS[args.gen_config], max_workers=args.max_workers,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{args.tag}_predictions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_json} ({len(merged)}개)")


if __name__ == "__main__":
    main()
