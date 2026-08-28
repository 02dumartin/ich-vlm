# 2단계: vlm_is_lesion.py 결과 중 vlm1_is_lesion=True인 인스턴스만 대상으로
# VLM(Reader#2)이 subtype(EDH/IPH/IVH/SAH/SDH)을 분류한다.
# vlm1_is_lesion=False였던 인스턴스는 vlm2_lesion_type="NONE"으로 그대로 통과시켜,
# 같은 결과 JSON에 vlm2_lesion_type 컬럼을 이어서 채운다 (2단계 VLM 호출 자체를 스킵).
#
# 사용 예:
#   python scripts/vlm_lesion_subtype.py \
#     --in-json /path/to/output/pilot_results.json \
#     --out-json /path/to/output/pilot_results.json --tag pilot

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.classify.client import GEN_CONFIGS, get_client
from src.vlm.classify.pipeline import classify_dataset_records
from src.vlm.classify.subtype_prompts import ICHSubtypeResult, build_messages
from src.vlm.image_crop import CROP_MODES


def parse_arguments():
    parser = argparse.ArgumentParser(description="VLM 2단계: vlm1_is_lesion=True 인스턴스의 subtype 분류")
    parser.add_argument("--in-json", required=True, type=Path, help="vlm_is_lesion.py가 만든 vlm1_is_lesion 포함 결과 JSON")
    parser.add_argument("--out-json", required=True, type=Path, help="vlm2_lesion_type까지 채워 저장할 경로 (보통 --in-json과 동일 = 제자리 갱신)")
    parser.add_argument("--tag", default="full", help="디버그 스냅샷 파일명 접두어 (예: pilot, full)")
    parser.add_argument("--base-url", default="http://localhost:8891/v1", help="vLLM OpenAI 호환 서버 주소")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B", help="vLLM에 올라간 모델 이름")
    parser.add_argument("--gen-config", default="non_thinking", choices=list(GEN_CONFIGS.keys()))
    parser.add_argument("--crop-mode", default="crop_min", choices=CROP_MODES, help="VLM Image 2로 쓸 crop 종류")
    parser.add_argument("--max-workers", type=int, default=8, help="슬라이스 그룹 내 병렬 요청 수")
    return parser.parse_args()


def run_lesion_subtype(records: list[dict], client, model_name: str, gen_config: dict,
                        crop_mode: str = "crop_min", max_workers: int = 8) -> tuple[list[dict], list[dict]]:
    """
    records 전체에 vlm2_lesion_type을 채워 반환한다.
    반환: (전체 레코드 - vlm2_lesion_type 포함, subtype VLM 호출 원본 결과 - 디버그용)
    """
    lesion_records = [r for r in records if r["vlm1_is_lesion"]]
    print(f"=== {len(lesion_records)}/{len(records)}개(vlm1_is_lesion=True) subtype 분류 시작 ===")

    subtype_results = classify_dataset_records(
        lesion_records,
        build_messages_fn=build_messages,
        schema_cls=ICHSubtypeResult,
        client=client,
        model_name=model_name,
        gen_config=gen_config,
        fallback={"lesion_type": "NONE"},
        crop_mode=crop_mode,
        max_workers=max_workers,
    )
    subtype_by_id = {r["id"]: r["lesion_type"] for r in subtype_results}

    merged = [
        {**r, "vlm2_lesion_type": subtype_by_id.get(r["id"], "NONE")}
        for r in records
    ]
    return merged, subtype_results


def main():
    args = parse_arguments()

    with open(args.in_json, encoding="utf-8") as f:
        records = json.load(f)

    client = get_client(args.base_url)
    merged, subtype_results = run_lesion_subtype(
        records, client, args.model, GEN_CONFIGS[args.gen_config],
        crop_mode=args.crop_mode, max_workers=args.max_workers,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    debug_json = args.out_json.parent / f"{args.tag}_debug_subtype.json"
    with open(debug_json, "w", encoding="utf-8") as f:
        json.dump(subtype_results, f, ensure_ascii=False, indent=2)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"저장: {args.out_json} ({len(merged)}개)")
    print(f"디버그 스냅샷: {debug_json}")


if __name__ == "__main__":
    main()
