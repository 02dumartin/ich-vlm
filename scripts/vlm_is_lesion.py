# 1단계: 5cls_dataset.json의 각 인스턴스에 대해 VLM(Reader#2)이 독립적으로
# "이게 진짜 병변인가"만 판단한다 (subtype은 vlm_lesion_subtype.py에서 별도 처리).
#
# --in-json/--out-json으로 파이프라인 전체가 하나의 결과 JSON을 누적해서 채워나간다.
# 첫 실행은 보통 --in-json에 dataset_prepare.py가 만든 원본 5cls_dataset.json(읽기 전용)을,
# --out-json에 실험 output 쪽 결과 파일 경로를 줘서 "복사하며 vlm1_is_lesion을 채우는"
# 효과를 낸다. 이후 vlm_lesion_subtype.py가 이 out-json을 --in-json으로 받아 이어서 채운다.
#
# 사용 예:
#   python scripts/vlm_is_lesion.py \
#     --in-json /path/to/vlm_datasets/mbh-seg25-5cls/thr0.05/5cls_dataset.json \
#     --out-json /path/to/experiments/.../output/pilot_results.json \
#     --tag pilot --case-limit 5

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.classify.client import GEN_CONFIGS, get_client
from src.vlm.classify.is_lesion_prompts import ICHIsLesionResult, build_messages, build_messages_example
from src.vlm.classify.pipeline import classify_dataset_records
from src.vlm.image_crop import CROP_MODES

PROMPT_VARIANTS = {"zeroshot": build_messages, "example": build_messages_example}


def parse_arguments():
    parser = argparse.ArgumentParser(description="VLM 1단계: is_lesion 판정")
    parser.add_argument("--in-json", required=True, type=Path, help="5cls_dataset.json 또는 이전 단계 결과 JSON")
    parser.add_argument("--out-json", required=True, type=Path, help="vlm1_is_lesion이 채워진 결과 JSON 저장 경로")
    parser.add_argument("--tag", default="full", help="디버그 스냅샷 파일명 접두어 (예: pilot, full)")
    parser.add_argument("--case-limit", type=int, default=None, help="앞 N개 케이스만 실행 (파일럿용)")
    parser.add_argument("--base-url", default="http://localhost:8891/v1", help="vLLM OpenAI 호환 서버 주소")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B", help="vLLM에 올라간 모델 이름")
    parser.add_argument("--gen-config", default="non_thinking", choices=list(GEN_CONFIGS.keys()))
    parser.add_argument("--crop-mode", default="crop_min", choices=CROP_MODES, help="VLM Image 2로 쓸 crop 종류")
    parser.add_argument("--max-workers", type=int, default=8, help="슬라이스 그룹 내 병렬 요청 수")
    parser.add_argument("--prompt-variant", default="zeroshot", choices=list(PROMPT_VARIANTS.keys()),
                         help="zeroshot: 기존 프롬프트만. example: PLOS FP 사례 few-shot을 앞에 붙임")
    return parser.parse_args()


def main():
    args = parse_arguments()

    with open(args.in_json, encoding="utf-8") as f:
        records = json.load(f)

    if args.case_limit is not None:
        case_ids = sorted({r["case_id"] for r in records})[:args.case_limit]
        records = [r for r in records if r["case_id"] in set(case_ids)]

    print(f"=== {len(records)}개 인스턴스 is_lesion 판정 시작 ===")

    client = get_client(args.base_url)
    results = classify_dataset_records(
        records,
        build_messages_fn=PROMPT_VARIANTS[args.prompt_variant],
        schema_cls=ICHIsLesionResult,
        client=client,
        model_name=args.model,
        gen_config=GEN_CONFIGS[args.gen_config],
        fallback={"is_lesion": False},
        crop_mode=args.crop_mode,
        max_workers=args.max_workers,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    debug_json = args.out_json.parent / f"{args.tag}_debug_is_lesion.json"
    with open(debug_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 스키마 응답 필드명(is_lesion)은 VLM 프롬프트/response_format과 맞춰야 해서 그대로 두고,
    # 최종 결과 JSON에 합칠 때만 vlm1_is_lesion으로 엔티티 접두어 네이밍을 적용한다.
    for r in results:
        r["vlm1_is_lesion"] = r.pop("is_lesion")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장: {args.out_json} ({len(results)}개)")
    print(f"디버그 스냅샷: {debug_json}")


if __name__ == "__main__":
    main()
