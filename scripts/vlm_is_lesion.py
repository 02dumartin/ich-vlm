# 1단계: 5cls_dataset.json의 각 인스턴스에 대해 VLM(Reader#2)이 독립적으로
# "이게 진짜 병변인가"만 판단한다 (subtype은 vlm_lesion_subtype.py에서 별도 처리).
#
# 사용 예:
#   python scripts/vlm_is_lesion.py \
#     --dataset-json /path/to/vlm_datasets/.../thr0.05/5cls_dataset.json \
#     --out-dir /path/to/experiments/260825_vlm-is-lesion-mbhseg25/output \
#     --tag pilot --case-limit 5

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.classify.client import GEN_CONFIGS, get_client
from src.vlm.classify.is_lesion_prompts import ICHIsLesionResult, build_messages
from src.vlm.classify.pipeline import classify_dataset_records


def parse_arguments():
    parser = argparse.ArgumentParser(description="VLM 1단계: is_lesion 판정")
    parser.add_argument("--dataset-json", required=True, type=Path, help="dataset_prepare.py가 만든 5cls_dataset.json")
    parser.add_argument("--out-dir", required=True, type=Path, help="{tag}_is_lesion.json 저장 위치")
    parser.add_argument("--tag", default="full", help="출력 파일명 접두어 (예: pilot, full)")
    parser.add_argument("--case-limit", type=int, default=None, help="앞 N개 케이스만 실행 (파일럿용)")
    parser.add_argument("--base-url", default="http://localhost:8891/v1", help="vLLM OpenAI 호환 서버 주소")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B", help="vLLM에 올라간 모델 이름")
    parser.add_argument("--gen-config", default="non_thinking", choices=list(GEN_CONFIGS.keys()))
    parser.add_argument("--max-workers", type=int, default=8, help="슬라이스 그룹 내 병렬 요청 수")
    return parser.parse_args()


def main():
    args = parse_arguments()

    with open(args.dataset_json, encoding="utf-8") as f:
        records = json.load(f)

    if args.case_limit is not None:
        case_ids = sorted({r["case_id"] for r in records})[:args.case_limit]
        records = [r for r in records if r["case_id"] in set(case_ids)]

    print(f"=== {len(records)}개 인스턴스 is_lesion 판정 시작 ===")

    client = get_client(args.base_url)
    results = classify_dataset_records(
        records,
        build_messages_fn=build_messages,
        schema_cls=ICHIsLesionResult,
        client=client,
        model_name=args.model,
        gen_config=GEN_CONFIGS[args.gen_config],
        fallback={"is_lesion": False},
        max_workers=args.max_workers,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{args.tag}_is_lesion.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_json} ({len(results)}개)")


if __name__ == "__main__":
    main()
