"""
VLM(vLLM) 분류 공용 파이프라인.

is_lesion 판정과 subtype 분류는 prompt/응답 스키마만 다르고, client 호출/재시도/
슬라이스 그룹 단위 배치/JSON 파싱 로직은 동일하다 (vlm_classifier_5cls.ipynb 기준).
그 공용 로직만 여기 모아두고, 태스크별 prompt/스키마는 is_lesion_prompts.py /
subtype_prompts.py에서 build_messages_fn / schema_cls로 주입받는다.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ValidationError

from src.vlm.classify.client import extract_first_json, to_rgb_pil


def instance_number(record: dict) -> int:
    m = re.search(r"_inst(\d+)$", record["id"])
    return int(m.group(1)) if m else 0


def group_by_slice(records: list[dict]) -> list[tuple[tuple[str, int], list[dict]]]:
    """(case_id, slice_idx)별로 묶고, 그룹 내부는 instance 번호 순으로 정렬."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        groups.setdefault((r["case_id"], r["slice_idx"]), []).append(r)
    for key in groups:
        groups[key].sort(key=instance_number)
    return sorted(groups.items(), key=lambda kv: kv[0])


def classify_one(full_img, crop_img, build_messages_fn: Callable, schema_cls: type[BaseModel],
                  client: OpenAI, model_name: str, gen_config: dict, max_retries: int = 1) -> dict | None:
    """
    한 인스턴스(overlay+crop 이미지 쌍)를 분류해 schema_cls 필드로 이루어진 dict를 반환.
    파싱이 max_retries 안에 끝내 실패하면 None (호출부에서 fallback 값으로 채움).
    """
    full_img, crop_img = to_rgb_pil(full_img), to_rgb_pil(crop_img)
    messages = build_messages_fn(full_img, crop_img)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema_cls.__name__, "schema": schema_cls.model_json_schema()},
    }

    last_err = None
    for _ in range(max_retries + 1):
        completion = client.chat.completions.create(
            model=model_name, messages=messages, response_format=response_format, **gen_config,
        )
        try:
            js = extract_first_json(completion.choices[0].message.content)
            return schema_cls.model_validate_json(js).model_dump()
        except (ValueError, ValidationError) as e:
            last_err = e

    print(f"[경고] 파싱 실패: {last_err}")
    return None


def classify_batch(image_pairs: list[tuple], classify_fn: Callable, max_workers: int = 8) -> list[dict | None]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda pair: classify_fn(*pair), image_pairs))


def classify_dataset_records(records: list[dict], build_messages_fn: Callable, schema_cls: type[BaseModel],
                              client: OpenAI, model_name: str, gen_config: dict, fallback: dict,
                              crop_mode: str = "crop_min", max_workers: int = 8) -> list[dict]:
    """
    슬라이스 그룹 단위 순차 + 그룹 내 병렬로 VLM 분류를 실행
    각 레코드는 원래 필드(GT 포함)를 그대로 유지한 채 
    instance_number + 분류 결과(schema_cls 필드)만 덧붙여 반환한다
    — 단일 JSON에 컬럼을 누적하는 구조이므로 입력 필드를 걸러내지 않음
    VLM에는 images[overlay]/images[crop_mode] 두 장만 build_messages_fn을 통해 전달되고, 
    그 외 필드(GT 포함)는 모델 입력에 섞이지 않음.
    """
    merged = []
    for (case_id, slice_idx), slice_records in group_by_slice(records):
        image_pairs = [
            (Image.open(r["images"]["overlay"]), Image.open(r["images"][crop_mode]))
            for r in slice_records
        ]

        def classify_fn(full_img, crop_img):
            return classify_one(full_img, crop_img, build_messages_fn, schema_cls, client, model_name, gen_config)

        results = classify_batch(image_pairs, classify_fn, max_workers=max_workers)
        for r, result in zip(slice_records, results):
            merged.append({
                **r,
                "instance_number": instance_number(r),
                **(result if result is not None else fallback),
            })
    return merged
