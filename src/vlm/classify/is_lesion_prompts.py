"""
is_lesion 판정(1단계: 후보 영역이 진짜 병변인지 여부만 판단) 스키마 + prompt.

FP 카테고리는 아래 4편을 종합:
    Kundisch A, et al. "Deep learning algorithm in detecting intracranial
        hemorrhages on emergency computed tomographies." PLOS ONE 2021.
        (빈도순: calcification > beam-hardening artifact > tumor > blood vessel)
    Gaizo/Osborne, et al. "Deep Learning to Detect Intracranial Hemorrhage in a
        National Teleradiology Program and the Impact on Interpretation Time."
        Radiology: AI 2024. (Figure 4: infarct, ventricle horn/temporal bone
        partial volume averaging, beam-hardening, metal streak artifact)
    "Utilization of Artificial Intelligence-based Intracranial Hemorrhage
        Detection on Emergent Noncontrast CT Images in Clinical Workflow."
        Radiology: AI 2022. (postop/postischemic defect 23.6% > artifact 19.7%
        > tumor 15.3%)
    Tombach F, et al. "Diagnostic performance and confidence of an optimized
        deep-learning algorithm for the detection of intracranial hemorrhages."
        Insights Imaging 2026. (A: hardening artifact, B: motion artifact,
        C: parenchymal calcification, D: malignancy with perifocal edema)
"""

from functools import lru_cache
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from src.vlm.classify.client import pil_to_base64_url, to_rgb_pil

FP_EXAMPLES_DIR = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/vlm_datasets/fp_examples/processed")

# Kundisch A, et al. PLOS ONE 2021(S2~S4 Fig)에서 fp_examples.py로 추출한 확인된 FP 사례.
# category/idx는 fp_examples.py의 SOURCES/파일명과 대응되고, note는 각 사례의 해부학적 근거
# (원문 figure caption 기반) - 왜 병변이 아닌지를 VLM에게 같이 알려주기 위함.
EXAMPLE_CASES = [
    {"category": "calcification", "idx": 0,
     "note": "Falx cerebri의 생리적 석회화. 정중선, 작고 둥글며 경계가 뚜렷함."},
    {"category": "calcification", "idx": 2,
     "note": "Infratentorial choroid plexus(좌측 lateral aperture 부근)의 생리적 석회화. "
             "고정된 정상 해부학적 위치."},
    {"category": "beam_hardening", "idx": 0,
     "note": "전두골 내판 바로 아래의 beam-hardening artifact. 뼈에서 방사되는 띠 모양이고 "
             "병변다운 형태가 아님."},
    {"category": "beam_hardening", "idx": 1,
     "note": "두개골에서 발생한 beam-hardening artifact(다른 증례)."},
    {"category": "tumor", "idx": 1,
     "note": "S4 Fig 원 캡션에 개별 진단명이 없는 증례(총괄 캡션 기준 hyperdense/부분 석회화 종양). "
             "정중선 부근 작고 둥근 고음영 병소이고 급성 혈종의 모양이 아님."},
    {"category": "tumor", "idx": 3,
     "note": "좌측 auditory canal의 vestibular schwannoma(extra-/intra-canalicular growth). "
             "소뇌교각(CPA)/내이도 부근의 종괴이며 급성 출혈의 밀도/모양이 아님."},
]


class ICHIsLesionResult(BaseModel):
    is_lesion: bool


def zeroshot_prompt() -> str:
    return """
    You are Reader #2, providing an independent second read of a candidate brain hemorrhage finding
    from a non-contrast head CT, originally flagged by a segmentation model (Reader #1).

    You are given two images of the flagged region.
    Image 1: the full axial slice, with the flagged region highlighted, showing its location relative
    to the skull/ventricles/brain surface.
    Image 2: a zoomed-in crop of the region, showing its shape and density pattern in detail.

    Reader #1 (the segmentation model) can make mistakes. Small or subtle true hemorrhages are common -
    do not default to "not a lesion" just because the finding is small or faint; judge based on
    location, shape, and density pattern, not size alone.

    Reader #1's false positives cluster into these categories (from published AI-ICH-detection error
    analyses, roughly ordered by how often they occur):

    1. Calcification - physiological (choroid plexus, pineal gland, falx cerebri, habenula: fixed
       midline/paired sites, round, small) or pathologic (basal ganglia, old granuloma: much denser
       and more chalk-white than acute blood, sharp margins, no edema).

    2. Beam-hardening / streak artifact - straight or fan-shaped streaks radiating from bone or
       metal, not an anatomically plausible lesion shape.

    3. Motion artifact - blurring or double-edge/ghosting over a broad region, not a discrete
       focal finding.

    4. Partial volume averaging - hyperdensity at a bone-brain interface (temporal bone, occipital
       horn of the lateral ventricle); check if it persists on neighboring slices.

    5. Tumor / mass - expansile mass-like shape, possibly with vasogenic edema, unlike an acute
       hematoma.

    Judge whether this is a true hemorrhage lesion (is_lesion). If it instead matches one of the FP
    categories above, or any other non-hemorrhage explanation, set is_lesion to false.

    STRICT OUTPUT RULES: Do NOT write any analysis, explanation, or text outside the JSON. Output ONLY the JSON object below, with nothing before or after it.
    {"is_lesion": true/false}
    """


def example_prompt() -> str:
    return zeroshot_prompt() + """

    Before this case, you were shown confirmed false-positive examples (published cases where an AI
    hemorrhage detector wrongly flagged calcification/artifact/tumor as hemorrhage). Use those only to
    calibrate what a *non*-lesion can look like - they are not evidence about this specific case, and a
    small/faint true hemorrhage should still be judged as is_lesion=true on its own merits.
    """


@lru_cache(maxsize=1)
def build_example_blocks() -> list[dict]:
    """EXAMPLE_CASES의 처리된 이미지(fp_examples.py 출력)를 few-shot 데모 content block으로 변환.
    각 사례를 "Image N (full)/(crop) + 근거 설명 + 정답"으로 제시해, 실제 질의보다 먼저 붙인다.
    EXAMPLE_CASES/이미지는 실행 중 안 바뀌는 고정값이라 lru_cache로 첫 호출에서만 디스크에서
    읽고 인코딩한다 - 인스턴스마다 build_messages_example -> 이 함수가 다시 불리는데, 캐싱 없이는
    데이터셋 크기만큼 같은 6쌍(12장) 이미지를 반복 재로딩/재인코딩하게 된다. 캐시로 돌려주는
    리스트는 호출부(build_messages_example)에서 +로 새 리스트를 만들어 붙이기만 하고 in-place로
    수정하지 않으므로 재사용해도 안전하다."""
    blocks = [{"type": "text", "text":
        "Study these confirmed non-hemorrhage examples before judging the actual case below "
        "(published false positives from an AI hemorrhage detector, Kundisch et al. PLOS ONE 2021). "
        "Each pair shows the full slice and a zoomed crop of the flagged region, followed by the "
        "correct answer and why."}]
    for ex in EXAMPLE_CASES:
        stem = f"{ex['category']}_{ex['idx']}"
        full_img = to_rgb_pil(Image.open(FP_EXAMPLES_DIR / f"{stem}_full.png"))
        crop_img = to_rgb_pil(Image.open(FP_EXAMPLES_DIR / f"{stem}_crop.png"))
        blocks += [
            {"type": "text", "text": f"Example ({ex['category']}) - full slice:"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_url(full_img)}},
            {"type": "text", "text": f"Example ({ex['category']}) - zoomed crop:"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_url(crop_img)}},
            {"type": "text", "text": f"{ex['note']} Correct answer: {{\"is_lesion\": false}}"},
        ]
    return blocks


def build_messages(full_img, crop_img, prompt_fn=zeroshot_prompt) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Judge whether this flagged region is a true hemorrhage lesion."},
            {"type": "text", "text": "Image 1 (full slice):"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_url(full_img)}},
            {"type": "text", "text": "Image 2 (zoomed crop):"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_url(crop_img)}},
            {"type": "text", "text": prompt_fn()},
        ],
    }]


def build_messages_example(full_img, crop_img) -> list[dict]:
    """build_messages와 동일한 실제 질의 뒤에 example_prompt를 붙이되, 
    그 앞에 few-shot 데모(build_example_blocks)를 먼저 넣는다. 
    pipeline.classify_one이 build_messages_fn(full_img, crop_img) 형태(인자 2개)로 호출하므로, 
    vlm_is_lesion.py --prompt-variant example에서 이 함수를 build_messages_fn으로 그대로 넘겨쓴다."""
    messages = build_messages(full_img, crop_img, prompt_fn=example_prompt)
    messages[0]["content"] = build_example_blocks() + messages[0]["content"] # 이어 붙이기
    return messages
