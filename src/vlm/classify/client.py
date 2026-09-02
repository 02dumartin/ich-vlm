"""
vLLM(OpenAI 호환) 클라이언트 생성과, 이미지/텍스트 응답 변환에 쓰는 공용 헬퍼.
"""

import base64
from io import BytesIO

import numpy as np
from openai import OpenAI
from PIL import Image

GEN_CONFIGS = {
    "non_thinking": dict(
        temperature=0.0, max_tokens=1024, seed=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    # Gemma3 계열(medgemma) chat template엔 enable_thinking 스위치가 없어서 non_thinking의
    # extra_body를 그대로 넘기면 템플릿 렌더링이 깨질 수 있음 - extra_body 없이 순수 생성
    # 파라미터만 사용.
    "medgemma_default": dict(temperature=0.0, max_tokens=1024, seed=0),
}


def get_client(base_url: str, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def to_rgb_pil(img) -> Image.Image:
    """np.ndarray / PIL / 경로 어떤 형태로 들어와도 RGB PIL로 통일."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return Image.fromarray(img).convert("RGB")
    if isinstance(img, (str, bytes)):
        return Image.open(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def pil_to_base64_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_first_json(text: str) -> str:
    """모델 출력에서 첫 번째 완결된 JSON 객체만 잘라냄 (중첩 괄호 대응)."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON found: {text}")
    depth, end = 0, None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f"Unterminated JSON: {text}")
    return text[start:end]
