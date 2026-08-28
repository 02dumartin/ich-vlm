"""
PLOS ONE FP figure(S2~S4 Fig, Kundisch A, et al. 2021, doi:10.1371/journal.pone.0260560)에서
is_lesion example_prompt(few-shot)용 이미지를 추출.

원본 TIF(vlm_datasets/fp_examples/*.tif)은 plain HCT 패널 + 색상 히트맵 패널이 격자로 붙어있는
논문 supplementary figure 원본이다. 패널 grid를 균등 분할로 잘라내고, 히트맵 패널에서 warm
color(적/주황, FP로 flag된 위치) 마스크를 뽑은 뒤, 그 옆 plain 패널에 두 가지를 그려 저장한다
(실제 파이프라인의 images["overlay"]/images[crop_mode] 두 장 구성과 맞춤):

1. full 오버레이: plain 패널 전체 위에 warm bbox 사각형(image_crop.draw_single_bbox_overlay와
   같은 스타일)을 그려 저장.
2. crop 컨투어: plain 패널을 bbox 주변으로 crop(margin+min_size, image_crop.crop_min과 동일
   계산)한 뒤 warm mask의 윤곽선(image_crop.crop_min_contour와 같은 방식)을 그려 저장. 단, 실제
   데이터셋(crop_min_contour)의 선 굵기(width=2)보다 얇게 그린다(CONTOUR_LINE_WIDTH) - 예시
   이미지가 실제 crop보다 작아서 굵은 선이 병변 모양 자체를 가릴 수 있어서.

결과: vlm_datasets/fp_examples/processed/{category}_{idx}_full.png / _crop.png
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import find_contours

from src.vlm.image_crop import _min_size_region

FP_DIR = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/ich-vlm/vlm_datasets/fp_examples")
OUT_DIR = FP_DIR / "processed"

CONTOUR_COLOR = (255, 0, 0)
CONTOUR_LINE_WIDTH = 1  # image_crop.crop_min_contour(width=2)보다 얇게

# (파일명, 카테고리, grid(rows, cols), [(plain_row, plain_col, heat_row, heat_col), ...])
# grid 좌표는 원본 figure가 빈 여백 없이 균등 격자로 이어붙어 있다는 관찰에 기반한 근사치.
SOURCES = [
    ("pone.0260560.s003.tif", "calcification", (3, 2), [(0, 0, 0, 1), (1, 0, 1, 1), (2, 0, 2, 1)]),
    ("pone.0260560.s004.tif", "beam_hardening", (2, 2), [(0, 0, 0, 1), (1, 0, 1, 1)]),
    ("pone.0260560.s005.tif", "tumor", (2, 4), [(0, 0, 0, 1), (0, 2, 0, 3), (1, 0, 1, 1), (1, 2, 1, 3)]),
]


def _panel(img: np.ndarray, grid: tuple, row: int, col: int) -> np.ndarray:
    rows, cols = grid
    h, w = img.shape[:2]
    ph, pw = h // rows, w // cols
    return img[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw]


def _warm_mask(heat_panel: np.ndarray) -> np.ndarray:
    """jet/turbo 계열 컬러맵에서 적/주황(가장 뜨거운) 영역의 boolean mask."""
    r, b = heat_panel[..., 0].astype(int), heat_panel[..., 2].astype(int)
    return (r - b > 60) & (r > 150)


def _bbox_from_mask(mask: np.ndarray) -> tuple | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _draw_full_overlay(panel: np.ndarray, bbox: tuple) -> Image.Image:
    img = Image.fromarray(panel.copy())
    draw = ImageDraw.Draw(img)
    min_row, min_col, max_row, max_col = bbox
    draw.rectangle([min_col, min_row, max_col, max_row], outline=CONTOUR_COLOR, width=2)
    return img


def _draw_crop_contour(panel: np.ndarray, mask: np.ndarray, bbox: tuple, margin: int, min_size: int) -> Image.Image:
    r0, c0, r1, c1 = _min_size_region(bbox, panel.shape[:2], margin=margin, min_size=min_size)
    crop_panel = panel[r0:r1, c0:c1]
    crop_mask = mask[r0:r1, c0:c1]

    img = Image.fromarray(crop_panel.copy())
    draw = ImageDraw.Draw(img)
    for contour in find_contours(crop_mask.astype(np.uint8), level=0.5):
        points = [(x, y) for y, x in contour]  # find_contours는 (row, col) 순서
        if len(points) >= 2:
            draw.line(points, fill=CONTOUR_COLOR, width=CONTOUR_LINE_WIDTH)
    return img


def extract_all(margin: int = 15, min_size: int = 80) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for fname, category, grid, pairs in SOURCES:
        img = np.array(Image.open(FP_DIR / fname).convert("RGB"))
        for idx, (pr, pc, hr, hc) in enumerate(pairs):
            plain = _panel(img, grid, pr, pc)
            heat = _panel(img, grid, hr, hc)
            mask = _warm_mask(heat)
            bbox = _bbox_from_mask(mask)
            entry = {"category": category, "idx": idx, "source": fname, "bbox": bbox}

            if bbox is None:
                print(f"[경고] {fname} panel {idx} ({category}): warm 영역을 못 찾음, 스킵")
                results.append({**entry, "status": "skipped"})
                continue

            full_img = _draw_full_overlay(plain, bbox)
            crop_img = _draw_crop_contour(plain, mask, bbox, margin, min_size)
            full_path = OUT_DIR / f"{category}_{idx}_full.png"
            crop_path = OUT_DIR / f"{category}_{idx}_crop.png"
            full_img.save(full_path)
            crop_img.save(crop_path)
            results.append({**entry, "status": "ok", "full_path": str(full_path), "crop_path": str(crop_path),
                             "full_size": full_img.size, "crop_size": crop_img.size})
    return results


if __name__ == "__main__":
    for r in extract_all():
        print(r)
