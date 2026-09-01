"""
PLOS ONE FP figure(S2~S4 Fig, Kundisch A, et al. 2021, doi:10.1371/journal.pone.0260560)에서
is_lesion example_prompt(few-shot)용 이미지를 추출.

원본 TIF(vlm_datasets/fp_examples/*.tif)은 plain HCT 패널 + 색상 히트맵 패널이 격자로 붙어있는
논문 supplementary figure 원본이다. 패널 grid를 균등 분할로 잘라내는데, plain/heat 두 패널이
원본에서 정확히 대칭으로 안 붙어있어서(예: s003 calcification_0은 두 패널이 가로로 약 27px
어긋남 - 실측 확인함) 히트맵에서 찾은 좌표를 그대로 plain 패널에 옮기면 실제 병변 위치에서
벗어난다. 처음엔 각 패널의 두개골 링(밝은 큰 연결영역)의 bbox로 정합을 시도했는데, 일부 패널이
원본의 흰 여백까지 함께 크롭되면서 "가장 밝은 연결영역"이 여백째로 잡혀 엉뚱한 스케일이 나오는
문제가 있었다(예: calcification_2, beam_hardening_1, tumor_2). 그래서 대신 두 패널 전체의
밝기 패턴을 skimage.registration.phase_cross_correlation으로 직접 상관관계 매칭해 이동량만
구한다(회전/스케일 없음 - 같은 슬라이스를 같은 배율로 보여주는 두 패널이라 순수 이동만 있음).
여백처럼 국소적인 이상치에 흔들리지 않고 패널 전체 anatomy 패턴을 보고 맞추므로 더 안정적이다.

패널 grid를 균등 분할로 1차로 잘라낸 뒤, 히트맵 패널에서 warm color(적/주황, FP로 flag된 위치)
마스크를 뽑고, 위 정합을 거쳐 그 옆 plain 패널에 두 가지를 그려 저장한다(실제 파이프라인의
images["overlay"]/images[crop_mode] 두 장 구성과 맞춤):

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
from scipy.ndimage import shift as ndi_shift
from skimage.measure import find_contours
from skimage.registration import phase_cross_correlation

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


def _align_mask_to_plain(mask_heat: np.ndarray, plain_panel: np.ndarray, heat_panel: np.ndarray) -> np.ndarray:
    """heat 패널 좌표계의 mask를 plain 패널 좌표계로 옮긴다. 두 패널 전체의 밝기(luminance)
    패턴을 phase_cross_correlation으로 직접 상관관계 매칭해 이동량(dy, dx)만 구한다(회전/스케일
    없음 - 같은 슬라이스를 같은 배율로 보여주는 두 패널이라 순수 이동만 있음). 국소적인 밝은
    영역(두개골 링 등) 하나만 보고 정합하면 원본의 흰 여백이 함께 잡히는 패널에서 엉뚱하게
    어긋나는 문제가 있어서, 패널 전체 anatomy 패턴을 보는 이 방식으로 바꿨다."""
    plain_gray = plain_panel.astype(float).mean(axis=-1)
    heat_gray = heat_panel.astype(float).mean(axis=-1)
    (dy, dx), _error, _diffphase = phase_cross_correlation(plain_gray, heat_gray, upsample_factor=4)

    # (ys+dy, xs+dx)를 반올림해 한 점씩 찍는 forward mapping은, dy/dx의 소수부가 일정하게
    # 반올림되면서 결과 mask에 체크보드형 구멍이 뚫리는 문제가 있었다(원래 꽉 찬 덩어리인데
    # 격자무늬로 듬성듬성해짐 - find_contours가 그 구멍 하나하나의 테두리까지 다 그려서 빗살무늬가
    # 됨). scipy.ndimage.shift는 출력 픽셀마다 원본에서 값을 가져오는 inverse mapping이라
    # 이런 구멍이 생기지 않는다.
    shifted = ndi_shift(mask_heat.astype(np.uint8), shift=(dy, dx), order=0, mode="constant", cval=0)
    return shifted > 0


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
            mask_heat = _warm_mask(heat)
            mask = _align_mask_to_plain(mask_heat, plain, heat)
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
