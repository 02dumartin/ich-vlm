"""
VLM 입력용 crop 이미지 4종 + 풀사이즈 bbox overlay.

crop 4종은 전부 dataset_prepare 시점에 미리 만들어 vlm_datasets/에 저장해두고,
어떤 crop을 VLM에 넣을지는 classify 단계에서 --crop-mode로 그 경로만 골라 쓴다
(이 파일에서는 매번 다시 만들지 않음).
"""

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import find_contours

CROP_MODES = ("crop_only", "crop_min", "crop_min_bbox", "crop_min_contour")


def draw_single_bbox_overlay(image_2d: np.ndarray, bbox: tuple, color: tuple = (255, 0, 0)) -> Image.Image:
    """이 인스턴스의 pred bbox 하나만 그린다 - GT는 표시하지 않음."""
    rgb = np.stack([image_2d] * 3, axis=-1).astype(np.uint8)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    min_row, min_col, max_row, max_col = bbox
    draw.rectangle([min_col, min_row, max_col, max_row], outline=color, width=2)
    return pil_img


def crop_only(image_2d: np.ndarray, bbox: tuple, margin: int = 10) -> Image.Image:
    """bbox+margin으로 자르기만 함 (최소 크기 보장 없음, 그리기 없음)."""
    min_row, min_col, max_row, max_col = bbox
    h, w = image_2d.shape
    r0, r1 = max(0, min_row - margin), min(h, max_row + margin)
    c0, c1 = max(0, min_col - margin), min(w, max_col + margin)
    return Image.fromarray(image_2d[r0:r1, c0:c1].astype(np.uint8))


def _min_size_region(bbox: tuple, image_shape: tuple, margin: int = 10, min_size: int = 30) -> tuple[int, int, int, int]:
    """bbox+margin 크기가 min_size(가로/세로 각각)보다 작으면 bbox 중심 기준으로
    min_size까지 확장한다. 이미지 경계를 벗어나면 clip."""
    min_row, min_col, max_row, max_col = bbox
    h, w = image_shape

    r0, r1 = min_row - margin, max_row + margin
    c0, c1 = min_col - margin, max_col + margin

    if r1 - r0 < min_size:
        center = (r0 + r1) / 2
        r0, r1 = center - min_size / 2, center + min_size / 2
    if c1 - c0 < min_size:
        center = (c0 + c1) / 2
        c0, c1 = center - min_size / 2, center + min_size / 2

    r0, r1 = max(0, int(round(r0))), min(h, int(round(r1)))
    c0, c1 = max(0, int(round(c0))), min(w, int(round(c1)))
    return r0, c0, r1, c1


def crop_min(image_2d: np.ndarray, bbox: tuple, margin: int = 10, min_size: int = 30) -> Image.Image:
    """crop_only와 같되, 잘린 영역이 min_size보다 작아지지 않도록 보장."""
    r0, c0, r1, c1 = _min_size_region(bbox, image_2d.shape, margin, min_size)
    return Image.fromarray(image_2d[r0:r1, c0:c1].astype(np.uint8))


def crop_min_bbox(image_2d: np.ndarray, bbox: tuple, margin: int = 10, min_size: int = 30,
                   color: tuple = (255, 0, 0)) -> Image.Image:
    """crop_min 위에 원본 bbox 사각형을 크롭 좌표계로 옮겨 그림."""
    r0, c0, r1, c1 = _min_size_region(bbox, image_2d.shape, margin, min_size)
    min_row, min_col, max_row, max_col = bbox

    rgb = np.stack([image_2d[r0:r1, c0:c1]] * 3, axis=-1).astype(np.uint8)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([min_col - c0, min_row - r0, max_col - c0, max_row - r0], outline=color, width=2)
    return pil_img


def crop_min_contour(image_2d: np.ndarray, bbox: tuple, instance_mask: np.ndarray, margin: int = 10,
                      min_size: int = 30, color: tuple = (255, 0, 0)) -> Image.Image:
    """crop_min 위에 instance_mask(해당 인스턴스 픽셀만)의 윤곽선을 그림."""
    r0, c0, r1, c1 = _min_size_region(bbox, image_2d.shape, margin, min_size)

    rgb = np.stack([image_2d[r0:r1, c0:c1]] * 3, axis=-1).astype(np.uint8)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    mask_crop = instance_mask[r0:r1, c0:c1]
    for contour in find_contours(mask_crop.astype(np.uint8), level=0.5):
        points = [(x, y) for y, x in contour]  # find_contours는 (row, col) 순서
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
    return pil_img


def crop_image(image_2d: np.ndarray, bbox: tuple, mode: str, instance_mask: np.ndarray = None,
               margin: int = 10, min_size: int = 30) -> Image.Image:
    """crop_mode 문자열로 위 4개 함수 중 하나를 디스패치."""
    if mode == "crop_only":
        return crop_only(image_2d, bbox, margin=margin)
    if mode == "crop_min":
        return crop_min(image_2d, bbox, margin=margin, min_size=min_size)
    if mode == "crop_min_bbox":
        return crop_min_bbox(image_2d, bbox, margin=margin, min_size=min_size)
    if mode == "crop_min_contour":
        if instance_mask is None:
            raise ValueError("crop_min_contour 모드는 instance_mask가 필요합니다")
        return crop_min_contour(image_2d, bbox, instance_mask, margin=margin, min_size=min_size)
    raise ValueError(f"알 수 없는 crop mode: {mode} (허용: {CROP_MODES})")
