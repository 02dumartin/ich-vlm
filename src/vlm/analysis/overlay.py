"""
MBHSeg25 5cls GT / nnUNet 예측(thr0.001, pp) / VLM reader 2cls(is_lesion) 중간 결과 /
VLM reader 최종 결과(5cls)를 슬라이스 단위로 1x4 그리드 PNG로 비교 오버레이한다.
컬러맵/윈도우/LPS 로드는 src/segmentation/nnunet/overlay.py와 동일한 것을 그대로 재사용.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.segmentation.nnunet.overlay import (
    apply_window,
    class_colors,
    class_names,
    cmap_2cls,
    cmap_5cls,
    lesion_color,
    load_lps_array,
    _draw_mask_panel,
)


def load_reader_case_volumes(
    img_path: Path, gt_path: Path, pred_path: Path, vlm_2cls_path: Path, vlm_path: Path,
) -> dict:
    """한 케이스의 CT/GT/nnUNet 예측(thr0.001, pp)/VLM reader 2cls/VLM reader 5cls 결과 볼륨을 한 번에 로드."""
    return {
        "img": load_lps_array(img_path),
        "gt": load_lps_array(gt_path).astype(np.uint8),
        "pred": load_lps_array(pred_path).astype(np.uint8),
        "vlm_2cls": load_lps_array(vlm_2cls_path).astype(np.uint8),
        "vlm": load_lps_array(vlm_path).astype(np.uint8),
    }


def pick_max_lesion_slice(gt_volume: np.ndarray) -> int:
    """GT 기준 병변 픽셀이 가장 많은 z 슬라이스 인덱스를 고른다(notebook show_case와 동일 로직)."""
    fg_per_slice = (gt_volume > 0).sum(axis=(1, 2))
    return int(fg_per_slice.argmax())


def slices_with_any_mask(volumes: dict) -> list[int]:
    """GT/nnUNet 예측/VLM reader 결과 중 하나라도 전경 픽셀이 있는 z 슬라이스 인덱스를 오름차순으로 반환."""
    has_fg = (volumes["gt"] > 0).any(axis=(1, 2)) \
        | (volumes["pred"] > 0).any(axis=(1, 2)) \
        | (volumes["vlm_2cls"] > 0).any(axis=(1, 2)) \
        | (volumes["vlm"] > 0).any(axis=(1, 2))
    return [int(z) for z in np.where(has_fg)[0]]


def plot_slice_reader_grid(
    case_id: str,
    case_idx: int,
    slice_idx: int,
    img_path: Path,
    gt_path: Path,
    pred_path: Path,
    vlm_2cls_path: Path,
    vlm_path: Path,
    vlm_label: str,
    out_dir: Path,
    volumes: dict = None,
) -> Path:
    """
    한 슬라이스에 대해 GT(5cls) / nnUNet 예측(5cls, thr0.001, pp) / VLM reader 2cls(is_lesion)
    중간 결과 / VLM reader 최종 결과(5cls)를 1x4 그리드로 그려 PNG 저장.

    vlm_label: [2,3] 패널 제목에 붙일 VLM reader 구분 이름(예: crop_min_contour, example 등).
    volumes: load_reader_case_volumes()로 미리 로드해 둔 볼륨(같은 케이스 반복 호출 시 재사용).
    없으면 이 함수 안에서 직접 로드.
    """
    if volumes is None:
        volumes = load_reader_case_volumes(img_path, gt_path, pred_path, vlm_2cls_path, vlm_path)

    img = volumes["img"]
    windowed = apply_window(img[slice_idx])
    gt_slice = volumes["gt"][slice_idx]
    pred_slice = volumes["pred"][slice_idx]
    vlm_2cls_slice = volumes["vlm_2cls"][slice_idx]
    vlm_slice = volumes["vlm"][slice_idx]
    n_slice = img.shape[0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    header = f"{case_id} (z={slice_idx}/{n_slice - 1}), case #{case_idx}"
    _draw_mask_panel(axes[0], windowed, gt_slice, f"{header}\nGT mask (5cls)")
    _draw_mask_panel(axes[1], windowed, pred_slice, "Predicted mask (5cls, thr=0.001, pp)")
    _draw_mask_panel(axes[2], windowed, vlm_2cls_slice, f"VLM reader mask (2cls, {vlm_label})",
                      cmap=cmap_2cls, vmax=1)
    _draw_mask_panel(axes[3], windowed, vlm_slice, f"VLM reader mask (5cls, {vlm_label})")

    legend_handles = [mpatches.Patch(color=np.array(rgb) / 255.0, label=class_names[cls])
                       for cls, rgb in class_colors.items()]
    legend_handles.append(mpatches.Patch(color=np.array(lesion_color) / 255.0, label="Lesion"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=11)

    plt.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_z{slice_idx:03d}_case{case_idx:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
