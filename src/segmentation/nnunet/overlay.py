"""scan/slice 단위 GT vs threshold별 예측 오버레이 이미지 생성.

notebook/nnUNet_result.ipynb의 show_case/show_case_thr와 동일한 방식(LPS 재정렬,
window/level, 5cls 컬러맵, 반투명 마스크 오버레이)을 재사용해서, 한 슬라이스에 대해
GT + 여러 threshold 예측을 한 장의 그리드 이미지로 저장한다.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap

class_names = {
    1: "EDH",
    2: "IPH",
    3: "IVH",
    4: "SAH",
    5: "SDH",
}

class_colors = {
    1: (198, 68, 66),    # EDH
    2: (76, 71, 199),    # IPH
    3: (171, 43, 171),   # IVH
    4: (168, 181, 112),  # SAH
    5: (4, 136, 133),    # SDH
}


def build_cmap(color_dict: dict, n_classes: int) -> ListedColormap:
    colors_rgb = np.zeros((n_classes, 3))
    for cls, rgb in color_dict.items():
        colors_rgb[cls] = np.array(rgb) / 255.0
    return ListedColormap(colors_rgb)


cmap_5cls = build_cmap(class_colors, 6)  # 0(background, black)~5


def apply_window(slice_hu: np.ndarray, window_level: float = 40, window_width: float = 80) -> np.ndarray:
    """뇌실질 window(level 40 / width 80)로 HU 값을 0~1로 정규화."""
    lo, hi = window_level - window_width / 2, window_level + window_width / 2
    return (np.clip(slice_hu, lo, hi) - lo) / (hi - lo)


def load_lps_array(path: Path) -> np.ndarray:
    """DICOM 표준 axial view(anterior=위, 방사선과 관례 좌우)로 보이도록 LPS 재정렬 후 (z, y, x) 배열 반환."""
    img_sitk = sitk.ReadImage(str(path))
    img_lps = sitk.DICOMOrient(img_sitk, "LPS")
    return sitk.GetArrayFromImage(img_lps)


def _draw_mask_panel(ax, windowed_slice: np.ndarray, mask_slice: np.ndarray, title: str):
    mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(windowed_slice, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask_overlay, cmap=cmap_5cls, vmin=0, vmax=5, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def load_case_volumes(img_path: Path, gt_path: Path, thr_pred_paths: dict) -> dict:
    """
    한 케이스(스캔)의 CT/GT/threshold별 예측 볼륨을 한 번에 로드.
    같은 케이스의 여러 슬라이스를 그릴 때 plot_slice_threshold_grid에 volumes로 넘겨
    케이스당 한 번만 로드하도록(반복 I/O 방지) 쓰는 용도.
    """
    return {
        "img": load_lps_array(img_path),
        "gt": load_lps_array(gt_path).astype(np.uint8),
        "thr": {thr: load_lps_array(p).astype(np.uint8) for thr, p in thr_pred_paths.items()},
    }


def plot_slice_threshold_grid(
    case_id: str,
    slice_idx: int,
    idx_num: int,
    img_path: Path,
    gt_path: Path,
    thr_pred_paths: dict,
    out_dir: Path,
    volumes: dict = None,
) -> Path:
    """
    한 슬라이스에 대해 GT(5cls) + 여러 threshold 예측(5cls, pp)을 2x3 그리드로 그려 PNG 저장.

    thr_pred_paths: {"0.5": Path(...), "0.1": Path(...), "0.05": Path(...), "0.01": Path(...)}
    grid 배치: (0,0)=GT, (0,1)=thr[0], (0,2)=thr[1], (1,0)=공란, (1,1)=thr[2], (1,2)=thr[3]
    (thr_pred_paths는 순서를 보존한 dict로 넘겨야 함: 0.5, 0.1, 0.05, 0.01 순)

    volumes: load_case_volumes()로 미리 로드해 둔 볼륨(같은 케이스 반복 호출 시 재사용).
    없으면 이 함수 안에서 직접 로드한다.
    """
    if volumes is None:
        volumes = load_case_volumes(img_path, gt_path, thr_pred_paths)

    img = volumes["img"]
    gt = volumes["gt"]
    windowed = apply_window(img[slice_idx])
    gt_slice = gt[slice_idx]
    n_slice = img.shape[0]

    thr_labels = list(thr_pred_paths.keys())
    thr_slices = {thr: volumes["thr"][thr][slice_idx] for thr in thr_labels}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(windowed, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap=cmap_5cls, vmin=0, vmax=5, alpha=0.6)
    axes[0, 0].set_title(f"{case_id}(z={slice_idx}/{n_slice - 1}), {idx_num}\nGT mask(5cls)", fontsize=10)
    axes[0, 0].axis("off")

    grid_pos = {thr_labels[0]: (0, 1), thr_labels[1]: (0, 2), thr_labels[2]: (1, 1), thr_labels[3]: (1, 2)}
    for thr, (r, c) in grid_pos.items():
        _draw_mask_panel(axes[r, c], windowed, thr_slices[thr], f"Predicted mask(5cls, thr={thr}, pp)")

    axes[1, 0].axis("off")

    legend_handles = [mpatches.Patch(color=np.array(rgb) / 255.0, label=class_names[cls])
                       for cls, rgb in class_colors.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(class_names),
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=11)

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_z{slice_idx:03d}_idx{idx_num:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
