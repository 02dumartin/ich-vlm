"""
segmentation/vlm 모듈 공용 유틸: LPS 재정렬, CT HU 윈도잉, 5cls 클래스 정의.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

CLASS_NAMES = {
    0: "NONE",
    1: "EDH",  # 경막외출혈
    2: "IPH",  # 뇌실질출혈
    3: "IVH",  # 뇌실내출혈
    4: "SAH",  # 지주막하출혈
    5: "SDH",  # 경막하출혈
}

CLASS_COLORS = {
    1: (198, 68, 66),    # EDH
    2: (76, 71, 199),    # IPH
    3: (171, 43, 171),   # IVH
    4: (168, 181, 112),  # SAH
    5: (4, 136, 133),    # SDH
}


def load_lps_array(path: Path) -> np.ndarray:
    """DICOM 표준 axial view로 보이도록 LPS 재정렬 후 (z, y, x) 배열 반환."""
    img_sitk = sitk.ReadImage(str(path))
    img_lps = sitk.DICOMOrient(img_sitk, "LPS")
    return sitk.GetArrayFromImage(img_lps)


def apply_window(slice_2d: np.ndarray, center: float = 40, width: float = 80) -> np.ndarray:
    """뇌 윈도우(HU center±width/2)로 CT slice를 0~255 uint8로 변환 (PNG 저장/VLM 입력용)."""
    lo, hi = center - width / 2, center + width / 2
    clipped = np.clip(slice_2d, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
