"""nnUNet 예측 결과의 클래스 라벨을 재구성하는 유틸.

다중 클래스(예: 5cls) 예측을 이진(배경/전경) 라벨로 합칠 때 사용.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def relabel_multiclass_to_binary(src_dir: Path, dst_dir: Path, skip_existing: bool = True):
    """src_dir의 nii.gz 예측(라벨 1..N)을 전경=1, 배경=0인 이진 라벨로 변환해 dst_dir에 저장한다.
    이미 변환된 파일이 있으면 건너뛴다."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_path in sorted(src_dir.glob("*.nii.gz")):
        dst_path = dst_dir / src_path.name
        if skip_existing and dst_path.exists():
            continue
        img = nib.load(str(src_path))
        arr = np.asarray(img.dataobj)
        binarized = (arr > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(binarized, img.affine, img.header), str(dst_path))
    print(f"[relabel] {src_dir} -> {dst_dir} 완료")