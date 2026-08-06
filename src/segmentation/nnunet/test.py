"""nnUNetv2 예측 -> 평가 파이프라인 공통 함수.

scripts/nnUNet_test_5cls.py, scripts/nnUNet_test_1cls.py 
nnUNetv2 CLI 실행 + summary.json 집계 로직
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_BIN_DIR = Path(sys.exec_prefix) / "bin"
PREDICT_BIN = _BIN_DIR / "nnUNetv2_predict"
FIND_BEST_BIN = _BIN_DIR / "nnUNetv2_find_best_configuration"
ENSEMBLE_BIN = _BIN_DIR / "nnUNetv2_ensemble"
EVALUATE_BIN = _BIN_DIR / "nnUNetv2_evaluate_folder"
APPLY_PP_BIN = _BIN_DIR / "nnUNetv2_apply_postprocessing"


@dataclass(frozen=True)
class NNUNetConfig:
    nnunet_root: Path
    dataset_id: int
    dataset_name: str
    trainer: str = "nnUNetTrainer"
    plans: str = "nnUNetPlans"
    folds: tuple = (0, 1, 2, 3, 4)

    def set_env(self):
        os.environ["nnUNet_raw"] = str(self.nnunet_root / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(self.nnunet_root / "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = str(self.nnunet_root / "nnUNet_results")


def run_predict(cfg: NNUNetConfig, input_dir: Path, output_dir: Path, config: str,
                 gpu_id: str, save_probabilities: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PREDICT_BIN),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(cfg.dataset_id),
        "-c", config,
        "-tr", cfg.trainer,
        "-p", cfg.plans,
        "-f", *[str(f) for f in cfg.folds],
        "-device", "cuda",
    ]
    if save_probabilities:
        cmd.append("--save_probabilities")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"RUN (GPU {gpu_id}):", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def run_evaluate_folder(cfg: NNUNetConfig, gt_dir: Path, pred_dir: Path) -> Path:
    """nnUNetv2_evaluate_folder로 pred_dir/summary.json을 생성한다."""
    dataset_json = cfg.nnunet_root / "nnUNet_preprocessed" / cfg.dataset_name / "dataset.json"
    plans_json = cfg.nnunet_root / "nnUNet_preprocessed" / cfg.dataset_name / f"{cfg.plans}.json"
    out_json = pred_dir / "summary.json"
    cmd = [
        str(EVALUATE_BIN),
        str(gt_dir), str(pred_dir),
        "-djfile", str(dataset_json),
        "-pfile", str(plans_json),
        "-o", str(out_json),
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return out_json


def run_find_best_config(cfg: NNUNetConfig, configs=("2d", "3d_fullres")) -> Optional[dict]:
    """nnUNetv2_find_best_configuration 실행 후, 그 결과인 inference_information.json(공식 결정
    파일: postprocessing까지 반영된 최종 승자)을 그대로 읽어 config 조합과 postprocessing 정보를
    반환한다. 실패/결과 없으면 None.

    configs로 후보를 좁히면(예: ("2d",)) 그 config 단독의 postprocessing 적용 여부만 판단한다
    (후보가 1개면 앙상블은 불가능). inference_information.json은 호출할 때마다 덮어써지지만
    postprocessing.pkl 자체는 config별 경로에 저장되므로, 반환값을 즉시 저장해두면 안전하다."""
    cmd = [
        str(FIND_BEST_BIN), str(cfg.dataset_id),
        "-c", *configs,
        "-tr", cfg.trainer, "-p", cfg.plans,
        "-f", *[str(f) for f in cfg.folds],
    ]
    print("RUN:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        print(f"[find_best_configuration 실패, 개별 결과만 사용] {e}")
        return None

    info_path = cfg.nnunet_root / "nnUNet_results" / cfg.dataset_name / "inference_information.json"
    if not info_path.exists():
        print("[경고] inference_information.json 없음")
        return None

    with open(info_path) as f:
        info = json.load(f)

    best = info["best_model_or_ensemble"]
    return {
        "configs": [m["configuration"] for m in best["selected_model_or_models"]],
        "postprocessing_pkl": Path(best["postprocessing_file"]),
        "plans_json": Path(best["some_plans_file"]),
    }


def run_apply_postprocessing(pred_dir: Path, output_dir: Path, pp_pkl: Path, plans_json: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(APPLY_PP_BIN),
        "-i", str(pred_dir),
        "-o", str(output_dir),
        "-pp_pkl_file", str(pp_pkl),
        "-np", "8",
        "-plans_json", str(plans_json),
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())


def run_best_config_predict(cfg: NNUNetConfig, configs: list, images_dir: Path, output_dir: Path, gpu_id: str):
    if len(configs) == 1:
        run_predict(cfg, images_dir, output_dir, configs[0], gpu_id)
        return

    prob_dirs = []
    for config in configs:
        prob_dir = output_dir.parent / f"_{config}_probs" / output_dir.name
        run_predict(cfg, images_dir, prob_dir, config, gpu_id, save_probabilities=True)
        prob_dirs.append(prob_dir)

    cmd = [
        str(ENSEMBLE_BIN),
        "-i", *[str(d) for d in prob_dirs],
        "-o", str(output_dir),
        "-np", "4",
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())


def predict_and_evaluate(cfg: NNUNetConfig, images_dir: Path, pred_dir: Path, labels_dir: Path,
                          config: str, gpu_id: str, save_probabilities: bool = False,
                          skip_existing: bool = True) -> Path:
    """summary.json이 이미 있으면 예측/평가를 건너뛰고 기존 경로만 반환한다."""
    summary_path = pred_dir / "summary.json"
    if skip_existing and summary_path.exists():
        print(f"[건너뜀] 이미 완료됨: {pred_dir}")
        return summary_path
    run_predict(cfg, images_dir, pred_dir, config, gpu_id, save_probabilities=save_probabilities)
    return run_evaluate_folder(cfg, labels_dir, pred_dir)


def best_config_predict_and_evaluate(cfg: NNUNetConfig, configs: list, images_dir: Path, pred_dir: Path,
                                      labels_dir: Path, gpu_id: str, skip_existing: bool = True) -> Path:
    """summary.json이 이미 있으면 예측/평가를 건너뛰고 기존 경로만 반환한다."""
    summary_path = pred_dir / "summary.json"
    if skip_existing and summary_path.exists():
        print(f"[건너뜀] 이미 완료됨: {pred_dir}")
        return summary_path
    run_best_config_predict(cfg, configs, images_dir, pred_dir, gpu_id)
    return run_evaluate_folder(cfg, labels_dir, pred_dir)


def postprocess_and_evaluate(cfg: NNUNetConfig, pred_dir: Path, pp_dir: Path, labels_dir: Path,
                              pp_pkl: Path, plans_json: Path, skip_existing: bool = True) -> Path:
    """summary.json이 이미 있으면 postprocessing 적용/평가를 건너뛰고 기존 경로만 반환한다."""
    summary_path = pp_dir / "summary.json"
    if skip_existing and summary_path.exists():
        print(f"[건너뜀] 이미 완료됨: {pp_dir}")
        return summary_path
    run_apply_postprocessing(pred_dir, pp_dir, pp_pkl, plans_json)
    return run_evaluate_folder(cfg, labels_dir, pp_dir)


def summary_to_row(summary_path: Path, test_name: str, row_label: str, class_names: dict) -> dict:
    """summary.json 값을 그대로 사용해 클래스별 Dice/IoU/Precision/Recall/n_cases를 뽑는다.
    전체 케이스의 TP/FP/FN을 합산해 한 번에 계산(micro-average)하지 않고,
    Dice/IoU와 동일하게 케이스별 값을 각각 구해 평균한다(macro-average).
    n_cases는 예측 결과와 무관하게, GT에 해당 병변이 실제로 존재하는(n_ref>0) 케이스 수."""
    with open(summary_path) as f:
        data = json.load(f)

    per_class = {
        name: {"dice": [], "iou": [], "precision": [], "recall": [], "n": 0}
        for name in class_names.values()
    }
    for case in data["metric_per_case"]:
        for cls, name in class_names.items():
            m = case["metrics"].get(str(cls))
            if m is None:
                continue
            stat = per_class[name]
            dice = m.get("Dice")
            if dice is not None and not (isinstance(dice, float) and np.isnan(dice)):
                stat["dice"].append(dice)
            iou = m.get("IoU")
            if iou is not None and not (isinstance(iou, float) and np.isnan(iou)):
                stat["iou"].append(iou)

            precision = m.get("Precision")
            if precision is None and m.get("TP") is not None and m.get("FP") is not None:
                tp, fp = m["TP"], m["FP"]
                precision = tp / (tp + fp) if (tp + fp) > 0 else None
            if precision is not None:
                stat["precision"].append(precision)

            recall = m.get("Recall")
            if recall is None and m.get("TP") is not None and m.get("FN") is not None:
                tp, fn = m["TP"], m["FN"]
                recall = tp / (tp + fn) if (tp + fn) > 0 else None
            if recall is not None:
                stat["recall"].append(recall)

            n_ref = m.get("n_ref")
            if n_ref is not None and n_ref > 0:
                stat["n"] += 1

    row = {"Test Dataset": test_name, "Dimension": row_label}
    for name, stat in per_class.items():
        row[f"{name}_Dice"] = float(np.mean(stat["dice"])) if stat["dice"] else float("nan")
        row[f"{name}_IoU"] = float(np.mean(stat["iou"])) if stat["iou"] else float("nan")
        row[f"{name}_Precision"] = float(np.mean(stat["precision"])) if stat["precision"] else float("nan")
        row[f"{name}_Recall"] = float(np.mean(stat["recall"])) if stat["recall"] else float("nan")
        row[f"{name}_n_cases"] = stat["n"]
    return row


def add_macro_average(row: dict, class_names: dict, metrics=("Dice", "IoU")) -> dict:
    """클래스별 값의 macro-average를 row에 추가한다 (예: mDice, mIoU).
    클래스가 여럿일 때만 의미가 있다 (1cls에서는 호출하지 않는다)."""
    for metric in metrics:
        values = [row[f"{name}_{metric}"] for name in class_names.values()]
        row[f"m{metric}"] = float(np.nanmean(values))
    return row


def build_results_df(results: list, class_names: dict, macro_metrics: tuple = ()) -> pd.DataFrame:
    class_cols = []
    for name in class_names.values():
        class_cols += [f"{name}_Dice", f"{name}_IoU", f"{name}_Precision", f"{name}_Recall", f"{name}_n_cases"]
    macro_cols = [f"m{metric}" for metric in macro_metrics]

    df = pd.DataFrame(results)[["Test Dataset", "Dimension"] + macro_cols + class_cols]
    n_cases_cols = [c for c in df.columns if c.endswith("_n_cases")]
    float_cols = [c for c in df.columns if c not in ["Test Dataset", "Dimension"] + n_cases_cols]
    df[float_cols] = df[float_cols].round(3)
    df[n_cases_cols] = df[n_cases_cols].astype(int)
    return df


def save_or_merge_csv(results_df: pd.DataFrame, out_csv: Path,
                       key_cols=("Test Dataset", "Dimension")) -> pd.DataFrame:
    """기존 CSV 중 이번에 실행한 (key_cols) 조합만 새 값으로 교체하고 나머지는 유지해서 저장한다."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    key_cols = list(key_cols)
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        new_keys = set(map(tuple, results_df[key_cols].values.tolist()))
        existing_df = existing_df[~existing_df[key_cols].apply(tuple, axis=1).isin(new_keys)]
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(out_csv, index=False)
    return results_df
