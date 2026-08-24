ich-vlm

Result 정리 규칙

실험 결과는 날짜/실험명/버전 기준으로 폴더를 구성한다.

폴더명

  {YYMMDD}_{실험명}_v{NN}

- 실험명: 공백 대신 대시(-) 사용, 동사-대상-데이터셋 형식 권장
  예: nnunet-train-mbhseg25, nnunet-infer-5cls-test-all
  Experiment_Record_HJ.xlsx의 실험 이름 컬럼 값과 동일하게 맞춘다. 엑셀 행과 폴더를 실험명으로 매칭한다.
- 버전(v01, v02...): 같은 실험을 코드 수정 후 다시 돌릴 때 증가
  예: postprocessing 버그 수정 후 재실행, v01에서 v02로

폴더 구조

  {폴더명}/
    input/    원본 데이터 직접 복사 X, 경로만 README에 기록하거나 심볼릭 링크만
    output/   summary.json, scan_eval.csv, 예측 nii.gz 등 결과물
    log/      실행 로그
    README.md 실행 명령어, 사용 데이터, GPU, 변수 설명, 사용한 코드의 저장소 내 경로

코드는 이 폴더에 스냅샷으로 복사하지 않는다. 대신 README에 저장소(ich-vlm) 안의 스크립트 경로를 적어서, 실행 당시 버전은 git 커밋 이력으로 추적한다.

실험 기록, 엑셀

- scripts/Experiment_Record_HJ.xlsx 에 수기로 기록
- 컬럼: 일자 / 데이터 / 실험 이름 / 실험 종류 / Class / 세팅 / 환경 / 진행상태
- 한 실험이 여러 데이터셋에 걸치면, 데이터셋별로 행을 나눠서 같은 실험 이름을 반복해서 적는다.
  한 셀에 데이터셋 여러 개를 콤마로 넣거나 멀티셀렉트 드롭다운을 쓰지 않는다. 필터/피벗이 안 되고, 파일이 Google Drive로 동기화될 예정이라 VBA 매크로 기반 멀티셀렉트는 Google Sheets에서 깨진다.
- 데이터셋 목록은 Dataset 시트에서 관리

결과 보관

- Google Drive(rclone 연동 예정) 공유 드라이브
- rclone은 요청 시에만 진행

문서 작성 규칙

- README와 CLAUDE.md에는 따옴표(홑따옴표, 겹따옴표, 백틱)를 쓰지 않는다.

코드 네이밍 규칙, eval과 analysis 구분

- eval: nnUNet 공식 evaluate_folder를 그대로 감싸거나, 단일 지표(Dice/IoU/Precision/Recall) 산출에 집중하는 모듈/스크립트
  예: slice_eval.py의 build_scan_df, nnUNet_infer_thr_sweep_5cls.py
- analysis: TP/FP/FN 세부 분해, error_type 분류처럼 원인 진단을 위해 지표를 더 풀어서 보는 모듈/스크립트
  예: slice_eval.py의 build_scan_analysis_df, build_slice_analysis_df, nnUNet_scan_slice_analysis_5cls.py
- 기존 slice_eval.py, nnUNet_slice_eval.py 파일명은 과거 실험 폴더 README와 커밋 이력에서 코드 경로로 참조되고 있어 리네임하지 않는다.
  새로 만드는 모듈/스크립트만 이 기준으로 eval 또는 analysis를 붙인다.
