# Physically Existing AI Competition Workspace

이 디렉토리는 대회 전략, 데이터 분석, 실행 진입점을 한곳에 모아 둔 작업 폴더다.

중요 경로:

- 대회 요약 문서: `/home/ubuntu/Desktop/Physically Existing AI Competition/대회정보 정리.md`
- 실제 데이터 루트: `/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/open`
- 구조 인식 메인 학습 스크립트: `/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/train_hybrid_structure_aware_gated_simaux.py`
- 보조 PhysNet 베이스라인: `/home/ubuntu/Desktop/Physically Existing AI Competition/model_file/baseline_model/physnet_multiview_baseline.py`

## 권장 순서

1. `conda activate multiview`
2. `python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/analyze_dataset.py"`
3. `python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid`
4. 필요하면 `python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-physnet`

## 추천 기본 전략

- 1순위: 구조 인식 + pseudo-3D + simulation auxiliary를 같이 쓰는 `hybrid_structure_aware_gated_simaux`
- 2순위: 영상 motion pseudo-target을 쓰는 `physnet_multiview_baseline`
- 최종 제출: 두 모델의 holdout 결과를 비교한 뒤 soft-voting 또는 temperature-scaled blending

세부 전략은 `strategy.md`에 정리했다.
