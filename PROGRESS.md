# PROGRESS.md

프로젝트 진행 상황 기록. 의미 있는 진행(설정 완료, 실험 결과, 중요한 결정)이 생길 때마다 날짜와 함께 추가한다.

## 2026-09-06

- **환경 세팅**: `venv`에 Python 3.14 + PyTorch 2.14.0+cu126 설치 확인, CUDA GPU 사용 가능 확인.
- **베이스라인 스크립트 작성**: `models/vgg16_cifar10_baseline.py` 추가.
  - CIFAR-10(32x32)에 맞게 조정한 VGG16 (5단계 conv+BN+ReLU+maxpool, 3-layer FC 분류기).
  - 방어 기법 없이 순수 SGD(momentum, weight decay, cosine LR)로 학습.
  - 학습 종료 후 최종 test accuracy 출력, 가중치를 `results/`에 저장.
  - 추후 SVD/RSR/quantization 방어 기법 적용 버전과 비교할 undefended baseline 역할.
- **1 에폭 테스트 실행 결과**: 스크립트 정상 동작 및 GPU 사용 확인.
  - device: `cuda`
  - train_loss=2.1512, train_acc=15.68% (1 epoch, 60.1초)
  - 최종 test accuracy: 20.05%
  - 체크포인트 저장 확인: `results/vgg16_cifar10_baseline.pth`
  - 에러 없이 exit code 0으로 종료.
- **Git**: 베이스라인 스크립트 커밋 및 GitHub(`origin/main`)에 푸시 완료.
- **3 에폭 테스트**: 에폭당 약 61~79초, test accuracy 62.46% (3 epoch만).
- **체크포인트/best-model 기능 추가**: `models/vgg16_cifar10_baseline.py`에 10에폭마다 체크포인트 저장, test accuracy가 갱신될 때마다 `results/best_model.pth`로 별도 저장하는 기능 추가.
- **100 에폭 전체 학습 1차 시도 실패**: 백그라운드 실행 중 약 62~63 에폭 부근에서 시스템 메모리 부족으로 프로세스가 강제 종료됨(exit code 1). 원인 추정: 매 에폭마다 DataLoader worker(`num_workers=4`)를 재생성하는 구조가 Windows 환경에서 장시간 실행 시 메모리를 서서히 누적시킴. 그 시점까지 `checkpoint_epoch60.pth`와 `best_model.pth`(test accuracy 88.74%)는 정상 저장되어 있었음.
- **`--resume` 기능 추가 및 메모리 완화 후 재개**: `num_workers` 기본값을 2로 낮추고 `persistent_workers=True` 적용, 체크포인트에 optimizer/scheduler 상태까지 포함해 저장하도록 확장, `--resume`/`--start-epoch`/`--initial-best-acc` 옵션으로 기존 체크포인트에서 이어서 학습 가능하게 함. CosineAnnealingLR을 이어서 재현할 때는 `last_epoch` 파라미터로 바로 점프하면 잘못된 LR이 나온다는 것을 스모크 테스트로 확인해, `scheduler.step()`을 이어받은 에폭 수만큼 재생(replay)하는 방식으로 수정함.
- **100 에폭 완료** (`checkpoint_epoch60.pth`에서 61~100 에폭 이어 학습):
  - epoch 100 시점 test accuracy: **93.80%**
  - 100 에폭 중 최고 기록(`best_model.pth`, 약 97 에폭 부근): **93.83%**
  - 순수 학습 시간 합계 약 3시간 35분 (1차 실행 60에폭 ≈ 2시간 23분 + 재개 40에폭 ≈ 1시간 12분), 최초 시작부터 완료까지 전체 경과 약 3시간 56분.
  - 재개 실행도 완료 직후(에폭 100 체크포인트 저장 후) 한 번 더 메모리 부족으로 프로세스가 죽었으나, 그 시점엔 이미 모든 결과가 저장된 뒤라 실질적 손실 없음. 즉 메모리 누수는 완화됐을 뿐 완전히 해결되진 않음 — 더 긴 학습을 돌릴 경우 `num_workers=0`으로 더 낮추거나 근본 원인을 추가로 조사할 필요 있음.
  - 결과물: `results/vgg16_cifar10_baseline.pth`(최종), `results/best_model.pth`(최고 기록), `results/checkpoints/checkpoint_epoch{10..100}.pth`(10에폭 단위 전체 이력).
- **RSR (SVD 기반 랜덤 재구성) 방어 구현**: `models/rsr.py` 추가.
  - 채널별 SVD 분해 후, 특이값 크기에 비례한 확률로 성분을 비복원추출하여 재구성 → R=10회 독립 시행 → 분류 결과 다수결(majority vote)로 최종 예측.
  - 학습된 `vgg16_cifar10_baseline.pth`(무방어 베이스라인, 93.80%)를 로드해 공격 없는 상태로 테스트셋 10,000장 전체 평가.
  - 무방어 forward pass를 이 스크립트에서 재현해도 93.80%로 일치(비교 신뢰성 확인).
  - **RSR 방어 적용 clean accuracy: 92.55%** (keep_ratio=0.5) → 베이스라인 대비 **-1.25%p**. 랜덤 재구성 기반 방어의 일반적인 클린 정확도 트레이드오프로 판단됨.
  - 공격 상황에서의 강건성은 `attacks/` 코드 준비 후 별도 측정 필요.
