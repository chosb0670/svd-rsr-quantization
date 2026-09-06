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
