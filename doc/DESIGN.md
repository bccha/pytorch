# 프로젝트 설계 (DESIGN)

## 1. 개요
PyTorch 학습 과정에서 생성되는 실습 코드의 구조와 설계를 기록합니다.

## 2. 디렉토리 구조
- `doc/`: 문서 폴더
- `src/`: 실습 코드 소스 (각 챕터별 독립적인 스크립트로 분할하여 관리)
  - `01_tensor_basics.py`: 텐서 기본 및 연산 실습
  - `02_autograd.py`: 자동 미분 및 연산 그래프 실습
  - `03_neural_network.py`: nn.Module 기반 모델 설계 실습
  - `04_training_loop.py`: Optimizer 선언 및 5단계 학습 루프 실습
  - `05_dataset_dataloader.py`: Custom Dataset 제작 및 DataLoader 실습
- `data/`: 학습 데이터셋
- `models/`: 학습 완료된 모델 가중치 파일
