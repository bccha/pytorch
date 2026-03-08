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
  - `06_mnist_fc.py`: MNIST 2-Layer FC 신경망 분류 실습
  - `07_mnist_cnn.py`: MNIST CNN (합성곱) 분류 실습 및 모델 저장(`save`)
  - `08_custom_image_dataset.py`: `ImageFolder`를 활용한 커스텀 데이터셋(개미/벌) 변환 실습
  - `09_custom_cnn_ants_bees.py`: 커스텀 데이터셋을 스크래치 CNN 모델로 학습하여 오버피팅 확인 실습
  - `10_transfer_learning_resnet.py`: ResNet18 사전 학습 모델의 파인 튜닝(Fine-Tuning) 실습
  - `10_2_test_original_resnet.py`: 순정 ResNet18 모델의 개미/벌 분류 정확도 테스트 실습
  - `11_rnn_sentiment_analysis.py`: 단어 임베딩과 RNN을 활용한 영화 리뷰 감성(긍정/부정) 분류 기초 실습
- `data/`: 학습 데이터셋
- `models/`: 학습 완료된 모델 가중치 파일
