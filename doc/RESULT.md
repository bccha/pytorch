# 학습 결과 (RESULT)

학습하면서 알게 된 핵심 내용이나 실습 결과를 요약하여 기록합니다.

## 1. MNIST 분류 실습 결과
0부터 9까지의 손글씨 숫자를 분류하는 기본 예제(MNIST) 테스트 결과입니다.

### 1-1. 2-Layer FC 신경망 (`src/06_mnist_fc.py`)
* **구조**: `28x28(784) 입력` -> `128 은닉층 (ReLU)` -> `10 출력층`
* **훈련 환경**: CPU / Batch Size: 64 / Epoch: 5
* **옵티마이저**: Adam (lr=0.001) / **손실 함수**: CrossEntropyLoss
* **결과 (Test Accuracy)**: **97.48%**

### 1-2. CNN 신경망 (`src/07_mnist_cnn.py`)
* **구조**: `Conv2d(32)` -> `MaxPool2d` -> `Conv2d(64)` -> `MaxPool2d` -> `FC(128)` -> `FC(10)`
* **훈련 환경**: CPU / Batch Size: 64 / Epoch: 5
* **결과 (Test Accuracy)**: **99.11%**

---

## 2. 이미지 분류 실전: 개미(Ants) vs 벌(Bees)
커스텀 이미지 데이터셋(약 250장의 소량 훈련 데이터) 환경에서, 모델을 바닥부터 훈련시킬 때와 전이 학습(Transfer Learning)을 적용했을 때의 뚜렷한 성능 차이를 비교합니다.

### 2-1. 바닥부터 훈련한 기본 CNN (`src/09_custom_cnn_ants_bees.py`)
* **구조**: `3-Layer Conv2d` + `FC(512)` 백지 상태의 깡통 가중치
* **훈련 환경**: CPU / Batch Size: 4 / Epoch: 10 / Optimizer: Adam(0.001)
* **문제점**: 데이터 수가 적어 쉽게 훈련 데이터에만 외워버리는(Overfitting) 현상이 바로 나타납니다.
* **결과 (Validation Accuracy)**: **대략 60~66% 내외**의 처참한 성적을 보여줍니다. 

### 2-2. 전이 학습 (Transfer Learning) (`src/10_transfer_learning_resnet.py`)
* **구조**: 수백만 장(ImageNet 1000개 클래스)으로 이미 사전 학습된 **`ResNet18`** 모델을 다운받고, 모델의 가장 끝단 1줄 머리 부품인 Linear Classifier 층만 2개 클래스(Ants, Bees)로 교체
* **훈련 환경**: CPU / Batch Size: 4 / Epoch: 10 / Optimizer: SGD(0.001, momentum=0.9)
* **결과 (Validation Accuracy)**: 단 1 Epoch 만에 92% 이상의 정확도를 달성하며, 최종적으로 **95% 이상의 파괴적 성능 향상**을 보여주었습니다.
* **시사점**: 현장의 90% 이상은 스크래치 코드가 아니라, 이처럼 강력한 ImageNet 사전 학습 가중치와 구조를 빌려와(Transfer Learning) 내 문제에 맞게 미세 조정(Fine-Tuning)하여 사용합니다.

### 2-3. 번외: 머리를 뜯기 전 순정 거인은? (`src/10_2_test_original_resnet.py`)
* **평가 목적**: 마지막 Classifier(머리)를 개미/벌 전용으로 교체하지 않고, 1000개의 영단어를 내뱉는 "순정 ResNet18" 그 자체를 돌려봅니다.
* **결과**: 놀랍게도 153장의 이미지 중 114장을 맞춰 **74.5%**의 확률로 정확히 "ant" 나 "bee"를 외칩니다.
* **시사점**: 우리는 전이 학습을 통해 거인의 눈(Conv 층)이 세상을 해석하는 방식을 배우는 것이 아니라, 이미 완벽한 그 해석을 "2개짜리 보기"로 깔때기만 꽂아 정리해주는 최적화 작업을 한 것임을 수치로 입증했습니다.

---

## 3. 자연어 처리: 영화 리뷰 감성 머신 (`src/11_rnn_sentiment_analysis.py`)
* **목적**: CNN의 공간(이미지) 처리를 벗어나, "시간의 흐름(순서)"을 파악해야 하는 시계열 데이터(자연어)에 대한 RNN의 성능을 검증합니다.
* **구조**: `Embedding(차원:8)` -> `RNN(은닉 상태 차원:16)` -> `Linear Classifier` -> `Sigmoid`
* **훈련 데이터**: 겨우 4단어짜리 초미니 영화 리뷰 예제 5문장
* **결과**: 100 Epoch 훈련 결과, **99.8% 이상의 확신(확률)**으로 주어진 문장이 긍정인지 부정인지 완벽하게 판별해 냅니다.
* **시사점**: 단순하게 단어 묶음을 넘기는 Sliding Window 한계를 벗어나, 은닉 상태(Hidden State)를 피드백(Feedback)으로 계속 넘겨받으며 **문맥의 분위기를 눈덩이처럼 누적(Recurrent)**할 수 있다는 강력한 증거입니다.
