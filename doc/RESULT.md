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
