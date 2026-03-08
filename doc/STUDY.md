# Learning PyTorch: 실전 딥러닝 구축 가이드

**이 가이드는 머신러닝 초심자를 위한 PyTorch 기반 실무 입문서 형식으로 작성되었습니다.**
각 장(Chapter)은 프로젝트 내의 `src/` 디렉터리 하위 예제 소스 코드 순서(`01` ~ `07`)와 완벽하게 맵핑되어 있으므로, 코드를 직접 실행해 보며 읽는 것을 권장합니다.

---

## 1장. 시작하며: 환경 설정
PyTorch 생태계에 뛰어들기 전, 격리된 가상환경에 엔진을 설치하는 과정입니다.

* **가상환경 활용**: `venv` 모듈을 사용해 프로젝트 의존성을 격리합니다. (`. venv/Scripts/Activate.ps1`)
* **설치**: `pip install torch torchvision torchaudio`

---

## 2장. 텐서(Tensor) 조작의 기술
> **관련 예제**: `src/01_tensor_basics.py`

딥러닝의 기본적인 데이터 구조인 텐서(Tensor)는 NumPy의 다차원 배열(`ndarray`)과 거의 동일한 인터페이스를 갖추고 있으나, **GPU 상에서 하드웨어 가속**이 가능하다는 치명적인 장점을 제공합니다.

### 2.1 텐서 생성과 속성
일반 리스트나 NumPy 배열로부터 텐서를 생성할 수 있습니다. 텐서를 디버깅할 때는 `shape`, `dtype`, `device` 세 가지 속성을 확인하는 것이 필수적입니다.
```python
import torch
import numpy as np

# 리스트와 NumPy 배열로부터 생성
x_data = torch.tensor([[1, 2], [3, 4]])
x_np = torch.from_numpy(np.array([[1, 2], [3, 4]]))

# 속성 확인 (메모리 위치 포함)
print(f"Shape: {x_data.shape}, Type: {x_data.dtype}, Device: {x_data.device}")
```

### 2.2 형태 변환: `view` vs `reshape`
데이터의 차원을 자유자재로 다루는 것은 신경망 입력 계층 설계의 핵심입니다.
* **`view()`**: 메모리 복사 없이 기존 데이터를 오버랩하여 보여줍니다. 가장 빠르지만, 메모리가 연속적이지 않으면(non-contiguous) 에러를 발생시킵니다.
* **`reshape()`**: 가능하면 `view()`처럼 작동하고, 불가능하면 메모리를 안전하게 복사(Copy)하여 결과를 반환하는 유연한 방식을 취합니다. 텐서 조작 시 `reshape()`의 사용이 일반적으로 더 안전합니다.

---

## 3장. 오토그라드(Autograd): 자동 미분 메커니즘
> **관련 예제**: `src/02_autograd.py`

모든 신경망은 미분(오차 역전파)을 통해 학습됩니다. PyTorch의 가장 큰 매력 중 하나는 수식 없이도 역전파 연산 그래프를 실시간(Dynamic Computation Graph)으로 그려주는 `autograd` 엔진입니다.

로직에 참여하는 텐서에 `requires_grad=True`를 부여한 뒤 최종 결과값에서 `.backward()`를 호출하면, 텐서가 흘러온 모든 경로를 역추적하여 `.grad` 속성에 기울기(Gradient)를 자동 누적합니다.

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = (y * y).mean() # 연산 그래프 구성

z.backward()       # 역전파(Backpropagation) 엔진 가동
print(x.grad)      # x가 z에 미친 영향력(기울기) 출력
```

---

## 4장. 신경망 모델의 뼈대 세우기
> **관련 예제**: `src/03_neural_network.py`

PyTorch에서 모든 신경망은 `torch.nn.Module`을 상속받아 정의합니다. 객체 지향적인 접근 방식을 통해 레이어(Layer)들을 레고 블록처럼 조립할 수 있습니다.

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 합성곱 계층 (채널 1개 -> 6개, 커널 5x5)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 완전 연결 계층
        self.fc1 = nn.Linear(6 * 10 * 10, 10)

    def forward(self, x):
        # 순전파 설계: 데이터가 흘러가는 길
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = torch.flatten(x, 1) # Dense 계층 진입 전 평탄화
        x = self.fc1(x)
        return x
```

> **📌 실전 설계 팁: Conv와 Linear 사이의 차원 맞추기**  
> 공간 구조를 가진 합성곱 레이어(Conv2d)의 출력을 1차원 선형 레이어(Linear)에 넣으려면 차원 크기를 맞춰야 합니다. 계산이 복잡하다면 임의의 데이터를 흘려보내 만나는 차원 오류 메시지를 참고하거나, 최신 `nn.LazyLinear(10)` 모듈을 사용하여 동적 할당을 위임할 수 있습니다.

---

## 5장. 손실 함수와 옵티마이저
분류(Classification) 문제에서 마주치는 가장 큰 오해 중 하나는 Softmax 연산의 호출 위치입니다.

### 5.1 손실 계산: Softmax의 함정
* **`Softmax`**: 원시 출력(Logits)을 1.0(100%) 합계의 확률로 변환합니다.
* **`Argmax`**: 가장 높은 확률을 가진 클래스의 인덱스 번호를 반환하여 최종 예측값을 제공합니다.

하지만 PyTorch의 표준 분류 손실 함수인 **`nn.CrossEntropyLoss`는 내부적으로 Softmax 연산과 극대화(Log) 연산을 모두 포함**하고 있습니다. 따라서 신경망의 맨 마지막 출력층에 Softmax를 강제로 씌우지 않고, 날것의 숫자값(Logits)을 곧바로 Loss 함수 블록에 골인시켜야 연산 효율과 수치 안정성이 보장됩니다.

### 5.2 장비 할당과 옵티마이저
가중치를 업데이트할 엔진을 장착할 차례입니다. 모델 내부의 모든 "학습 가능한 파라미터(`net.parameters()`)"를 옵티마이저에 연결합니다.
```python
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)

# Adam 최적화 알고리즘 장착
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 6장. 데이터 다루기: Dataset과 DataLoader
> **관련 예제**: `src/05_dataset_dataloader.py`

딥러닝의 거대한 데이터를 처리하기 위해 PyTorch는 데이터 수급을 위한 강력한 파이프라인 패턴을 권장합니다.

### 6.1 `Dataset`: 데이터 샘플 자판기
모든 원본 데이터는 `Dataset` 클래스로 감싸집니다. 데이터의 총 개수 파악(`__len__`)과 특정 인덱스의 데이터 샘플을 한 개만 끄집어내어 텐서로 가공하는(`__getitem__`) 기능만을 담당합니다.

### 6.2 `DataLoader`: 묶음 배송 트럭
`DataLoader`는 `Dataset`에서 데이터를 계속 꺼내 지정된 양치기(Batch)로 압축 및 셔플링을 수행한 뒤 GPU 파이프라인으로 전송하는 역할을 합니다.
```python
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=64, # 한 번에 64장 단위로 전송
    shuffle=True   # 데이터 순서 섞기 (편향 암기 방지용)
)
```

### 6.3 Train Set과 Test Set (데이터 분할의 정석)
모델의 과적합(Overfitting, 암기 현상)을 통제하고 일반화된 성능 평가를 위해서는, 정답을 가리고 실전 평가만을 담당할 Test Set 공간을 분리해야만 합니다.

* **내장 데이터셋 활용 (예: MNIST)**: `datasets.MNIST(train=True/False)` 옵션만으로 이미 제작자에 의해 분할 배포된 데이터셋을 손쉽게 가져올 수 있습니다.
* **커스텀 데이터 세트 분할**: 임의의 데이터를 8:2 로 분리할 때는 `random_split`을 이용합니다.
  ```python
  from torch.utils.data import random_split
  train_dataset, test_dataset = random_split(my_dataset, [800, 200])
  ```

---

## 7장. 딥러닝 훈련 루프 완성하기
> **관련 예제**: `src/04_training_loop.py`

PyTorch 개발자들이 가장 많이 작성하게 되는 5단계 보일러플레이트(표준 학습 루프)입니다. 

### 7.1 에폭(Epoch)과 배치(Batch)
* **에폭(Epoch)**: 훈련용 문제장(전체 데이터셋)을 맨 앞장부터 뒷장까지 1회독 완료 상태.
* **배치 크기(Batch Size)**: 컴퓨터 메모리 터짐(OOM)을 방지하기 위해 한 번에 연산 노드에 넘기는 이미지의 장수. `batch_size`를 키울수록 입력 텐서(행렬)의 첫 번째 차원 행렬이 비대해지며 높은 VRAM을 요구합니다.

### 7.2 표준 학습 5단계 스텝
1. **기울기 초기화 (`optimizer.zero_grad()`)**: 반복될 때 이전 배치의 미분 찌꺼기가 합산되는 현상을 차단합니다.
2. **순전파 (`model(inputs)`)**: 신경망 필터를 흘려보내 예측값을 얻어냅니다.
3. **손실 계산 (`loss_fn()`)**: 정답과 비교해 모델의 오차량을 측정합니다.
4. **역전파 (`loss.backward()`)**: 오차를 이용해 신경망을 구성하는 계층의 모든 로컬 노드들의 기울기를 산출합니다.
5. **가중치 업데이트 (`optimizer.step()`)**: 기울기가 가리키는 방향으로 지정된 학습률(Learning Rate)만큼 파라미터를 조율합니다.

### 7.3 무작위성(Randomness) 제어와 시드(Seed)
똑같은 코드로 학습을 재차 돌릴 때마다 정확도가 수시로 출렁인다면, 이는 가중치 초기화와 데이터 로더의 셔플(Shuffle)에 숨어있는 요소 때문입니다. 알고리즘 실험의 정확한 재현성(Reproducibility)을 입증하려면 스크립트 최상단에 고정된 시드를 주입하세요.
```python
# 가장 많이 쓰이는 관용적인 시드 번호 주입
torch.manual_seed(42)  
```

---

## 8장. 실전 프로젝트: MNIST 손글씨 분류
> **관련 예제**: `src/06_mnist_fc.py`, `src/07_mnist_cnn.py`

확보한 개념을 병합하여 28x28 픽셀의 공간 정보(0~9 숫자 데이터)를 분류해 냅니다. 공간의 차원을 파괴하는 선형 구조(FC) 대비 시각적 특성을 추출하는 합성곱 구조(CNN)가 이미지 분석에서 얼마나 절대적인 위치를 가지는지 파악하는 것이 이 장의 핵심입니다.

### 8.1 🚨 실력을 검증하라: 평가 모드 스위칭 기술
가중치 업데이트 과정이 종료되고 Test 데이터에 직면했을 때, 시스템의 두 가지 주요 컴포넌트를 비활성화하는 조작이 필요합니다.

1. **`model.eval()`: 모델의 상태 변경 전환**
   * Dropout 이나 Batch Normalization 계층 등 오직 모델의 '학습' 단계에서만 발동되는 노이즈/패널티 레이어를 정지시킵니다.
   * *[Tip]* 에폭 단위 반복문(`for epoch...`) 최상단에 `model.train()`을 배치하세요. 이 함수 호출 덕분에 모델이 Test 평가를 마치고 위로 루프를 다시 돌 때 자동으로 평가 환경 설정이 해제되고 훈련 모드로 스위치가 원상 복구됩니다.
2. **`with torch.no_grad():`: 자동 미분 배관 차단**
   * 평가는 그저 정답을 맞추는 순전파(Forward Pass) 연산만 필요합니다. 
   * 파이썬의 `with` 구문 스코프 내에서 역전파용 자동 미분 엔진을 강제 블로킹하여 상당량의 연산 오버헤드와 VRAM 소모를 제거합니다. 이 들여쓰기 블록을 빠져나오면 시스템은 자동 파이프라인 수집 상태로 복귀합니다.
