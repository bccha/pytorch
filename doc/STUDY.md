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

# 무작위 값 또는 상수 텐서 생성
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# 속성 확인 (메모리 위치 포함)
print(f"Shape: {x_data.shape}, Type: {x_data.dtype}, Device: {x_data.device}")
```

### 2.2 인덱싱과 슬라이싱 (Indexing & Slicing)
```python
tensor = torch.ones(4, 4)
print(f"첫 행: {tensor[0]}")
print(f"첫 열: {tensor[:, 0]}")
print(f"마지막 열: {tensor[..., -1]}")

# 특정 값 변경(in-place)
tensor[:, 1] = 0
```

### 2.3 형태 변환: `view` vs `reshape`
데이터의 차원을 자유자재로 다루는 것은 신경망 입력 계층 설계의 핵심입니다.
* **`view()`**: 메모리 복사 없이 기존 데이터를 오버랩하여 보여줍니다. 가장 빠르지만, 메모리가 연속적이지 않으면(non-contiguous) 에러를 발생시킵니다.
* **`reshape()`**: 가능하면 `view()`처럼 작동하고, 불가능하면 메모리를 안전하게 복사(Copy)하여 결과를 반환하는 유연한 방식을 취합니다. 텐서 조작 시 `reshape()`의 사용이 일반적으로 더 안전합니다.

```python
x = torch.randn(4, 4)       # 4x4 행렬 (총 16개 요소)
y = x.view(16)              # 1차원 벡터로 변환
z = x.reshape(-1, 8)        # -1은 다른 모양을 보고 알아서 추론 (여기선 2x8이 됨)
```

### 2.4 텐서 합치기 (Concatenate & Stack)
* **`torch.cat`**: 기존 차원을 따라 텐서들을 이어 붙일 때 사용합니다.
  ```python
  # 1차원 방향(가로)으로 이어 붙이기
  t1 = torch.cat([tensor, tensor, tensor], dim=1)
  ```
* **`torch.stack`**: 아예 새로운 차원을 만들면서 텐서들을 쌓아 올릴 때 사용합니다.
  ```python
  # 4x4 텐서 3개를 하나로 합쳐서 (3, 4, 4) 텐서로 만듦
  t2 = torch.stack([tensor, tensor, tensor])
  ```

### 2.5 스칼라(Scalar) 값 뽑아내기
1개의 요소만 들어있는 텐서에서 순수한 파이썬 숫자(int, float 등) 값만 꺼낼 때는 `.item()`을 사용합니다.
```python
agg = tensor.sum()       # 모든 원소의 합계 텐서 (크기 1)
agg_item = agg.item()    # 파이썬 숫자로 변환
print(agg_item, type(agg_item))
```

---

## 3장. 오토그라드(Autograd): 자동 미분 메커니즘
> **관련 예제**: `src/02_autograd.py`

모든 신경망은 미분(오차 역전파)을 통해 학습됩니다. PyTorch의 가장 큰 매력 중 하나는 수식 없이도 역전파 연산 그래프를 실시간(Dynamic Computation Graph)으로 그려주는 `autograd` 엔진입니다.

로직에 참여하는 텐서에 `requires_grad=True`를 부여한 뒤 최종 결과값에서 `.backward()`를 호출하면, 텐서가 흘러온 모든 경로를 역추적하여 `.grad` 속성에 기울기(Gradient)를 자동 누적합니다.

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean() # 연산 그래프 구성

out.backward() # 역전파(Backpropagation) 엔진 가동
print(x.grad)  # d(out)/dx의 결과가 출력됨
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
        super(SimpleNet, self).__init__()
        # 합성곱 계층 (채널 1개 -> 6개, 커널 5x5)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 완전 연결 계층 (선형 연산 y = Wx + b)
        self.fc1 = nn.Linear(6 * 10 * 10, 10)

    def forward(self, x):
        # 순전파(Forward pass) 설계: 데이터가 흘러가는 길
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = torch.flatten(x, 1) # 공간 차원을 평탄화
        x = self.fc1(x)
        return x

net = SimpleNet()
input_tensor = torch.randn(1, 1, 24, 24) # Batch, Channel, Height, Width
out = net(input_tensor) # 모델 예측
```

> **📌 실전 설계 팁: Conv와 Linear 사이의 차원 맞추기**  
> 공간 구조를 가진 합성곱 레이어(Conv2d)의 출력을 1차원 선형 레이어(Linear)에 넣으려면 차원 크기를 맞춰야 합니다. 계산이 복잡하다면 임의의 데이터를 흘려보내 만나는 차원 오류 메시지를 참고하거나, 최신 `nn.LazyLinear(10)` 모듈을 사용하여 동적 할당을 위임할 수 있습니다.

---

## 5장. 손실 함수와 옵티마이저
분류(Classification) 문제에서 마주치는 가장 큰 오해 중 하나는 Softmax 연산의 호출 위치입니다.

### 5.1 손실 계산: Softmax의 함정
* **`Softmax`**: 원시 출력(Logits)을 1.0(100%) 합계의 확률로 변환합니다.
* **`Argmax`**: 가장 높은 확률을 가진 클래스의 인덱스 번호를 반환하여 최종 예측값을 제공합니다.

하지만 PyTorch의 표준 분류 손실 함수인 **`nn.CrossEntropyLoss`는 내부적으로 Softmax 연산과 극대화(Log) 연산을 모두 포함**하고 있습니다. 따라서 모델을 시각적으로 확인할 때는 밖에서 `softmax`를 쓰더라도, 실제 학습 시 Loss를 계산할 때는 절대 신경망의 맨 마지막 출력층에 Softmax를 씌우지 않고, 날것의 숫자값(Logits)을 곧바로 Loss 함수 블록에 골인시켜야 오작동이 일어니지 않습니다.

```python
loss_fn = nn.CrossEntropyLoss()

# 1. 모델에서 나온 날것의 출력 (Softmax 안 거침!)
out = net(input_tensor) 

# 2. 정답 라벨 (예: 10개 클래스 중 정답은 8번)
target = torch.tensor([8]) 

# 3. 모델 출력(out)을 그대로 넣으면, 내부에서 알아서 확률 변환 및 Loss 계산
loss = loss_fn(out, target)
```

### 5.2 장비 할당과 옵티마이저
가중치를 업데이트할 엔진을 장착할 차례입니다. 모델 내부의 모든 "학습 가능한 파라미터(`net.parameters()`)"를 옵티마이저에 연결합니다.
```python
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)

# Adam 최적화 알고리즘 장착
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

> **[Tip] 가중치 목록 확인하기**  
> `nn.Module`을 상속받았기 때문에 내부에 레이어를 선언하면 알아서 학습 파라미터로 등록됩니다. 
> `net.named_parameters()`를 순회하면 `conv1.weight`, `fc1.bias` 등의 자동 부여된 이름과 형태를 확인할 수 있습니다.

---

## 6장. 데이터 다루기: Dataset과 DataLoader
> **관련 예제**: `src/05_dataset_dataloader.py`

실제 딥러닝에서 데이터는 보통 수만 개 단위이므로 모두 한 번에 메모리에 올릴 수 없습니다. PyTorch는 이 데이터 수급을 위한 강력한 파이프라인 패턴을 권장합니다.

### 6.1 `Dataset`: 데이터 샘플 자판기
모든 원본 데이터는 `Dataset` 클래스로 감싸집니다. 데이터의 총 개수 파악(`__len__`)과 특정 인덱스의 데이터 샘플을 한 개만 끄집어내어 텐서로 가공하는(`__getitem__`) 기능만을 담당합니다.

### 6.2 `DataLoader`: 묶음 배송 트럭
`Dataset`이 데이터를 하나씩만 줄 수 있다면, `DataLoader`는 이를 받아 우리가 지정한 **배치 크기(Batch Size)**만큼 차곡차곡 모아(Stacking) 모델에 한 뭉텅이씩 전달해 줍니다. 
```python
train_loader = DataLoader(
    dataset=my_dataset, 
    batch_size=32,      # 한 번에 32개씩 묶어주세요
    shuffle=True,       # 뽑기 전에 데이터 순서를 뒤죽박죽 섞어주세요 (편향 암기 방지)
    drop_last=True      # 데이터 개수가 32개로 안 떨어지면 Shape 에러 방지를 위해 자투리 버림
)
```

### 6.3 Train Set과 Test Set (데이터 분할의 정석)
모델의 과적합(Overfitting, 기출문제 단순 암기 현상)을 통제하고 처음 보는 데이터에 대한 일반화(Generalization) 능력을 확인하려면, 정답을 가리고 실전 평가만을 담당할 Test Set을 반드시 약 20% 분리해 두어야 합니다.

* **방법 A. 내장 데이터셋 모듈 활용 (예: MNIST)**: `torchvision.datasets.MNIST(train=True/False)` 옵션만으로 이미 제작자에 의해 분할 배포된 데이터셋을 손쉽게 가져올 수 있습니다.
* **방법 B. 커스텀 데이터 세트 분할**: 임의의 커스텀 데이터를 8:2 로 분리할 때는 무작위로 섞어서 쪼개주는 `random_split`을 이용합니다.
  ```python
  from torch.utils.data import random_split
  
  # 1000개 데이터셋을 800번, 200개로 쪼개기
  train_dataset, test_dataset = random_split(my_dataset, [800, 200])
  ```

---

## 7장. 딥러닝 훈련 루프 완성하기
> **관련 예제**: `src/04_training_loop.py`

학습 루프 코드를 짜다 보면 마주치는 가장 기본적인 용어를 비유해 봅시다.
* **데이터셋(Dataset)** = 우리가 이번 학기에 풀어야 할 전체 "수학 문제집 1권" 
* **배치 크기(Batch Size)** = 하루에 풀 "문제 개수" (모델 메모리 한계 방지)
* **에폭(Epoch)** = 수학 문제집 1권을 처음부터 끝까지 "N번째 다시 풀기" (1 Epoch = 문제집 1회독 완독)

### 7.1 표준 학습 5단계 스텝
PyTorch 개발자들이 가장 많이 작성하게 되는 표준 학습 루프입니다. 

```python
for epoch in range(10): 
    # [1단계] 입력 데이터와 정답 준비
    inputs, labels = get_data() 

    # [2단계] 미분값(Gradient) 초기화
    # 저번 학습 스텝에서 누적된 기울기를 0으로 비워줍니다. (안 하면 기울기가 계속 누적됨)
    optimizer.zero_grad() 

    # [3단계] 순전파 (Forward) 및 Loss 계산
    outputs = net(inputs)
    loss = loss_fn(outputs, labels)

    # [4단계] 역전파 (Backward)
    # Loss를 바탕으로 모델 각 가중치들이 얼마나 틀렸는지(기울기) 전부 계산합니다.
    loss.backward()

    # [5단계] 가중치 업데이트 (Step)
    # 4단계에서 구해둔 기울기(Grad)를 보고, Optimizer가 한 걸음 이동해 가중치를 바꿉니다.
    optimizer.step()
```
> **[주의사항]** `optimizer.zero_grad()`를 깜빡하면 이전 배치의 그래디언트가 더해져 방향을 잃습니다. 모든 루프 사이클 시작 시 무조건 기울기를 0으로 리셋하세요.

### 7.2 무작위성(Randomness) 제어와 시드(Seed)
똑같은 코드로 학습을 재차 돌릴 때마다 정확도가 수시로 출렁인다면, 이는 가중치(Weight) 난수 초기화와 데이터 로더의 셔플(Shuffle)에 숨어있는 무작위 요소 때문입니다. 알고리즘 실험의 정확한 재현성(Reproducibility)을 입증하려면 스크립트 최상단에 고정된 시드를 주입하세요.
```python
import torch
import random
import numpy as np

# 파이토치 난수 생성기 고정
torch.manual_seed(42)  
```

---

## 8장. 실전 프로젝트: MNIST 손글씨 분류
> **관련 예제**: `src/06_mnist_fc.py`, `src/07_mnist_cnn.py`

확보한 개념을 병합하여 28x28 픽셀의 공간 정보(0~9 숫자 손글씨 이미지 총 7만 장)를 분류해 냅니다. 공간의 차원을 파괴하는 선형 구조(FC, 97.48%) 대비 시각적 특성을 추출하는 합성곱 구조(CNN, 99.11%)가 이미지 분석에서 얼마나 절대적인 정확도 우위를 가지는지 결과로 증명하는 챕터입니다.

### 8.1 🚨 실력을 검증하라: 평가 모드 스위칭 기술
가중치 업데이트를 동반하는 학습 과정이 1 Epoch 종료되고 Test 데이터를 만나 평가를 진행할 때, 시스템의 두 가지 주요 컴포넌트를 비활성화하는 조작이 훈련 결과에 치명적입니다.

1. **`model.eval()`: 모델의 상태 변경 전환**
   * Dropout 이나 Batch Normalization 계층 등 오직 모델의 '학습' 단계에서만 발동되는 노이즈/패널티 레이어를 정지시킵니다.
   * *[Tip]* 에폭 단위 반복문(`for epoch...`) 최상단에 `model.train()`을 배치하세요. 이 함수 호출 덕분에 모델이 Test 평가를 마치고 위로 루프를 다시 돌 때 자동으로 평가 환경 설정이 해제되고 훈련 모드로 스위치가 안전하게 원상 복구됩니다.
2. **`with torch.no_grad():`: 자동 미분 배관 차단**
   * 평가는 그저 정답을 맞추는 순전파(Forward Pass) 연산만 필요합니다. 자동 미분 배관이 켜져 있으면 메모리 VRAM 버퍼 폭주가 일어납니다.
   * 이에 파이썬의 `with` 구문 스코프 내에서 역전파용 자동 미분 엔진을 강제 블로킹합니다. 들여쓰기 된 블록 내부 안에서만 모델 배관의 전원이 꺼지고, 이 들여쓰기 블록을 빠져나오면 시스템은 다음 루프를 위해 자동으로 파이프라인 수집 상태로 복귀합니다.
