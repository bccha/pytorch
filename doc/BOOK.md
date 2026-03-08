# 실전 PyTorch 프로그래밍: 기초부터 CNN 영상 인식까지

**본 문서는 초보자를 위한 PyTorch 기반 실무 입문서(O'Reilly 스타일)입니다.**
모든 내용은 `src/` 디렉터리의 실습 파일 1개당 1개의 챕터로 완벽하게 대응하도록 구성되었습니다. 코드를 보면서 차례대로 따라오세요.

---

## 1장. 시작하며: 환경 설정 및 텐서(Tensor) 기초
**[ 실습 파일: `src/01_tensor_basics.py` ]**

### 1.1 설치 및 환경 확인
- **가상환경**: 격리된 개발 환경을 위해 `venv`를 사용합니다. (`Activate.ps1` 또는 `deactivate`)
- **설치 명령어**: `pip install torch torchvision torchaudio`

### 1.2 텐서(Tensor)란?
PyTorch의 핵심 데이터 구조는 텐서(Tensor)입니다. NumPy의 `ndarray`와 매우 유사하지만, **GPU를 활용한 연산 가속**이 가능하다는 점이 가장 큰 차이입니다.

### 1.3 텐서의 생성과 주요 속성
```python
import torch
import numpy as np

# 1. 일반 리스트와 NumPy 배열로부터 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
x_np = torch.from_numpy(np.array(data))

# 2. 무작위 값 또는 상수 텐서 생성
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# 3. 주요 속성 (Attributes) 확인
tensor = torch.rand(3, 4)
print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device: {tensor.device}") # cpu 또는 cuda:0 등
```

### 1.4 텐서의 기본 조작 (Operations)
NumPy와 유사하게 인덱싱, 슬라이싱, 형태 변환, 결합 등의 풍부한 연산을 지원합니다.

* **인덱싱과 슬라이싱 (Indexing & Slicing)**
  ```python
  tensor = torch.ones(4, 4)
  print(f"첫 행: {tensor[0]}")
  print(f"첫 열: {tensor[:, 0]}")
  print(f"마지막 열: {tensor[..., -1]}")
  
  # 특정 값 변경(in-place)
  tensor[:, 1] = 0
  ```

* **형태 변환 (View / Reshape)**
  텐서의 데이터 개수는 유지하되 모양(차원)만 바꿀 때 `.view()` 또는 `.reshape()`를 사용합니다.
  - **`view()`**: 메모리를 복사하지 않고 기존 데이터를 그대로 공유합니다. 데이터가 메모리상 연속적(contiguous)일 때만 작동하며, 아니면 에러를 발생시켜 안전한 동작을 보장합니다.
  - **`reshape()`**: 가능하면 `view()`처럼 일하지만, 불가능할 경우 새로운 텐서로 알아서 복사(Copy) 결과를 내놓는 유연하고 안전한 방식입니다.
  ```python
  x = torch.randn(4, 4)       # 4x4 행렬 (총 16개 요소)
  y = x.view(16)              # 1차원 벡터로 변환
  z = x.reshape(-1, 8)        # -1은 다른 모양을 보고 알아서 추론 (여기선 2x8이 됨)
  ```

* **텐서 합치기 (Concatenate & Stack)**
  - `torch.cat`: 기존 차원을 따라 텐서들을 이어 붙일 때 사용합니다.
    ```python
    # 1차원 방향(가로)으로 3개 이어 붙이기
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    ```
  - `torch.stack`: 아예 새로운 차원을 만들면서 텐서들을 위로 쌓아 올릴 때 사용합니다.
    ```python
    # 4x4 텐서 3개를 하나로 합쳐서 새로 (3, 4, 4) 텐서로 확장
    t2 = torch.stack([tensor, tensor, tensor])
    ```

* **스칼라(Scalar) 값 뽑아내기**
  1개의 요소만 들어있는 텐서에서 순수한 파이썬 숫자(int, float 등) 값만 꺼낼 때는 `.item()`을 사용합니다.
  ```python
  agg = tensor.sum()       # 모든 원소의 합계 텐서 (크기 1)
  agg_item = agg.item()    # 파이썬 숫자로 변환
  print(agg_item, type(agg_item))
  ```

---

## 2장. 오토그라드(Autograd): 자동 미분 메커니즘
**[ 실습 파일: `src/02_autograd.py` ]**

PyTorch의 모든 신경망 연산의 핵심은 `autograd` 패키지입니다. 이 엔진은 텐서가 수행하는 모든 연산에 대해 자동 미분을 실시간으로 제공합니다.

- `requires_grad=True` 로 설정된 텐서는 자신이 거치는 모든 연산 파이프라인을 내부적으로 추적합니다.
- 계산이 완료된 후 `.backward()` 를 호출하면 모든 그래디언트(Gradient, 기울기)가 자동으로 계산되어 텐서의 `.grad` 속성에 누적됩니다.

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward() # 역전파(Backpropagation) 수행 엔진 가동
print(x.grad)  # d(out)/dx의 추적 결과(기울기)가 출력됨!
```

---

## 3장. 신경망 모델 (torch.nn) 및 손실 함수 설계
**[ 실습 파일: `src/03_neural_network.py` ]**

### 3.1 신경망의 뼈대 세우기
PyTorch에서 신경망은 객체 지향적으로 구성된 `torch.nn` 패키지를 사용하여 생성합니다. `nn.Module`을 상속받아 모델 구조(Layer)들을 레고 블록처럼 정의합니다.

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 1개의 입력 채널, 6개의 출력 채널, 5x5 합성곱 층
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 선형 연산 (y = Wx + b)
        self.fc1 = nn.Linear(6 * 10 * 10, 10)

    def forward(self, x):
        # 순전파(Forward pass) 정의: 데이터가 흘러가는 길
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = torch.flatten(x, 1) # 빈틈없이 공간 차원을 평탄화
        x = self.fc1(x)         # 최종 선형 신경망 통과
        return x

net = SimpleNet()
input_tensor = torch.randn(1, 1, 24, 24) # Batch, Channel, Height, Width 지정
out = net(input_tensor) # 모델 예측
```

> **💡 [꿀팁] Conv-Linear 연결부 입력 차원 뚝딱 맞추기**  
> 위 코드의 `nn.Linear(6 * 10 * 10, 10)`처럼 공간(2D) 정보 위주의 합성곱(Conv)층을 평탄화(Flatten)한 뒤 완전연결층(Linear)으로 넘길 때는 입력 차원을 미리 사람이 직접 계산해서 적어줘야 합니다. 복잡하게 계산하기 싫다면?
> 1. **에러로 확인하기**: `nn.Linear(1, 10)` 처럼 엉뚱한 숫자를 넣고 데이터를 흘려보면 내뱉는 에러 메시지(차원 불일치)를 참조하여 정답을 알아냅니다.
> 2. **`nn.AdaptiveAvgPool2d` 사용**: 입력과 무관하게 출력 크기를 고정하는 레이어로 통제합니다.
> 3. **`nn.LazyLinear` 사용 (최신 권장)**: `nn.LazyLinear(10)` 처럼 들어올 노드 수를 비워두면, 첫 데이터가 흘러갈 때 PyTorch가 알아서 동적으로 공간 차원 크기를 세팅해 줍니다!

### 3.2 손실 함수(Loss Function)와 Softmax 의 함정
분류(Classification) 모델을 만들 때 가장 헷갈리는 것이 모델의 최종 출력값과 손실 함수의 연결입니다.

- **`Softmax`**: 원시 숫자(Logits)들을 모두 합쳐 `1.0(100%)`이 되도록 **확률 값으로 변환**합니다.
- **`Argmax`**: 모델 출력값 중에서 가장 높은 확률의 **인덱스(순서) 번호**, 즉 실제 정답의 위치를 뽑습니다.

> **💡 그럼 Loss Function 에는 정답을 뭘 구해서 넣어야 할까?**  
> **절대로 모델 출력 끝에 Softmax나 Argmax를 달고 그 결과물을 넣지 마세요!**  
> PyTorch에서 가장 많이 쓰는 `nn.CrossEntropyLoss`는 내부 설계 구조상 이미 그 안에 `Softmax` 연산과 `Log(로그)` 연산 엔진이 완벽하게 짬뽕되어 내장되어 있습니다.

```python
loss_fn = nn.CrossEntropyLoss()

# 1. 모델에서 나온 날것의 숫자 (Logits 출력, Softmax 안 거침!)
out = net(input_tensor) 

# 2. 정답 라벨
target = torch.tensor([8]) 

# 3. 모델 출력(out)을 그대로 넣으면, 내부에서 알아서 Softmax 확률 변환 -> Loss 도출 처리
loss = loss_fn(out, target)
print(loss)
```

우리가 콘솔 창에 결과를 예쁘게 출력(`print`)해서 눈으로 볼 때만 바깥에서 임시로 확률 변환(Softmax)을 하고, **실제 딥러닝 학습 로직 파이프라인 안에서는 날것의 숫자(Logits)를 그대로 바로 넘겨주어야** 치명적인 에러와 비효율이 발생하지 않습니다.

---

## 4장. 딥러닝 훈련 루프와 옵티마이저 완성
**[ 실습 파일: `src/04_training_loop.py` ]**

모델의 오차(Loss)를 구하고 역전파 미분을 수행했다면, 기울기 방향대로 모델을 똑똑하게 수정(학습)시켜야 합니다. 이 역할을 **옵티마이저(Optimizer)**가 수행합니다. 더불어 딥러닝 개발자라면 매일 숨 쉬듯 쓰게 되는 표준 반복문 루프에 대해 알아봅시다.

### 4.1 딥러닝 필수 용어 해설 (에폭, 배치, 스텝)
학습 루프 코드를 짜면 반드시 튀어나오는 용어를 직관적인 비유로 풀어봅시다.
- **데이터셋(Dataset)** = 우리가 이번 학기에 풀어야 할 전체 "수학 문제집 1권" (예: 문제 총 1,000개)
- **배치(Batch Size)** = 하루에 풀 "문제 개수" (예: 하루 100문제씩 풀기 = Batch Size 100)
- **스텝(Step/Iteration)** = 오답 노트를 쓰고 채점하는 횟수 (1,000문제를 100개씩 묶어 풀면 1바퀴 돌리는데 총 **10번** 채점을 받습니다.)
- **에폭(Epoch)** = 수학 문제집 전체 1권을 처음부터 끝까지 "N번째 다시 풀기" (예: 5 Epoch = 문제집 똑같은걸 5번 회독하기)

> **💡 왜 묶어서(Batch) 풀고, 여러 번(Epoch) 회독 시키나요?**  
> * **Batch Size를 주는 이유**: 100만 장의 이미지를 한 번에 채점(Loss)시키려고 하면 그래픽카드 메모리(VRAM)가 당장 터집니다(`OOM 에러`). 반대로 1픽셀짜리 조각 이미지 1장씩만 보고 채점하면 학습 궤도가 너무 들쭉날쭉 널뛰기합니다. "적당량의 큰 보따리 단위(Batch)"를 욱여넣어야 효율적인 학습 속도와 안정성을 동시에 취할 수 있습니다.
> * **Epoch를 주는 이유**: 사람도 한 번 본 문제를 완벽히 다 못 맞추듯 AI도 기출문제를 거듭 반복 연습해야 미세한 가중치 규칙을 찾아 암기합니다.

### 4.2 옵티마이저 생성 및 파라미터 확인
`torch.optim` 패키지에서 아담(Adam) 등 옵티마이저를 끄집어낼 때 "어떤 가중치를 업데이트할 것인지" 지정합니다.

```python
import torch.optim as optim
# 모델 내 파라미터를 Adam 에게 학습률 0.001로 튜닝해달라 통제권 부여
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

> **💡 `net.named_parameters()` 의 마법**  
> `nn.Module`을 상속받았기에 레이어를 짰기만 해도 스스로 학습 파라미터가 등록됩니다. 모델 안에 어떤 변수가 무슨 이름표를 달고 있는지 궁금하다면 아래처럼 순회하여 내부를 발가벗겨 볼 수 있습니다!
> ```python
> for name, param in net.named_parameters():
>     print(f"이름: {name} | 형태: {param.shape}") # 예: conv1.weight, fc1.bias 
> ```

### 4.3 PyTorch 표준 5단계 학습 루프 풀코드
```python
# 에폭: 수능 대비 문제집을 10번 완독하겠다!
for epoch in range(10): 
    # [1단계] 입력 데이터와 정답 보따리 풀기
    inputs, labels = get_data() 

    # [2단계] 미분값(Gradient) 초기화
    optimizer.zero_grad() 

    # [3단계] 순전파 (Forward) 및 Loss(오차 점수) 계산
    outputs = net(inputs)
    loss = loss_fn(outputs, labels)

    # [4단계] 역전파 (Backward) 자동 실행
    loss.backward()

    # [5단계] 가중치 업데이트 (Step)
    optimizer.step()
    
    print(f"Epoch {epoch} - Loss: {loss.item()}")
```
> **🚨 절대 주의!** `optimizer.zero_grad()` 단계가 빠진다면 어제 푼 문제집의 오답(기울기)이 오늘 푼 새 문제집 기울기에 그대로 중첩(+)되어 학습이 산으로 가버립니다.

> **💡 [꿀팁] PyTorch 함수의 파이썬 튜플 규칙 `(4, )`**  
> `torch.randn`이나 형태를 지정하는 옵션에서 괄호 안에 `(4, 5)` 말고 `(4, )` 처럼 쉼표로 끝나는 코드를 볼 수 있습니다.  
> `(4)`라고만 치면 파이썬 인터프리터는 이를 "숫자 4에 연산자 괄호친 것"으로 착각하므로, "안에 요소가 하나뿐인 1차원 벡터 묶음(Tuple)"이라는 사실을 알려주기 위해 뒤에 눈치껏 `,`를 찍어준 것 뿐입니다!

### 4.4 결괏값 난동 지우기: 무작위성(Seed) 고정
코드를 돌릴 때마다 어떨 때는 성능 90%고, 어떨 때는 93%로 튀어 오른다면 **무작위성(Randomness)**이 방해하는 것입니다. 
1. `nn.Conv2d` 같은 레이어는 가중치를 처음에 아무 랜덤 숫자(난수)로 채웁니다. 출발선이 난장판이니 매번 도착 결과도 다릅니다.
2. 데이터를 부르는 로더가 `shuffle=True` 옵션으로 문제 순서를 매번 현란하게 뒤섞습니다.

버그를 잡거나 논문을 재현하려면 코드를 100번 돌려도 소수점 아래까지 정확히 같아야 합니다. 이를 위해 스크립트 맨 상단에 "시드 번호"를 철저히 고정시키면 마법처럼 재현성이 100% 확보됩니다.

```python
import torch
import random
import numpy as np

# 파이토치 자체 난수와 GPU 캐시, 넘파이, 파이썬 기반 난수를 모조리 통제! (42는 관용적 번호)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
```

---

## 5장. 데이터 파이프라인 설계: Dataset과 DataLoader
**[ 실습 파일: `src/05_dataset_dataloader.py` ]**

훈련 루프에서 사용한 `get_data()`를 현업 스케일(메모리에 절대 한 번에 안 올라가는 기가바이트 단위 데이터)로 우아하게 자동화해주는 콤비입니다.

### 5.1 `Dataset`: 데이터 1개씩 꺼내주는 자판기
모든 데이터셋은 PyTorch의 `torch.utils.data.Dataset` 클래스를 부모로 상속받습니다. 이 안에는 반드시 3개의 필수(Magic) 함수가 정의됩니다.
- `__init__()`: 초기화 및 원본 경로 설정
- `__len__()`: 전체 데이터 길이 반환
- `__getitem__(idx)`: 순서(idx)번째에 해당하는 데이터/라벨 딱 1개 세트를 반환

### 5.2 `DataLoader`: 잘라서 섞어주는 배달 트럭
위에서 설계한 자판기(Dataset)에서 데이터를 조금씩 꺼내 모아서 거대한 트럭(Batch)에 싣고 모델 입으로 운반해 줍니다.
```python
my_loader = DataLoader(
    dataset=my_dataset, 
    batch_size=32,      # 한 번에 32개씩 묶어주세요
    shuffle=True,       # 뽑기 전에 문제 순서를 뒤죽박죽 섞어주세요 (외워 풀기 방지)
    drop_last=True      # 마지막에 데이터가 32개로 딱 안 떨어지는 자투리는 버려주세요!
)
```
> **[Tip]** `drop_last=True` 옵션은 매우 요긴합니다. 행렬(차원) 모양에 민감한 신경망의 경우 마지막에 갓길로 빠져나온 18개짜리 초라한 배치 찌꺼기를 모델에 던지면 차원 에러 스파크가 튀기 일쑤이기 때문입니다.

### 5.3 Train Set과 Test Set 분리 (일반화 이론)
모델에게 100장의 기출문제(Train Set)를 주구장창 풀렸다고 칩시다. 채점(평가)을 할 때 똑같은 기출문제를 내버리면, 모델이 "아, 수학의 본질을 깨달았다!"가 아니라 그저 "이 종이에 적힌 잉크의 픽셀 위치를 통째로 사진 찍듯 암기해버렸다!"는 과적합(Overfitting) 사태가 터집니다. 제대로 된 일반화 능력 평가를 위해선 반드시 한 번도 안 본 수능 문제(Test Set)를 완전히 떼어두어야 합니다.

* **A. 내장 데이터셋 (MNIST 등)**
  이미 제작자의 배려로 아래처럼 훈련 데이터와 평가 데이터 파일이 스위치 옵션 하나로 분할 제공됩니다.
  ```python
  train_dataset = datasets.MNIST(root='./data', train=True)
  test_dataset = datasets.MNIST(root='./data', train=False)
  ```
* **B. 일반 커스텀 사진 폴더를 자를 때 (`random_split`)**
  내가 가진 데이터들 1,000개를 8:2 로 토막 내고 싶을 때는 이 함수가 비율을 받아 알아서 무작위로 뒤섞어 멋지게 썰어줍니다.
  ```python
  from torch.utils.data import random_split
  
  my_dataset = MyCustomDataset(data_size=1000)
  train_size = int(len(my_dataset) * 0.8)  # 800개
  test_size = len(my_dataset) - train_size # 200개
  
  train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])
  ```

---

## 6장. 실전 프로젝트 1: MNIST 완전 연결 계층 분류
**[ 실습 파일: `src/06_mnist_fc.py` ]**

0~9까지 그려진 흑백 손글씨 6만 장(Train)을 모델에 64장(Batch)씩 잘게 썰어, 총 5번의 완독(Epoch)을 시키는 훈련 루프 예제입니다. 이 기본 선형 모델(FC 구조)은 이미지 픽셀 28x28(784개)을 일렬 강강술래로 `view`를 통해 쭈욱 펴버린 공간 무시 구조이나, 테스트 데이터 1만 장 대상에서 꽤 놀라운 **97.48%** 의 정확도를 달성했습니다.  

### 🚨 평가(Test) 모드 진입의 핵심: 2가지 스위치 조작
실전 시험 세트에서는 훈련(Train)에서 해오던 기능들을 당장 멈추고 방어막을 쳐야 코드가 터지지 않습니다.

1. **`model.eval()`: "모델아, 이제 실전 평가 모드야!"**
   * Dropout 이나 Batch Normalization 계층 등 오직 모델의 '학습' 단계에서만 발동되는 특수 훈련용 장치들을 강제로 정지시키는 스위치입니다. 
2. **`with torch.no_grad():`: "미분 파이프라인 전원 차단!"**
   * 평가는 그저 정답을 맞추는 **순전파(Forward)** 연산만 있으면 됩니다. 불필요한 역전파 수집용 오토그라드 파이프라인이 켜져 있으면 메모리를 심각하게 잡아먹어 VRAM이 터집니다. 
   * 파이썬의 `with` 구문 특성상 **들여쓰기 된 블록 구간 내에서만** 배관 전원이 꺼지고, 이 평가 블록을 넘어가면 다시 미분 수집기계가 돌아가는 알뜰한 설계입니다.

> **💡 [꿀팁] 평가 끝난 스위치를 다시 학습(`train()`)으로 어떻게 돌리죠?**  
> 딥러닝 코드에서는 대개 매 에폭(Epoch)이 빙글빙글 반복해서 시작되는 반복문(`for epoch...`)의 **첫 번째 줄에 떡하니 `model.train()` 을 명시**합니다.  
> 실전 코드를 짜다 보면 밑에서 중간에 `eval()`을 써서 스위치를 내려버렸다고 치더라도, 코드가 평가를 마치고 위로 올라가 다시 **다음 훈련 에폭 주기를 준비할 때 무조건 처음 마주치는 게 `train()` 부팅**이므로 알아서 원래 훈련 모드로 무사 귀환하게 됩니다. 매우 훌륭한 안전망(Fail-safe) 설계죠!

---

## 7장. 실전 프로젝트 2: MNIST 합성곱 신경망 (CNN)
**[ 실습 파일: `src/07_mnist_cnn.py` ]**

이미지 처리에 특화된 "공간 보존"의 강력함을 피부로 증명할 고급 신경망 챕터입니다.
- **FC 모델 구조의 한계**: 바로 앞장(6장)에서 만든 선형 모델은 이미지를 한 줄기 끈처럼 구겨 펴버려, "숫자 구멍의 안쪽과 바깥쪽", "이전 줄 픽셀과 다음 줄 픽셀의 선분 연결" 같은 2차원 공간 연관성을 박살 내버렸습니다.
- **성과**: 6장의 FC 모델과 똑통한 파라미터 구조와 단지 5번(5 Epoch)의 똑같은 조건으로 재훈련을 진행한 결과, 눈 깜짝할 사이에 미친 퍼포먼스를 내며 **99.11%**라는 최고 수준의 정확도를 보였습니다. 이미지 데이터는 1차원 평탄화(Linear) 대비 공간 밀착형 합성곱(CNN) 뷰어 파이프라인이 압도적이라는 진리를 방증합니다.

---

## 8장. 천재의 뇌 영구 보존하기: 모델 Save & Load
**[ 연관 실습 파일: `src/07_mnist_cnn.py` ]**

며칠 동안 엄청난 전기세(GPU)를 태워 얻어낸 99% 정확도의 모델이 파이썬 창을 닫는 순간 휘발된다면 악몽일 것입니다. 완성된 신경망의 "가중치(Weight)"를 영구적인 파일 구조로 하드디스크에 구워내는 법을 배웁니다.

### 8.1 껍데기 말고 알맹이(가중치)만 빼서 저장하기
객체 통째로(`torch.save(model, path)`) 저장하는 것은 코드 구조에 종속되어 나중에 불러올 때 클래스 의존성 에러가 밥 먹듯 터집니다. **실무의 표준은 오직 숫자로만 이루어진 신경망 파라미터 딕셔너리(`state_dict`)만을 순수하게 축출해 저장하는 것입니다.**

```python
# 1. 모델의 모든 레이어의 파라미터가 담긴 순수 숫자 덩어리 추출
weights_only = model.state_dict()

# 2. .pth 혹은 .pt 확장자로 영구 보존
torch.save(weights_only, './mnist_cnn_model.pth')
print("가중치 저장 완료!")
```

### 8.2 영혼 주입하기: 가중치 불러오기 (Inference / 추론)
나중에 서비스(예: 서버/앱 환경)에 이 모델을 배포하거나 NPU에 탑재하기 직전, 평가만 쌩쌩하게 돌리고 싶을 때 사용합니다.

```python
# 1. 뼈대(클래스)를 똑같이 백지상태로 조립합니다.
model = MNIST_CNN().to(device)

# 2. 하드디스크에 보존된 영혼(가중치)을 불러옵니다.
loaded_weights = torch.load('./mnist_cnn_model.pth')

# 3. 뼈대에 영혼을 주입합니다. (이때 구조가 단 1줄이라도 다르면 튕겨냅니다!)
model.load_state_dict(loaded_weights)

# 4. 🚨 초보자의 잦은 실수: 가중치를 불러왔으면 제발 훈련 스위치를 끄세요!
model.eval() 
```

---

## 9장. 실전 전이 학습의 시작: `ImageFolder` 마법
**[ 실습 파일: `src/08_custom_image_dataset.py` ]**

지금까지는 PyTorch가 곱게 포장해둔 `datasets.MNIST` 장난감만 가지고 놀았습니다. 현업에서는 내 폴더에 쌓인 수십만 장의 불규칙한 사진 데이터를 모델에 직접 먹여야 합니다.

### 9.1 현업 표준 폴더 트리 구조 만들기
전 세계 컴퓨터 비전 연구자들이 공용으로 합의한 가장 아름다운 데이터 폴더 구조입니다. 사진이 담긴 **"가장 마지막 폴더의 이름"이 곧 모델이 외워야 할 "정답 반장(Label/Class)"**이 됩니다.

```text
hymenoptera_data/
  ├── train/               
  │     ├── ants/          <-- 이 폴더 안의 사진들은 알아서 예측 정답 '0'번표 획득!
  │     │    ├── 1.jpg
  │     │    └── 2.jpg
  │     └── bees/          <-- 이 폴더 안의 사진들은 알아서 예측 정답 '1'번표 획득!
  │          └── 3.jpg
  └── val/                 <-- 평가용 폴더도 train과 완벽히 동일한 구조로 짭니다.
```

### 9.2 `ImageFolder` 단 한 줄의 기적
데이터 폴더 구조만 위처럼 예쁘게 깎아두면, PyTorch의 `datasets.ImageFolder`라는 마법의 모듈이 사진 천만 장의 경로를 몽땅 스캔해서 **정답 라벨링 + 원핫 인코딩 딕셔너리 매핑을 단 1줄 만에 끝내줍니다.**

```python
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 1. [필수] 스마트폰 사진, DSLR 사진 크기가 전부 다르므로 입력 사이즈를 강제 통일!
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 모델 입으로 들어가기 위해 224x224 도마 위에서 균일하게 자름
    transforms.ToTensor(),         # 이미지 픽셀(0~255)을 파이토치 텐서(0.0~1.0)로 변환
])

# 2. 기적의 폴더 스캐닝
train_dataset = ImageFolder(root='./data/hymenoptera_data/train', transform=custom_transform)

print(f"클래스 종류: {train_dataset.classes}")        # ['ants', 'bees']
print(f"클래스 번호 매핑: {train_dataset.class_to_idx}") # {'ants': 0, 'bees': 1}

# 3. 배달 트럭(DataLoader)에 태우면 학습 준비 완료
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
```
이제 이 배달 트럭(`DataLoader`)을 4장에서 배웠던 **5단계 학습 루프**에 그대로 밀어 넣으면, "개미와 벌을 구별하는 나만의 새로운 AI 파라미터 구조"가 완성됩니다. 이것이 딥러닝 실무 파이프라인의 완성입니다!

---

## 10장. 거인의 어깨 위에 서다: 전이 학습 (Transfer Learning)
**[ 실습 파일: `src/09_custom_cnn_ants_bees.py` & `src/10_transfer_learning_resnet.py` ]**

우리가 모은 250장짜리 불쌍한 개미/벌 사진으로 모델을 밑바닥부터 훈련시키면(9번 스크립트) 정확도는 **60~66%** 근방에서 처참하게 무너집니다. 사진 개수가 너무 적어 모델이 수능의 본질을 깨닫지 못하고 기출문제의 "픽셀 위치만 통째로 암기"해버리는 최악의 과적합(Overfitting) 사태에 직면하기 때문입니다.

이를 타개할 현업 최강의 무기가 바로 **'전이 학습(Transfer Learning)'** 입니다. 구글, MS의 초천재 연구원들이 수백만 장의 방대한 사진으로 몇 달간 훈련시켜둔 "천재의 시각 피질 뇌 구조(Pre-trained Model)"를 그대로 다운로드해 빌려 쓰고, 가장 마지막 판단을 내리는 전두엽 세포(Linear Layer)만 우리가 원하는 클래스(개미, 벌)에 맞게 싹둑 자르고 꿰매어 쓰는 수술 기법입니다.

### 10.1 천재의 뇌 소환하기 (ResNet18)
`torchvision.models` 에는 이미 수없이 많은 선배들의 훌륭한 뇌(Architecture + Weight)가 준비되어 있습니다.

```python
from torchvision import models
from torchvision.models import ResNet18_Weights

# 1. 1,000개의 클래스를 기가 막히게 맞추는 ResNet18 구조와 "훈련을 마친 가중치" 다운로드
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 2. 마지막 전두엽 층(fc)의 원래 입력 크기를 기억해둠
num_features = model.fc.in_features 

# 3. 기존의 1000개짜리 출력 노드 덩어리를 떼어버리고, 
# 방금 기억해둔 크기에서 오직 2개(개미/벌)로만 출력을 꽂아주는 새로운 신형 부품(머리) 장착!
model.fc = nn.Linear(num_features, 2)
```

### 10.2 전이 학습 시의 마법 같은 훈련 속도
이렇게 뇌 수술을 마친 모델을 이전 4장에서 배웠던 **표준 5단계 훈련 루프**에 태워 훈련을 돌려보기만 하면 됩니다. 
* **결과**: `09_custom_cnn` 모델이 10번을 반복해도 60%대를 간신히 넘긴 반면, 이 스크립트(`10_transfer_learning`)에서는 거인의 어깨 위에 올라탄 덕분에 **단 1 에폭(Epoch) 만에 테스트 정확도 92%** 를 상회하며 최종적으로 **95% 이상의 파괴적 퍼포먼스**를 뽐냅니다!
* 현업 컴퓨터 비전 실무의 90% 이상은 처음부터 내 코드로 C++ 처럼 짜는 것이 아니라, 이렇게 Hugging Face 나 torchvision 에서 증명된 최고의 뇌를 빌려와 나의 200장짜리 커스텀 데이터셋에 구겨 넣어 재학습(Fine-Tuning) 시키는 것으로 마무리됩니다.

### 10.3 셜록 홈즈 파견 근무 (도대체 어떻게 이식이 가능한가?)
"1,000가지를 구별하던 모델이 어떻게 갑자기 개미와 벌만 구별하게 될까요?"
그 원리는 CNN의 두 가지 분업 구조에 있습니다.
1. **눈 (특징 추출기 - Conv 층들)**: ResNet18은 수백만 장의 사진을 분석하며 "둥근 모양, 질감, 색상, 가장자리" 같은 사물의 패턴을 완벽히 흡수한 **망막 세포(눈)**입니다. 이 '눈'의 능력은 세계 최고 수준입니다.
2. **머리 (분류기 - 맨 마지막 Linear 층)**: 이 눈에서 보내온 수많은 단서를 조합해 "아하! 이건 강아지다!"라고 외치는 곳입니다. 원래는 1,000개의 단어를 외치도록 설계되어 있습니다.

우리(10.1 섹션 코드)는 바로 이 마지막 1,000개짜리 입(머리)만 십자드라이버로 뜯어내 버리고, **"0번은 개미, 1번은 벌" 딱 2가지로만 말할 수 있는 백지상태의 깡통 스피커**를 새로 끼워 넣은 것입니다. 거인의 눈은 그대로 쓰기 때문에, 고작 200장의 데이터만 한 번 구경시켜 줘도 금세 "아, 더듬이가 둥글면 0번(개미)이라고 외치면 되는구나" 하고 규칙을 폭발적으로 깨닫게 됩니다.

### 10.4 미세 조정(Fine-Tuning) vs 특성 고정(Feature Extraction)
새로운 2개짜리 머리를 끼운 후 훈련을 시킬 때, 두 가지 전략의 갈림길에 서게 됩니다.

* **전략 A. 미세 조정 (Fine-Tuning) 👈 우리가 10번 스크립트에서 쓴 전략**
  새로운 파이프(머리)를 끼운 뒤, 전체 파라미터(거인의 눈부터 새 머리까지)를 한꺼번에 약한 힘(lr=0.001)으로 살살 조여가며 맞춥니다. 거인의 훌륭한 눈도 우리 데이터(개미/벌 숲 배경 등)의 **고유한 질감에 맞게 영점 조절**을 받게 되므로 성능이 가장 뛰어납니다.
* **전략 B. 특성 추출 (Feature Extraction)**
  "거인의 눈은 이미 완벽해!"라고 선언하고, 아예 기존 Conv 층의 파라미터들이 업데이트되지 않게 꽁꽁 얼려버립니다(`requires_grad=False`). 오직 **우리가 방금 끼운 '마지막 머리(Linear 층)' 단 하나만 가중치가 업데이트** 되도록 훈련합니다. 이 방식은 모바일에 올릴 때처럼 컴퓨터 자원이 극히 부족하거나 데이터가 10~20장 수준일 때 빠르게 훈련할 수 있는 비장의 무기입니다.

### 10.5 (보너스) 머리를 바꾸지 않은 순정 ResNet은 어떨까?
만약 파이프(Linear 머리)를 아예 교체하지 않고, **1000개의 영단어를 내뱉는 "순정 ResNet18"**(`10_2_test_original_resnet.py`)에게 우리 개미/벌 사진을 그냥 보여주면 어떻게 될까요?
놀랍게도 1000개의 보기(강아지, 자동차, 비행기 등) 속에서도 **74.5%의 확률로 개미(ant)와 벌(bee) 단어를 정확히 외칩니다!**
거인의 눈이 이미 개미와 벌의 형상을 완벽히 알고 있다는 명백한 증거이며, 전이 학습은 단지 이 방대한 지식을 '2가지 보기'로 깔때기만 꽂아 100%에 가깝게 정리해 주는 작업일 뿐입니다.

---

## 11장. "기억"을 가진 신경망: RNN 기초 (자연어 처리의 시작)
**[ 실습 파일: `src/11_rnn_sentiment_analysis.py` ]**

이미지(CNN)의 세계를 완벽히 정복했다면, 이제 가장 핫한 분야인 자연어 처리(NLP - Natural Language Processing)로 눈을 돌릴 차례입니다. ChatGPT 같은 언어 모델의 가장 기초적인 조상님인 **RNN(순환 신경망)**을 직접 조립해 보며 "시간과 순서"를 이해하는 AI를 만듭니다.

### 11.1 CNN의 한계와 RNN의 등장
앞선 CNN이 무결점처럼 보이지만 치명적인 약점이 딱 하나 있습니다. 바로 **"기억력(Memory)이 0"**이라는 점입니다. "나는 배가 너무 고프다. 그래서 사과를 __?" 라는 문장이 있을 때, CNN은 앞선 "배가 고프다"의 문맥 정보를 전혀 기억하지 못한 채 마지막 단어만 보고 답을 찍어야 합니다.

이를 극복하기 위해 등장한 **RNN(Recurrent Neural Network)** 은 마치 책을 읽는 인간처럼 동작합니다.
1. 첫 단어를 읽고 머릿속 빈 **메모장(Hidden State)** 에 감상을 적습니다.
2. 두 번째 단어를 읽을 때, "방금 전에 적어둔 첫 단어의 메모"를 보면서 합쳐서 감상을 덮어 씁니다.
3. 이렇게 문장 끝까지 순서대로 누적하며 읽어나가는 것이 RNN의 "순환(Recurrent)" 구조입니다!

### 11.2 영화 리뷰 긍정/부정 판단기 (감성 분석)
자연어 처리의 가장 클래식한 입문 문제인 감성 분석(Sentiment Analysis)을 실습합니다. 모델에게 단 4단어짜리 5개의 초소형 예제 데이터만 주고도, 긍정문(1)과 부정문(0)을 구별하도록 훈련합니다.

1. **임베딩(Embedding)**: "영화", "최고" 같은 한글을 컴퓨터는 읽지 못합니다. 단어에 번호표(예: 4번)를 부여한 뒤, 이를 아무 의미 없는 숫자가 아니라 **"의미를 내포한 방향성을 가진 좌표(Dense Vector)"** 로 바꿔주는 임베딩 층을 통과시킵니다.
2. **순차적 섭취 (RNN 층)**: 임베딩된 단어를 하나씩 순서대로 꿀꺽꿀꺽 삼키며 메모장(Hidden State)을 서서히 업데이트합니다.
3. **최종 결론 (Linear 층)**: 문장의 마지막 단어까지 다 읽어치우고 남은 **최종 메모장**을 들여다보고, "음.. 이건 99% 확률로 긍정이군!" 이라고 최종 점수를 냅니다.

### 11.3 단 100번의 훈련, 99.8%의 확신
데이터의 양이 고작 5개로 극단적으로 적어 오버피팅시키기 좋은 환경이기도 하지만, 파이토치 기본 기능(`nn.Embedding`, `nn.RNN`, `nn.Linear`) 단 세 개를 이은 초간단 뼈대를 이용해 에폭 100번을 돌린 결과, 모델은 주어진 문장의 긍정과 부정을 **99% 이상의 압도적인 확률값**으로 기가 막히게 구별해 냈습니다. 

이것이 언어 모델(Language Model)이 문맥을 이해하기 시작하는 가장 위대한 첫걸음입니다.

### 11.4 칵테일 섞기 (피드백 루프의 마법)
그렇다면 RNN은 매 단어가 들어올 때마다 "+1점, -1점" 식으로 점수를 단순히 더하는 걸까요? 아닙니다.

"정말" -> "돈" -> "아까워" 라는 문장이 들어온다고 상상해 봅시다.
1. 처음엔 맹물(초기 Hidden State)로 시작합니다.
2. "정말"이 들어오면 맹물에 '강조'라는 빨간 시럽을 섞어둡니다. (아직 긍정/부정 판단 불가)
3. "돈"이 들어오면 빨간 시럽 물에 섞습니다. (분위기가 어떻게 변할지 미묘함)
4. "아까워"라는 아주 쓴 독약이 들어오면서, 앞선 '정말' 시럽과 결합해 전체 물이 엄청나게 쓴맛의 **완성된 칵테일(최종 Hidden State)** 로 변합니다.
5. 마지막 Linear 층(소믈리에)은 이 최종 칵테일의 맛을 살짝 보고 "음, 이건 99% 부정이야!" 라고 확정 짓습니다.

과거의 인공지능(예: Andrej Karpathy의 `makemore` 초기 모델)은 **고정된 창문(Sliding Window)** 방식을 써서 한 번에 3글자씩만 쳐다보고 다음 글자를 예측했습니다. 그래서 창문 바깥(4글자 이전)의 일은 영원히 기억하지 못하는 바보였습니다.
하지만 RNN은 자신이 방금 만든 칵테일(과거의 기억)을 **다시 자기 입으로 들이부으며(피드백)** 새로운 재료(단어)를 섞기 때문에 아무리 긴 문장이라도 전체 문맥을 계속 눈덩이처럼 누적시킬 수 있는 기적을 만들어냅니다.
