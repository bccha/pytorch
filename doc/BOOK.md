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
- **CNN 구조의 우월함**: 인간의 각막 망막 구조처럼 2차원 공간 형태를 그대로 유지하면서, **돋보기 역할(Conv2d 레이어)**로 이미지를 훑어 주변 외곽선 패턴(특징) 등을 뽑아냅니다.
- **성과**: 6장의 FC 모델과 똑통한 파라미터 구조와 단지 5번(5 Epoch)의 똑같은 조건으로 재훈련을 진행한 결과, 눈 깜짝할 사이에 미친 퍼포먼스를 내며 **99.11%**라는 최고 수준의 정확도를 보였습니다. 이미지 데이터는 1차원 평탄화(Linear) 대비 공간 밀착형 합성곱(CNN) 뷰어 파이프라인이 압도적이라는 진리를 방증합니다.
