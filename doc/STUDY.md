# PyTorch Study
## 1. 개요
PyTorch 학습을 위한 문서입니다.

## 2. 설치 및 환경 확인
- **가상환경**: `venv` 사용 (`Activate.ps1` 또는 `deactivate`)
- **설치 명령어**: `pip install torch torchvision torchaudio`

## 3. 텐서(Tensor) 기초
PyTorch의 핵심 데이터 구조는 텐서(Tensor)입니다. NumPy의 `ndarray`와 매우 유사하지만, **GPU를 활용한 연산 가속**이 가능하다는 점이 큰 차이입니다.

### 3.1. 텐서 생성
```python
import torch

# 1. 일반 리스트로부터 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2. NumPy 배열로부터 생성
import numpy as np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. 무작위 값 또는 상수 텐서 생성
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```

### 3.2. 텐서의 주요 속성 (Attributes)
```python
tensor = torch.rand(3, 4)
print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device: {tensor.device}") # cpu 또는 cuda:0 등
```

### 3.3. 텐서의 기본 조작 (Operations)
NumPy와 유사하게 인덱싱, 슬라이싱, 형태 변환, 결합 등의 풍부한 연산을 지원합니다.

#### 인덱싱과 슬라이싱 (Indexing & Slicing)
```python
tensor = torch.ones(4, 4)
print(f"첫 행: {tensor[0]}")
print(f"첫 열: {tensor[:, 0]}")
print(f"마지막 열: {tensor[..., -1]}")

# 특정 값 변경(in-place)
tensor[:, 1] = 0
```

#### 형태 변환 (View / Reshape)
텐서의 데이터 개수는 유지하되 모양(차원)만 바꿀 때 `.view()` 또는 `.reshape()`를 사용합니다.
- **`view()`**: 메모리를 복사하지 않고 기존 데이터를 그대로 공유합니다. 데이터가 메모리상 연속적(contiguous)일 때만 작동하며, 아니면 에러를 발생시켜 안전한 동작을 보장합니다.
- **`reshape()`**: 가능하면 `view()`처럼 메모리를 복사하지 않지만, 불가능할 경우 새로운 텐서로 알아서 복사(Copy)하여 결과를 내놓는 유연하고 안전한 방식입니다.

```python
x = torch.randn(4, 4)       # 4x4 행렬 (총 16개 요소)
y = x.view(16)              # 1차원 벡터로 변환
z = x.reshape(-1, 8)        # -1은 다른 모양을 보고 알아서 추론 (여기선 2x8이 됨)
```

#### 텐서 합치기 (Concatenate & Stack)
- `torch.cat`: 기존 차원을 따라 텐서들을 이어 붙일 때 사용합니다.
```python
# 1차원 방향(가로)으로 이어 붙이기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
```
- `torch.stack`: 아예 새로운 차원을 만들면서 텐서들을 쌓아 올릴 때 사용합니다.
```python
# 4x4 텐서 3개를 하나로 합쳐서 (3, 4, 4) 텐서로 만듦
t2 = torch.stack([tensor, tensor, tensor])
```

#### 스칼라(Scalar) 값 뽑아내기
1개의 요소만 들어있는 텐서에서 순수한 파이썬 숫자(int, float 등) 값만 꺼낼 때는 `.item()`을 사용합니다.
```python
agg = tensor.sum()       # 모든 원소의 합계 텐서 (크기 1)
agg_item = agg.item()    # 파이썬 숫자로 변환
print(agg_item, type(agg_item))
```

## 4. Autograd (자동 미분)
PyTorch의 모든 신경망 연산의 핵심은 `autograd` 패키지입니다. 이 패키지는 텐서의 모든 연산에 대해 자동 미분을 제공합니다.
- `requires_grad=True` 로 설정된 텐서는 모든 연산을 추적합니다.
- 계산이 완료된 후 `.backward()` 를 호출하면 모든 그래디언트(Gradient)가 자동으로 계산되어 `.grad` 속성에 누적됩니다.

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward() # 역전파(Backpropagation) 수행
print(x.grad)  # d(out)/dx의 결과가 출력됨
```


## 5. 신경망 모델 만들기 (torch.nn)
PyTorch에서 신경망은 `torch.nn` 패키지를 사용하여 생성합니다. `nn.Module`을 상속받아 모델 구조를 정의합니다.

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
        # 순전파(Forward pass) 정의
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = torch.flatten(x, 1) # 공간 차원을 평탄화
        x = self.fc1(x)
        return x

net = SimpleNet()
input_tensor = torch.randn(1, 1, 24, 24) # Batch, Channel, Height, Width
out = net(input_tensor) # 모델 예측
```

### 💡 꿀팁: Conv-Linear 연결부 입력 차원 맞추는 법
위 코드의 `nn.Linear(6 * 10 * 10, 10)` 부분처럼, 합성곱(Conv)층을 지나고 평탄화(Flatten)된 다음 완전연결층(Linear)으로 넘길 때는 입력 차원을 미리 계산해서 적어줘야 합니다. 계산하기가 복잡하다면 다음 방법들을 사용할 수 있습니다.

1. **에러로 확인하기**: `nn.Linear(1, 10)` 처럼 엉뚱한 숫자를 넣고 더미 데이터를 넣어보면 나타나는 에러 메시지의 차원을 파악하여 수정합니다.
2. **`nn.AdaptiveAvgPool2d` 사용**: 입력 크기 상관없이 고정된 크기(예: `5x5`)로 출력해주는 레이어를 추가하여 계산을 쉽게 고정합니다.
3. **`nn.LazyLinear` 사용 (최신 PyTorch 지원)**: `nn.LazyLinear(10)` 처럼 입력 노드 수는 생략하고 출력 결과(10)만 지정하면, 첫 데이터가 들어올 때 동적으로 입력 차원을 세팅해 줍니다!

## 6. 손실 함수(Loss Function)와 Softmax / Argmax 연산
분류(Classification) 모델을 만들 때, 마지막 출력(Logits)을 어떻게 처리하여 손실(Loss)을 구하는지 아는 것이 중요합니다.

- **`Softmax`**: 모델이 출력한 가공되지 않은 숫자(Logits)들을 모두 합쳐 `1.0(100%)`이 되도록 **확률 값으로 변환**해주는 함수입니다.
- **`Argmax`**: 모델 출력값 중에서 가장 높은 값(또는 가장 높은 확률)이 있는 곳의 **인덱스(순서) 번호**를 반환합니다. 결과적으로 모델이 예측한 최종 "정답 위치"를 알려줍니다.

### 💡 Loss Function(손실 함수)은 무엇으로 계산할까?
결론부터 말씀드리면, **보통 `Argmax`나 밖에서 계산한 `Softmax` 값 둘 다 쓰지 않고 모델이 출력한 "날것의 숫자값(Logits)"을 그대로 손실 함수에 넘겨줍니다!**

PyTorch에서 분류 시 가장 많이 쓰는 `nn.CrossEntropyLoss` 객체는 내부 구조가 매우 똑똑하게 설계되어 있습니다.
```python
loss_fn = nn.CrossEntropyLoss()

# 1. 모델에서 나온 날것의 출력 (Softmax 안 거침!)
out = net(input_tensor) 

# 2. 정답 라벨 (예: 10개 클래스 중 정답은 8번)
target = torch.tensor([8]) 

# 3. 모델 출력(out)을 그대로 넣으면, 내부에서 알아서 Softmax 확률 변환을 하고 Loss를 구함
loss = loss_fn(out, target)
print(loss)
```

**[정리]**
* PyTorch의 `nn.CrossEntropyLoss`는 **내부적으로 `Softmax` 연산과 `Log(로그)` 연산을 모두 포함**하고 있습니다.
* 모델을 시각적으로 확인(`print`)할 때는 밖에서 `softmax`나 `argmax`를 임시로 써서 결과를 보는 것이 맞습니다.
* 하지만 실제 학습 과정 중 **Loss 값을 구할 때는 절대 모델 끝에 Softmax를 강제로 씌우거나 Argmax를 한 값을 넣지 않습니다.** 모델의 마지막 Linear 층에서 나온 상태(Logits)를 그대로 Loss 통(Function) 안에 골인시키면 됩니다!

## 7. 옵티마이저(Optimizer)와 기본 학습 루프 (Training Loop)
모델이 출력한 값으로 Loss(오차)를 구하고 `.backward()`로 미분값(기울기, Gradient)을 계산했다면, 이제 이 기울기 방향대로 모델의 가중치를 업데이트(학습)해야 합니다. 이 역할을 하는 것이 **옵티마이저(Optimizer)** 입니다.

### 7.1. 옵티마이저 생성 및 모델 파라미터 확인
`torch.optim` 패키지에 다양한 옵티마이저(`SGD`, `Adam`, `RMSprop` 등)가 있습니다. 옵티마이저를 만들 때 "어떤 가중치를 업데이트할 것인가"를 반드시 알려줘야 합니다.

```python
import torch.optim as optim

# 우리가 만든 모델(net)의 모든 파라미터(net.parameters())를 Adam 방식으로 업데이트함.
# lr 은 학습률(Learning Rate)로, 한 걸음(업데이트)의 크기를 결정합니다.
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

#### 💡 `net.parameters()`의 마법과 `named_parameters()`
우리가 `SimpleNet`을 만들 때 `nn.Module`을 상속받았기 때문에, 내부에 `nn.Conv2d`나 `nn.Linear` 등을 선언하는 순간 부모 클래스가 알아서 "학습해야 할 가중치" 목록에 이들을 등록해 둡니다.

만약 모델 안에 어떤 파라미터 변수들이 어떤 이름과 크기로 등록되어 있는지 확인하고 싶다면 **`net.named_parameters()`** 를 사용합니다.

```python
# 모델 내 학습 가능한 파라미터의 '이름'과 '형태(Shape)' 확인하기
print("--- 모델 파라미터 정보 ---")
for name, param in net.named_parameters():
    print(f"이름: {name} | 형태: {param.shape}")
```

**[실행 결과 예시]**
```text
--- 모델 파라미터 정보 ---
이름: conv1.weight | 형태: torch.Size([6, 1, 5, 5])
이름: conv1.bias | 형태: torch.Size([6])
이름: fc1.weight | 형태: torch.Size([10, 600])
이름: fc1.bias | 형태: torch.Size([10])
```
이름에 `.weight`(가중치)와 `.bias`(편향)가 자동으로 붙어 생성되는 것을 볼 수 있습니다! 이를 활용해 추후 특정 레이어만 골라서 학습시키는(전이 학습) 등 세밀한 통제가 가능해집니다.

### 7.2. PyTorch 표준 5단계 학습 루프
PyTorch 개발자라면 누구나 숨 쉬듯 자연스럽게 쓰는 5단계 표준 학습 루프입니다.

```python
# 에폭(Epoch): 전체 데이터를 몇 번 반복 학습할 것인지 횟수
for epoch in range(10): 
    # [1단계] 입력 데이터와 정답 준비
    inputs, labels = get_data() 

    # [2단계] 미분값(Gradient) 초기화
    # 저번 학습 스텝에서 누적된 기울기를 0으로 비워줍니다. (안 하면 기울기가 계속 계속 더해짐)
    optimizer.zero_grad() 

    # [3단계] 순전파 (Forward) 및 Loss 계산
    outputs = net(inputs)
    loss = loss_fn(outputs, labels)

    # [4단계] 역전파 (Backward)
    # Loss를 바탕으로 모델 각 가중치들이 얼마나 틀렸는지(기울기) 전부 계산합니다.
    loss.backward()

    # [5단계] 가중치 업데이트 (Step)
    # 4단계에서 구해둔 기울기(Grad)를 보고, Optimizer가 한 걸음(Step) 이동해 가중치를 바꿉니다.
    optimizer.step()
    
    print(f"Epoch {epoch} - Loss: {loss.item()}")
```

**[주의할 점]**
`optimizer.zero_grad()`를 깜빡하면 이전 배치의 기울기가 누적되어 제대로 된 방향으로 스텝을 밟지 못하게 됩니다. 루프가 돌 때마다 기울기는 무조건 초기화해주어야 한다는 것을 꼭 기억하세요!

### 💡 꿀팁: `(4, )` 처럼 쉼표가 들어간 이유 (파이썬 튜플 요소값 지정)
PyTorch의 `torch.randn`이나 `torch.randint` 같이 텐서의 크기(Shape)를 넘겨줘야 하는 함수들은 입력으로 리스트(`[]`)나 튜플(`()`)을 받습니다.

- `(4, 5)`: 4행 5열이라는 2차원 크기를 뜻하는 튜플
- **`(4,)`**: **원소가 4개인 1차원 벡터**를 뜻하는 튜플입니다.
  - 파이썬에서 괄호 안에 숫자를 딱 하나만 넣고 `(4)`라고 쓰면, 컴퓨터는 이것을 튜플이 아니라 그냥 일반적인 **숫자 4**를 괄호로 감싼 것으로 처리합니다.
  - 그래서 "이거 빈 숫자 아니고 1개짜리 튜플 묶음이야!" 라는 걸 파이썬에게 확실히 알려주기 위해 뒤에 쉼표(`,`)를 찍어주는 것입니다.

헷갈리신다면 `[4]` 처럼 리스트 기호를 쓰셔도 똑같이 1차원 배열(길이 4)로 잘 인식합니다!

## 8. 데이터셋(Dataset)과 데이터로더(DataLoader)
우리가 앞서 만든 5단계 학습 루프에서 `inputs, labels = get_data()` 부분을 해결해 주는 녀석들입니다. 실제 딥러닝에서 데이터는 보통 수만 개~수억 개 단위이므로 모든 데이터를 한 번에 메모리에 올릴 수 없습니다. 때문에 **조금씩 잘라서(Batch) 메모리에 올려주는 기능**이 필요합니다.

### 8.1. `Dataset`: 데이터 1개씩 꺼내주는 자판기
모든 데이터셋은 PyTorch의 `torch.utils.data.Dataset` 클래스를 부모로 상속받아 만들어야 합니다. 이 안에는 반드시 3개의 필수(Magic) 함수가 정의되어야 합니다.
- **`__init__()`**: 초기화, 원본 데이터 불러오기 파일 경로 지정 등
- **`__len__()`**: 이 데이터셋의 총 길이(개수)를 반환
- **`__getitem__(idx)`**: 순서(idx)번째에 해당하는 데이터와 라벨 딱 1개 묶음을 찾아서 가공해 뱉어내기

### 8.2. `DataLoader`: 잘라서 섞어주는 배달 트럭
`Dataset`이 데이터를 하나씩만 줄 수 있다면, `DataLoader`는 이를 받아 우리가 지정한 **배치 크기(Batch Size)**만큼 차곡차곡 모아(Stacking) 모델에 한 뭉텅이씩 전달해 줍니다. 
```python
my_loader = DataLoader(
    dataset=my_dataset, 
    batch_size=32,      # 한 번에 32개씩 묶어주세요
    shuffle=True,       # 뽑기 전에 데이터 순서를 뒤죽박죽 섞어주세요
    drop_last=True      # 마지막에 데이터가 32개로 딱 안 떨어지면 버리세요.
)

# for 문을 돌리면 지정된 배치 크기만큼 텐서가 튀어나옵니다!
for batch_x, batch_y in my_loader:
    out = net(batch_x)
```
- **`shuffle=True`**: 매우 강력히 권장됩니다. 데이터를 섞지 않으면 모델이 데이터의 "순서"를 정답으로 외워버리기 때문입니다.
- **`drop_last=True/False`**: 데이터 개수가 배치 크기로 딱 안 떨어질 때 씁니다. 배치 차원이 일치해야 돌아가는 까다로운 모델 구조에서는 마지막 자투리 배치가 Shape 에러를 뿜을 수 있어 주로 `True`로 버리는 옵션을 킵니다.

## 9. 실전 미니 프로젝트: MNIST 손글씨 분류
지금까지 배운 내용(Dataset, DataLoader, nn.Module, Optimizer)을 모두 합쳐서 실제 0~9 손글씨 이미지를 분류하는 예제를 살펴봅시다. 실습 코드는 `src/06_mnist_fc.py`와 `src/07_mnist_cnn.py`에 구현되어 있습니다.

### 9.1. 데이터 전처리와 `Batch`
* **MNIST 데이터셋**: 6만 장의 흑백 손글씨 훈련 이미지와 1만 장의 테스트용 이미지로 구성됩니다.
* 이 6만 장의 사진을 모델에게 한 번에 보여주면 컴퓨터 메모리(VRAM)가 터져버립니다! 따라서 `DataLoader`를 통해 **64장 단위(Batch Size=64)로 쪼개서** 모델에게 야금야금 보여줍니다. 
* 이렇게 64장씩 묶은 덩어리를 "미니 배치(Mini-batch)"라고 부릅니다. 6만 장을 64개씩 묶으면 1번에 약 938개의 묶음(Step)이 나오게 됩니다.

### 9.2. 모델 훈련과 `Epoch`
* **에폭(Epoch)**: 938개의 묶음을 처음부터 끝까지 다 보여줘서 6만 장의 '문제집' 전체를 모델이 **1번 완전하게 다 풀어본 것(1회독)을 "1 Epoch"** 라고 부릅니다.
* 실습 코드에서는 똑같은 6만 장의 데이터 묶음을 **총 5번 반복(Epoch=5)** 해서 보여주도록 파이썬 반복문(for 문)을 짰습니다. 모델이 5회독을 거치면서 미세한 특징까지 외우도록(학습) 유도한 것입니다.

### 9.3. 2-Layer FC (초급자 버전)
- **구조**: 이미지를 가로, 세로 유지 없이 `28x28(784) 픽셀`의 1차원 긴 줄로 쭈욱 펴버린 다음(`view`), `128개의 은닉 노드`를 거쳐 `10개(0~9 클래스)`의 최종 노드로 데이터를 압축합니다.
- **결과**: `Epoch`가 반복될수록 `Loss`값이 떨어지며 스스로 오차를 줄여나갔고, 테스트 이미지 1만 장을 대상으로 **97.48%** 의 꽤 훌륭한 정확도를 보여주었습니다.

### 9.4. CNN (고급자 버전)
- **구조**: 사람의 시각 처리와 똑같이 작동합니다. 이미지를 1차원으로 구겨 펴지 않고 2차원 공간 형태를 그대로 유지하면서, **돋보기 역할(Conv2d 레이어)**로 훑으며 주변의 외곽선, 명암 등 이미지 픽셀 패턴(특징)을 조각조각 뽑아내는 이미지 특화 구조입니다. 공간 정보가 살아 있습니다!
- **결과**: 이미지 구조를 보존하며 똑같이 5번(5 Epoch)을 훈련한 결과, 테스트 데이터에서 **99.11%**라는 놀라운 정확도를 보였습니다. 이미지 데이터는 FC 신경망보다 Convolutional (CNN) 망이 압도적으로 유리하다는 가장 극명한 사례입니다.

## 10. 딥러닝 필수 용어: 에폭(Epoch)과 배치(Batch)
학습 루프 코드를 짜다 보면 `Epoch`와 `Batch`라는 단어가 끊임없이 등장합니다. 딥러닝 모델이 공부하는 방식을 비유를 통해 이해해 봅시다.

- **데이터셋(Dataset)** = 우리가 이번 학기에 풀어야 할 전체 "수학 문제집 1권" (예: 문제 총 1,000개)
- **배치(Batch Size)** = 하루에 풀 "문제 개수" (예: 하루 100문제씩 풀기 = Batch Size 100)
- **스텝(Step/Iteration)** = 채점하고 오답 노트를 쓰는 횟수 (1,000문제를 100개씩 풀면 총 **10번** 채점을 받아야 문제집 끝까지 다 풀 수 있죠? 즉, 10 Steps)
- **에폭(Epoch)** = 수학 문제집 1권을 처음부터 끝까지 "N번째 다시 풀기" (예: 1 Epoch = 문제집 1번 완독, 5 Epoch = 똑같은 문제집 5회독)

### 💡 왜 묶어서(Batch) 풀고, 왜 여러 번(Epoch) 반복해서 풀까?
1. **배치(Batch)**: 한 번에 수만 개의 문제를 풀고 한 번만 채점(Loss 계산)하면 컴퓨터 메모리(VRAM)가 터져버리고, 반대로 딱 1문제씩 풀고 채점하면 학습 방향이 들쭉날쭉 너무 불안정해집니다. 따라서 "적당량의 문제 묶음"을 풀어서 전체적인 경향(기울기)을 파악하는 것이 효율적입니다.
2. **에폭(Epoch)**: 사람도 어려운 수학 문제를 한 번 풀었다고 다 외우지 못하듯이, AI 모델도 수만 장의 사진을 1번만 보면 특징을 다 파악하지 못합니다. 똑같은 데이터셋을 여러 번 돌려보며(Epoch 반복) 미세한 특징까지 학습(가중치 조율)해야 높은 정확도가 나옵니다.

## 11. Train Set과 Test Set (데이터 분리)
머신러닝과 딥러닝에서 데이터를 수집하면 절대 전체 데이터를 모두 모델 학습에 쓰지 않고, 보통 **8:2** 또는 **7:3** 비율로 쪼개서 사용합니다.

* **Train Set (학습 데이터, 약 80%)**: 모델이 반복해서 보면서 스스로 가중치(규칙)를 업데이트(학습)하는 데 사용하는 데이터입니다. (기출문제)
* **Test Set (테스트 데이터, 약 20%)**: 학습이 끝난 후 모델의 성능을 "평가"하기 위해서만 사용하는 데이터입니다. 학습 중에는 절대 모델에게 보여주지 않으며, `loss.backward()`나 `optimizer.step()` 같은 업데이트 로직도 돌리지 않습니다. (수능 실전문제)

### 💡 왜 나눠야 하나요? (과적합, Overfitting 방지)
모델에게 100장의 고양이 사진(Train Set)을 달달 외우도록 훈련시켰다고 가정합시다. 만약 평가할 때도 그 100장의 똑같은 사진으로 시험을 본다면 모델은 사진 속의 고양이가 어떻게 생겼는지 본질을 학습한 것이 아니라, 그저 **100장 사진 자체의 픽셀 위치를 통째로 외워버렸을 확률**이 높습니다. 이를 과적합(Overfitting)이라고 합니다.

따라서 모델이 처음 보는 새로운 사진(Test Set)에 대해서도 맞출 수 있는 **일반화(Generalization) 능력**을 갖추었는지 공정하게 테스트하기 위해, 정답을 가리고 평가만 하는 Test Set을 떼어두어야 합니다!

### 💻 코드에서는 어떻게 적용하나요?

데이터의 출처에 따라 Train/Test 분리 구현 방법이 다릅니다.

**방법 A. 파이토치 내장 데이터셋 모듈 사용시 (예: MNIST)**
`torchvision.datasets`에서 제공하는 유명한 데이터셋들은 이미 제작 시점부터 "훈련용(Train)" 파일 덩어리와 "평가용(Test)" 파일 덩어리가 아예 별개로 분리되어 인터넷에 올라가 있습니다.
따라서 우리는 코드를 직접 분할(Split)할 필요 없이 아래처럼 다운로드 옵션만 다르게 주면 됩니다.
```python
# 분리된 원본 6만 장짜리 훈련용 데이터 뭉치 가져오기
train_dataset = datasets.MNIST(root='./data', train=True)

# 분리된 원본 1만 장짜리 평가용 데이터 뭉치 가져오기
test_dataset = datasets.MNIST(root='./data', train=False)
```

**방법 B. 내가 가진 일반 데이터셋(Custom Dataset)을 쪼갤 때**
내 컴퓨터에 있는 사진 1,000장을 하나로 묶어놓은 `MyCustomDataset`을 직접 8:2로 쪼개야 할 때는 `random_split` 함수를 사용합니다. 이 함수는 순서대로 쪼개지 않고 데이터를 이름 그대로 무작위(Random)로 섞어준 뒤 정확한 개수만큼 두 덩어리로 분할해 줍니다.
```python
from torch.utils.data import random_split

# 1. 원본 데이터셋 1000개
my_dataset = MyCustomDataset(data_size=1000)

# 2. 비율 계산 (8:2)
train_size = int(len(my_dataset) * 0.8)  # 800개
test_size = len(my_dataset) - train_size # 200개

# 3. random_split 함수로 알아서 섞어서 쪼개기!
train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])
```

### 🚨 주의사항: 평가(Test)를 돌릴 때 꼭 써야 하는 2가지 마법의 주문!
학습(Train)이 끝난 모델을 Test 데이터로 평가할 때는 절대로 데이터로 학습을 진행해서는 안 됩니다. 파이토치에서는 이를 위해 다음 두 가지 방어막을 반드시 선언해야 합니다.

1. **`model.eval()`**: 모델아, 이제부터 실전 평가 모드야!
   * 딥러닝 모델 중에는 오직 '학습'을 할 때만 켜져야 하는 특수한 기술들(Dropout, BatchNorm 등)이 있습니다.
   * 실전 평가를 할 때 이 기술들이 켜져 있으면 결과가 엉망으로 튀기 때문에, `model.eval()`을 호출해 **"평가용으로 내부 스위치를 다 꺼라"** 라고 지시해야 합니다. (다시 학습으로 돌아갈 때는 `model.train()`을 호출합니다.)

2. **`with torch.no_grad():`**: 미분 계산 파이프라인 잠그기!
   * 파이토치는 기본적으로 텐서가 지나가는 모든 길에 "역전파를 위한 미분 계산 파이프라인(Autograd)"을 깔아둡니다. 이 파이프라인은 엄청난 메모리와 컴퓨터 자원을 잡아먹습니다.
   * 평가는 그저 정답을 맞추는 **순전파(Forward)** 만 필요하고 역전파 미분을 통한 업데이트가 전혀 필요없으므로, `with torch.no_grad():` 블록을 열어 **"이 아래부터는 미분 계산 파이프라인을 전부 잠가라!"** 라고 지시하면 속도도 훨씬 빨라지고 메모리 터짐(OOM)을 방지할 수 있습니다.
   * **💡 들여쓰기(Block) 범위**: 파이썬의 `with` 구문 특성상, 이 코드로 **들여쓰기 된 블록 내부**에서만 배관 전원이 꺼지고, 이 블록을 빠져나오면 자동으로 다시 배관 전원이 켜지므로 다음 Epoch 학습 시 별도의 켜는 코드를 적을 필요가 없습니다!

## 12. 돌릴 때마다 결과(정확도)가 달라지는 이유: 무작위성(Randomness)과 시드(Seed)
같은 코드, 같은 데이터를 써서 학습을 시켰는데 어떨 때는 정확도가 97%, 어떨 때는 98%로 계속 바뀌는 것을 볼 수 있습니다. 딥러닝에는 기본적으로 **"무작위(Random)" 요소**가 여러 군데 숨어 있기 때문입니다.

### 🎲 무작위성이 발생하게 되는 원인
1. **가중치 초기화 (Weight Initialization)**: 처음에 `nn.Linear`나 `nn.Conv2d` 같은 레이어를 만들면 PyTorch는 그 안에 있는 가중치(w) 값을 0이 아니라 아무 "랜덤한 무작위 숫자"로 채워 넣습니다. 출발선이 매번 다르니 도착지(결과)도 미세하게 달라집니다.
2. **배치 셔플 (DataLoader Shuffle)**: `DataLoader`에서 `shuffle=True` 옵션을 주었기 때문에 매 Epoch마다 모델이 문제를 푸는(Batch) "순서"가 무작위로 계속 뒤섞입니다.

### 🔒 시드(Seed) 고정: 매번 똑같은 결과를 얻고 싶다면?
실험 결과를 남들에게 증명해야 하거나, 버그를 찾으려면 코드를 돌릴 때마다 결과가 100% 똑같이 나와야 합니다. 이를 "재현성(Reproducibility)"이라고 부르며, 다음과 같이 맨 윗줄에 **Random Seed 번호를 고정**해 주면 됩니다. (어떤 숫자든 상관없으며 보통 `42`를 많이 씁니다.)

```python
import torch
import random
import numpy as np

# 파이토치 시드 고정
torch.manual_seed(42)

# (선택) GPU 연산, 파이썬 기본 난수, 넘파이 난수까지 모두 고정하고 싶을 때
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
```
이렇게 시드를 고정하면 내부의 "난수 생성기"가 항상 같은 패턴의 가짜 난수만 뱉어내게 되므로, 100번을 돌려도 정확히 일치하는 소수점 아래 정확도가 나오게 됩니다!
