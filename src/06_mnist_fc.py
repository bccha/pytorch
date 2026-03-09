import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 2-Layer Fully Connected (FC) 신경망 정의
# ==========================================
class MNIST_FC(nn.Module):
    def __init__(self):
        super(MNIST_FC, self).__init__()
        # MNIST 이미지는 28x28 픽셀 = 784 픽셀
        # Layer 1: 입력 784 -> 은닉층 128
        self.fc1 = nn.Linear(28 * 28, 128)
        # 활성화 함수
        self.relu = nn.ReLU()
        # Layer 2: 은닉층 128 -> 출력 10 (0~9 숫자)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 이미지를 1차원 벡터로 쫙 폅니다. [Batch, 1, 28, 28] -> [Batch, 784]
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    """
    [4] 모델 학습을 수행하는 루프 함수입니다.
    """
    for epoch in range(1, epochs + 1):
        # ⭐️ 중요: 학습 시작 전에는 항상 '훈련 모드(Train Mode)'로 스위치를 켜야 합니다.
        model.train() 
        running_loss = 0.0
        
        # DataLoader에서 미니 배치(64개씩)를 순서대로 뽑아옵니다. (총 938번 반복)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 뽑아온 배치 이미지 데이터와 정답 데이터를 장치(CPU/GPU)로 보냅니다.
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()           # 1. (초기화) 이전 기울기 찌꺼기 버리기
            outputs = model(inputs)         # 2. (순전파) 모델에 64장 넣어서 예측값 64개 뱉어내기
            loss = criterion(outputs, labels) # 3. (Loss) 예측값과 정답 비교해 오차 구하기
            loss.backward()                 # 4. (역전파) 오차를 바탕으로 각 노드의 미분값 계산
            optimizer.step()                # 5. (업데이트) 계산된 기울기 방향대로 실질적인 가중치 수정
            
            # 진행 상황을 보기 위해 Loss값 누적
            running_loss += loss.item()

        # 에폭(전체 문제집 1회독)마다 평균 Loss 출력
        print(f"Epoch [{epoch}/{epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """
    [5] 모델 평가 전용 함수입니다. 
    @torch.no_grad() 데코레이터를 붙여 함수 내부의 모든 연산에서 메모리 절약을 위한 미분 추적을 원천 차단합니다.
    """
    # ⭐️ 평가 시에는 가중치 업데이트나 Dropout 절대금지! '평가 모드(Eval Mode)'로 스위치를 내립니다.
    model.eval() 
    correct = 0 # 맞춘 개수 누적
    total = 0   # 전체 문제 개수 누적
    
    # 평가용 1만 장의 Test 데이터를 DataLoader에서 가져옵니다.
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 순전파로 예측만 수행 (미분 계산 그래프 X)
        outputs = model(inputs)
        
        # 예측값(10개 노드 중 가장 높은 확률을 가진 노드의 인덱스 번호 = 숫자) 구하기
        preds = torch.argmax(outputs, dim=1)
        
        # 정답 개수와 맞춘 개수 카운트
        total += labels.size(0)
        correct += int((preds == labels).sum())

    # 최종 확률(맞춘 개수 / 백분율) 계산
    accuracy = 100 * correct / total
    print(f"\n최종 테스트 정확도: {accuracy:.2f}%")

def main():
    print("=== MNIST 분류 실습: 2-Layer FC 신경망 ===")
    
    # [1] 데이터 준비 전처리(Transform) 과정
    transform = transforms.Compose([transforms.ToTensor()])

    # [2] Dataset 다운로드 및 DataLoader 생성
    print("데이터셋 다운로드 및 로드 중...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # [3] 모델, 손실함수, 옵티마이저 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")
    
    model = MNIST_FC().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 깔끔하게 분리된 훈련 함수 호출
    print("\n[훈련 시작]")
    train_model(model, train_loader, criterion, optimizer, device, epochs=5)

    # 깔끔하게 분리된 평가 함수 호출
    print("\n[평가 시작]")
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
