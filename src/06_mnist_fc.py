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

def main():
    print("=== MNIST 분류 실습: 2-Layer FC 신경망 ===")
    
    # [1] 데이터 준비 전처리(Transform) 과정
    # 이미지를 Pytorch Tensor로 변환하고 0~1 사이로 정규화합니다.
    transform = transforms.Compose([transforms.ToTensor()])

    # [2] Dataset 다운로드 및 DataLoader 생성
    # 처음 실행 시 파이토치가 알아서 인터넷에서 데이터를 다운받아 옵니다.
    print("데이터셋 다운로드 및 로드 중...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # [3] 모델, 손실함수, 옵티마이저 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")
    
    model = MNIST_FC().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # [4] 모델 학습 루프 (Epoch: 5)
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train() # 모델을 학습 모드로 설정
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()        # 1. 기울기 초기화
            outputs = model(inputs)      # 2. 순전파
            loss = loss_fn(outputs, labels) # 3. 손실 계산
            loss.backward()              # 4. 역전파
            optimizer.step()             # 5. 가중치 업데이트
            
            running_loss += loss.item()

        # 에폭마다 평균 Loss 출력
        print(f"Epoch [{epoch}/{epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

    # [5] 모델 평가 (Test)
    model.eval() # 모델을 평가 모드로 설정 (Dropout, BatchNorm 등 동작 변경)
    correct = 0
    total = 0
    
    # 평가할 때는 기울기(Gradient) 계산이 필요 없으므로 메모리를 아끼기 위해 끕니다.
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # 예측값 구하기 (가장 값이 큰 인덱스)
            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += int((preds == labels).sum())

    accuracy = 100 * correct / total
    print(f"\n최종 테스트 정확도: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
