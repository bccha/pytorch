import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. Convolutional Neural Network (CNN) 정의
# ==========================================
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # 이미지 크기: 1채널 28x28
        
        # 첫 번째 Conv 레이어: 입력 채널 1, 출력 채널 32, 커널 3x3, stride 2 (크기 반으로 줆)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # [추가] 2D 공간 정보 유지 정규화
        self.relu1 = nn.ReLU()
        # MaxPool 제거
        
        # 두 번째 Conv 레이어: 입력 32, 출력 64, 커널 3x3, stride 2 (다시 크기 반으로 줆)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # [추가] 64채널 2D 정규화
        self.relu2 = nn.ReLU()
        # MaxPool 제거
        
        # 완전 연결층 (Fully Connected Layer)
        # 특징 맵 크기: 64채널 * 7가로 * 7세로
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128) # [추가] 1차원으로 펴진 벡터 정규화
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 0~9 클래스 출력

    def forward(self, x):
        # 1층 통과: Conv(stride=2) -> BatchNorm -> ReLU
        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.relu1(x)
        
        # 2층 통과: Conv(stride=2) -> BatchNorm -> ReLU
        x = self.conv2(x)
        x = self.bn2(x) 
        x = self.relu2(x)
        
        # 2차원 공간 정보를 1차원 벡터로 일렬로 쫙 폅니다.
        # 크기 변환: [Batch, 64, 7, 7] -> [Batch, 64 * 7 * 7]
        x = x.view(-1, 64 * 7 * 7)
        
        # 마지막 FC층: Linear -> BatchNorm -> ReLU -> Linear
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    """
    [3] 모델 학습 루프
    """
    for epoch in range(1, epochs + 1):
        # ⭐️ 중요: 학습 시작 전에는 항상 '훈련 모드(Train Mode)'로 스위치를 켜야 합니다.
        model.train()
        running_loss = 0.0
        
        # DataLoader에서 미니 배치를 순서대로 뽑아옵니다.
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()           # 1. 초기화
            outputs = model(inputs)         # 2. 순전파
            loss = criterion(outputs, labels) # 3. 오차 계산
            loss.backward()                 # 4. 역전파
            optimizer.step()                # 5. 가중치 업데이트
            
            # Loss 누적
            running_loss += loss.item()
            
            # 200번 배치(약 12,800장 처리)가 돌 때마다 콘솔에 진행 상황 찍어주기
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"   [Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # 에폭 평균 Loss 출력
        print(f"Epoch [{epoch}/{epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """
    [4] 모델 평가 함수. 데코레이터를 통해 미분 파이프라인 전원을 차단합니다.
    """
    # ⭐️ 평가 시에는 가중치 업데이트, Dropout 절대 금지. '평가 모드'
    model.eval()
    correct = 0 # 맞춘 개수 누적
    total = 0   # 전체 문제 개수 누적
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 순전파 예측
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        
        total += labels.size(0)
        correct += int((preds == labels).sum())

    # 최종 확률 계산
    accuracy = 100 * correct / total
    print(f"\n최종 테스트 정확도 (CNN): {accuracy:.2f}%")

def main():
    print("=== MNIST 분류 실습: CNN 신경망 ===")
    
    # [1] 데이터 준비 전처리
    transform = transforms.Compose([transforms.ToTensor()])

    print("데이터셋 다운로드 및 로드 중...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # [2] 장치 설정 및 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")
    
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # [3] 모듈화된 학습 함수 호출
    print("\n[훈련 시작]")
    train_model(model, train_loader, criterion, optimizer, device, epochs=5)

    # [4] 모듈화된 평가 함수 호출
    print("\n[평가 시작]")
    evaluate_model(model, test_loader, device)

    # [5] 모델 저장 (Save)
    save_path = './mnist_cnn_model.pth'
    print(f"\n[저장] 훈련된 모델의 가중치를 '{save_path}'에 저장합니다...")
    torch.save(model.state_dict(), save_path)
    print("저장 완료! 이제 코드가 종료되어도 뇌(가중치)는 하드디스크에 남아있습니다.")

if __name__ == '__main__':
    main()
