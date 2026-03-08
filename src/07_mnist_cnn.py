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
        
        # 첫 번째 Conv 레이어: 입력 채널 1, 출력 채널 32, 커널(필터) 사이즈 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 크기가 28x28 -> 14x14 로 줄어듦
        
        # 두 번째 Conv 레이어: 입력 32, 출력 64, 커널 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 크기가 14x14 -> 7x7 로 줄어듦
        
        # 완전 연결층 (Fully Connected Layer)
        # 특징 맵 크기: 64채널 * 7가로 * 7세로
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 0~9 클래스 출력

    def forward(self, x):
        # Feature Extraction (특징 추출 파트)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # 분류를 위해 1차원으로 쭉 폅니다 (Flatten)
        x = x.view(x.size(0), -1) 
        
        # Classification (분류 파트)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # [3] 모델 학습
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"   [Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch}/{epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

    # [4] 모델 평가
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += int((preds == labels).sum())

    accuracy = 100 * correct / total
    print(f"\n최종 테스트 정확도 (CNN): {accuracy:.2f}%")

if __name__ == '__main__':
    main()
