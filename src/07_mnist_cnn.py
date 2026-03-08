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
    # 파이토치는 GPU(cuda)가 있으면 GPU로, 없으면 CPU로 자동 할당하는 장치 코드를 관용적으로 씁니다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")
    
    # 모델을 장치(CPU/GPU) 메모리에 올립니다.
    model = MNIST_CNN().to(device)
    # MNIST는 다중 분류(0~9, 10개 클래스) 문제이므로 Softmax가 내장된 CrossEntropyLoss를 씁니다.
    loss_fn = nn.CrossEntropyLoss()
    # Adam 옵티마이저는 모델 내 모든 파라미터(W, b)를 찾아 학습률(0.001)만큼씩 업데이트 해 줍니다.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # [3] 모델 학습
    epochs = 5
    for epoch in range(1, epochs + 1):
        # ⭐️ 중요: 학습 시작 전에는 항상 '훈련 모드(Train Mode)'로 스위치를 켜야 합니다. Dropout 등 최적화기법이 동작하기 시작합니다.
        model.train()
        running_loss = 0.0
        
        # DataLoader에서 미니 배치(64개씩)를 순서대로 뽑아옵니다. (총 938번 반복)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 뽑아온 배치 이미지 데이터와 정답 데이터를 아까 올린 장치(CPU/GPU)로 보냅니다.
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()           # 1. (초기화) 이전 기울기 찌꺼기 버리기
            outputs = model(inputs)         # 2. (순전파) 모델에 64장 넣어서 예측값 64개 뱉어내기
            loss = loss_fn(outputs, labels) # 3. (Loss) 예측값(outputs)과 정답(labels) 비교해 오차 구하기
            loss.backward()                 # 4. (역전파) 오차를 바탕으로 각 노드의 미분값(기울기) 계산하기
            optimizer.step()                # 5. (업데이트) 계산된 기울기 방향대로 실질적인 가중치 수정(+/-)
            
            # 진행 상황을 보기 위해 Loss값 중 소수점(.item())만 뽑아서 누적합니다.
            running_loss += loss.item()
            
            # 200번 배치(약 12,800장 처리)가 돌 때마다 콘솔에 진행 상황 찍어주기
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"   [Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # 에폭(전체 문제집 1회독)마다 평균 Loss 출력
        print(f"Epoch [{epoch}/{epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

    # [4] 모델 평가
    # ⭐️ 평가 시에는 가중치 업데이트나 Dropout 등이 일어나면 절대 안 되므로 '평가 모드(Eval Mode)'로 스위치를 내립니다.
    model.eval()
    correct = 0 # 맞춘 개수 누적
    total = 0   # 전체 문제 개수 누적
    
    # ⭐️ 평가할 때는 계산 배관장치(그래디언트 계산 기능)를 완전히 잠가버리면 메모리와 연산 속도를 엄청나게 아낄수 있습니다.
    with torch.no_grad():
        # 이제 평가용 1만 장의 Test 데이터를 DataLoader에서 가져옵니다.
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 순전파로 예측만 수행합니다. (가중치는 바뀌지 않음)
            outputs = model(inputs)
            
            # 예측값(10개 노드 중 가장 높은 확률을 가진 노드의 인덱스 번호 = 숫자) 구하기
            preds = torch.argmax(outputs, dim=1)
            
            # 정답 개수와 맞춘 개수 카운트
            total += labels.size(0)
            correct += int((preds == labels).sum())

    # 최종 확률(맞춘 개수 / 백분율) 계산
    accuracy = 100 * correct / total
    print(f"\n최종 테스트 정확도 (CNN): {accuracy:.2f}%")

    # ==========================================
    # [5] 모델 저장 (Save)
    # ==========================================
    save_path = './mnist_cnn_model.pth'
    print(f"\n[저장] 훈련된 모델의 가중치를 '{save_path}'에 저장합니다...")
    # 모델의 현재 가중치(파라미터 숫자들)만 깔끔하게 빼내서 딕셔너리 형태로 묶어 영구 저장합니다.
    torch.save(model.state_dict(), save_path)
    print("저장 완료! 이제 코드가 종료되어도 똑똑해진 뇌(가중치)는 하드디스크에 남아있습니다.")

if __name__ == '__main__':
    main()
