import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 밑바닥부터 훈련할 쌩초보 CNN 모델 정의
# ==========================================
class SimpleAntBeeCNN(nn.Module):
    def __init__(self):
        super(SimpleAntBeeCNN, self).__init__()
        # 입력: [배치, 3채널(RGB), 224, 224]
        # (이미지 사이즈가 크므로 풀링을 더 많이 줍니다)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 224 -> 112
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112 -> 56
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )
        
        # 64채널 * 가로28 * 세로28 = 50,176
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 2) # 개미와 벌 (2개 클래스)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, epochs=10):
    """
    [4] 훈련 및 검증 루프
    각 Epoch 마다 Train과 Val 단계를 번갈아가며 진행하며 최적의 모델 정확도를 기록합니다.
    """
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

    print(f"\n학습 완료! 제일 성적이 좋았던 검증(Val) 정확도: {best_acc:.4f}")
    return model

def main():
    print("=== 비교 실험: 가중치 도움 없이 밑바닥부터 배우는 개미/벌 분류 ===")
    
    # [1] 데이터 전처리 (단순 리사이즈 및 텐서 변환)
    custom_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),         
    ])

    # [2] 데이터셋 로드
    data_dir = './data/hymenoptera_data'
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), custom_transform),
        'val':   datasets.ImageFolder(os.path.join(data_dir, 'val'),   custom_transform)
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=4, shuffle=True),
        'val':   DataLoader(image_datasets['val'],   batch_size=4, shuffle=False)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")
    
    # [3] 모델, 손실함수, 옵티마이저 초기화
    # 🚨 기존 가중치(Pre-trained) 없이 무작위 깡통 가중치부터 시작합니다!
    model = SimpleAntBeeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # [4] 모듈화된 학습/검증 함수 호출
    print("\n[훈련 및 검증 시작]")
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, epochs=10)

    print("\n💡 데이터가 고작 200장 남짓이므로 바닥부터 학습하면 60~70% 내외의 처참한 성적이 나옵니다.")
    print("이제 10번 스크립트에서 '천재의 뇌(ResNet18 가중치)'를 이식하면 어떻게 되는지 확인해 봅시다!")

if __name__ == '__main__':
    main()
