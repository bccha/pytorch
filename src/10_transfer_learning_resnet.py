import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, epochs=10):
    """
    [5] 전이 학습 루프 (Training & Validation)
    """
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 15)

        # 각 에폭(Epoch)마다 Train과 Val(검증) 단계를 번갈아가며 진행합니다!
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 훈련 스위치 ON
            else:
                model.eval()   # 검증 스위치 OFF

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 배치 단위로 뽑아오기
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 학습 상태일 때만 파이프라인 비우기
                optimizer.zero_grad()

                # 평가(val) 단계일 때는 그래디언트 엔진 전원 차단!
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 가장 높은 확률의 번호
                    loss = criterion(outputs, labels)

                    # 훈련(train) 단계일 때만 가중치 교정(업데이트)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계 수집
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # 가장 똑똑해진 순간의 최고 기록 갱신 및 모델 복사
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_ant_bee_model.pth')

    print(f"\n학습 완료! 제일 성적이 좋았던 검증(Val) 정확도: {best_acc:.4f}")
    print("가장 똑똑한 순간의 상태가 'best_ant_bee_model.pth' 에 저장되었습니다.")
    return model

def main():
    print("=== 실무 전이 학습(Transfer Learning): 개미 vs 벌 분류 ===")
    
    # [1] 데이터 전처리 파이프라인 (Data Augmentation & Normalization)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),     # 무작위로 일부만 잘라서 확대 (데이터 뻥튀기 효과)
            transforms.RandomHorizontalFlip(),     # 무작위로 좌우 반전 시킴
            transforms.ToTensor(),                 # 0.0~1.0 사이 텐서 변환
            transforms.Normalize([0.485, 0.456, 0.406], # ResNet18이 훈련했던 RGB 평균값
                                 [0.229, 0.224, 0.225]) # RGB 표준편차
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),                # 약간 크게 늘린 후
            transforms.CenterCrop(224),            # 정중앙을 정확히 224x224로 도려냄
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # [2] 데이터셋 로드!
    data_dir = './data/hymenoptera_data'
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val':   datasets.ImageFolder(os.path.join(data_dir, 'val'),   data_transforms['val'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=4, shuffle=True),
        'val':   DataLoader(image_datasets['val'],   batch_size=4, shuffle=False)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"클래스 종류: {class_names}")
    print(f"훈련 데이터 개수: {dataset_sizes['train']}장, 검증 데이터 개수: {dataset_sizes['val']}장\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")

    # [3] 천재의 뇌(Pre-trained Model) 소환하기!
    print("\n[진행] ResNet18 사전 학습 가중치 다운로드 및 모델 초기화 중...")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    num_features = model.fc.in_features 
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    # [4] 손실 함수 및 옵티마이저 구성
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # [5] 모듈화된 전이 학습(Train & Val) 루프 호출
    print("\n[훈련 및 검증 시작]")
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, epochs=10)

if __name__ == '__main__':
    main()
