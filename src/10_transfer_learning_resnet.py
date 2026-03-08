import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

def main():
    print("=== 실무 전이 학습(Transfer Learning): 개미 vs 벌 분류 ===")
    
    # [1] 데이터 전처리 파이프라인 (Data Augmentation & Normalization)
    # 사전 학습된 천재 모델(ResNet)은 'ImageNet' 규칙에 맞춰진 입력 데이터를 원합니다.
    # 따라서 규격(224x224) 컷팅과 픽셀 정규화(Normalize)를 필수로 맞춰주어야 합니다!
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

    # [2] 이전(08번) 실습에서 받은 실제 개미/벌 데이터 디렉토리 로드!
    data_dir = './data/hymenoptera_data'
    
    # 앞서 배운 ImageFolder 마법을 이용해 Train, Val을 다 긁어옵니다.
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
    # torchvision에서 1000개 사물을 무려 정확도 90% 이상으로 맞추는 ResNet18을 가져옵니다.
    print("\n[진행] ResNet18 사전 학습 가중치 다운로드 및 모델 초기화 중...")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # ResNet18의 가장 마지막 분류기 층(Linear 레이어)의 입력 크기를 확인해 봅니다.
    num_features = model.fc.in_features 
    
    # 기존에는 1000개의 강아지/고양이/비행기 등으로 나가는 구조였지만, 
    # 우리는 오직 '개미(0)', '벌(1)' 2개로만 나뉘면 되므로 끝부분(머리)만 싹둑 자르고 새로 붙여줍니다!
    model.fc = nn.Linear(num_features, 2)
    
    model = model.to(device)

    # [4] 손실 함수 및 옵티마이저 구성
    criterion = nn.CrossEntropyLoss()
    
    # 🚨 중요: 사전 학습된 가중치가 망가지지 않도록, 우리가 새로 붙인 끝부분(fc)만 빠르게 학습시키거나,
    # 아니면 전체 모델을 아주 미세한 학습률(lr=0.001)로 아주 조심스럽게 미세 튜닝(Fine-Tuning)합니다.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # [5] 전이 학습 루프 (Training & Validation)
    epochs = 10
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 15)

        # 각 에폭(Epoch)마다 Train과 Val(검증) 단계를 번갈아가며 진행합니다!
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 훈련 스위치 ON (Dropout, BatchNorm 활성화)
            else:
                model.eval()   # 검증 스위치 OFF (모델 가중치 방어막)

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 배치(Batch=4) 단위로 뽑아오기
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 학습 상태일 때만 파이프라인 비우기
                optimizer.zero_grad()

                # 평가(val) 단계일 때는 그래디언트 역전파 엔진을 전원 차단!
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 가장 높은 확률의 번호(0:개미, 1:벌) 추출
                    loss = criterion(outputs, labels)

                    # 훈련(train) 단계일 때만 뒤로 미분해서 가중치 교정(업데이트)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계 수집
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # 훈련할 때마다 제일 똑똑해진 순간(Best)의 뇌세포(State Dict)를 잠깐 보관해둠
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_ant_bee_model.pth')

    print(f"\n학습 완료! 제일 성적이 좋았던 검증(Val) 정확도: {best_acc:4f}")
    print("가장 똑똑한 순간의 상태가 'best_ant_bee_model.pth' 에 저장되었습니다.")

if __name__ == '__main__':
    main()
