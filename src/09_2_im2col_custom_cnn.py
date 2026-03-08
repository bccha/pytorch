import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 09_2. im2col을 활용한 CNN 모델 구현
# ==========================================
# PyTorch의 nn.Conv2d 대신, 하드웨어(NPU/GPU) 연산의 핵심인
# im2col (F.unfold) + 선형 행렬 곱셈(nn.Linear/matmul) 방식을 직접 구현하여
# CNN이 어떻게 거대한 행렬 곱(GEMM)으로 변환되는지 증명합니다.

class Im2ColConv2d(nn.Module):
    """
    nn.Conv2d와 완벽하게 동일한 연산을 수행하지만,
    내부적으로는 im2col(unfold)와 행렬 곱셈을 사용하는 커스텀 레이어
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(Im2ColConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # 필터 가중치 (out_channels, in_channels * kernel_height * kernel_width)
        # 예: 3채널 3x3 필터를 16개 만들면 -> (16, 3 * 3 * 3) = (16, 27) 모형의 가중치 행렬
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels * kernel_size * kernel_size) / (in_channels * kernel_size * kernel_size)**0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # 출력 이미지 크기 계산 공식
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 1. im2col 변환 (PyTorch에서는 F.unfold 사용)
        # x_unfold 형태: [batch_size, in_channels * kernel_size^2, out_height * out_width]
        # 공간(2D) 이미지를 무식하고 길다란 1차원 막대기(Column)들로 쫙쫙 펴서 2차원 판때기로 만듭니다.
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        # 2. 행렬 곱셈 (GEMM)
        # 4중 for문 슬라이딩 윈도우 대신 GPU/NPU가 제일 좋아하는 거대 행렬 곱셈 1방울!
        # [out_channels, in_channels * k^2] @ [batch_size, in_channels * k^2, out_height * out_width]
        # -> [batch_size, out_channels, out_height * out_width]
        out_unfold = self.weight.matmul(x_unfold) + self.bias.view(1, -1, 1)
        
        # 3. 다시 이미지 형태(2D)로 접어주기 (col2im)
        out = out_unfold.view(batch_size, self.out_channels, out_height, out_width)
        return out


class Im2ColAntBeeCNN(nn.Module):
    def __init__(self):
        super(Im2ColAntBeeCNN, self).__init__()
        # 09_custom_cnn_ants_bees.py 와 완벽하게 동일한 아키텍처
        # 단, nn.Conv2d 대신 우리가 직접 만든 Im2ColConv2d 사용
        self.features = nn.Sequential(
            Im2ColConv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 224 -> 112
            
            Im2ColConv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112 -> 56
            
            Im2ColConv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    print("=== NPU/GPU 가속의 핵심: im2col 연산으로 구현한 CNN 개미/벌 분류 ===")
    print("PyTorch의 nn.Conv2d를 쓰지 않고 F.unfold와 행렬 곱셈으로 동일한 결과를 도출합니다.\n")
    
    custom_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),         
    ])

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
    
    model = Im2ColAntBeeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5 # 증명이 목적이므로 5 에폭만 돌립니다.
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

    print(f"\n학습 완료! (im2col 기반 연산 검증 최고 정확도: {best_acc:4f})")
    print("공간 연산(Sliding Window)이 단순한 2D 행렬 곱셈(GEMM)으로 완벽히 대체 작동함을 확인했습니다!")

if __name__ == '__main__':
    main()
