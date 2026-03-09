import torch
import torch.nn as nn
from torchvision import models

# torchinfo 패키지 임포트 (모델 구조를 예쁜 트리 표 형태로 출력해줌)
from torchinfo import summary

def main():
    print("="*60)
    print("1. 원본 ResNet-18 (ImageNet 1000개 사물 분류용) 구조 출력")
    print("="*60)
    # weights 매개변수를 사용하여 최신 방식으로 사전 학습된 모델 로드
    model_original = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # summary(모델객체, 우리가 넣을 가짜 사진 1장의 차원 크기)
    # input_size = (배치 사이즈 1장, 컬러 RGB 3채널, 세로 224, 가로 224)
    summary(model_original, input_size=(1, 3, 224, 224))
    
    
    print("\n\n" + "="*60)
    print("2. 우리가 아까 개조(Transfer Learning)한 개미/벌 2개 분류용 모델")
    print("="*60)
    model_custom = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 마지막 fc 레이어(1000개 출력) 잘라내고, 2개짜리로 대체하기!
    num_features = model_custom.fc.in_features
    model_custom.fc = nn.Linear(num_features, 2)
    
    summary(model_custom, input_size=(1, 3, 224, 224))
    
    print("\n[관찰 포인트]")
    print("1. 트리 구조 맨 아래 'Linear' 노드의 Output Shape가 [1, 1000]에서 [1, 2]로 바뀐 것을 확인하세요!")
    print("2. Param # (파라미터 개수)가 513,000개에서 1,026개로 대폭 줄어든 것도 보입니다.")

if __name__ == '__main__':
    main()
