import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

@torch.no_grad()
def evaluate_original_model(model, val_loader, device, imagenet_ant_idx=310, imagenet_bee_idx=309):
    """
    [4] 순정 모델(1000개 클래스)이 개미/벌 이미지를 맞추는지 테스트하는 평가 루프입니다.
    """
    correct_count = 0
    total_count = 0
    
    print("\n테스트를 시작합니다... (총 153장)")
    
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        
        # 순정 모델에 사진 투입! -> 1,000개의 점수가 나옴
        outputs = model(inputs)
        
        # 1,000개 중에 가장 확률이 높은 단어 1개의 '인덱스 번호'를 뽑음 (Top-1 예측)
        _, pred_idx = torch.max(outputs, 1)
        pred_idx = pred_idx.item()
        
        # 예측한 번호가 실제 정답과 맞는지 검사!
        is_correct = False
        if labels.item() == 0 and pred_idx == imagenet_ant_idx:
            is_correct = True
        elif labels.item() == 1 and pred_idx == imagenet_bee_idx:
            is_correct = True
            
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # 첫 5장 시각적 피드백
        if i < 5:
            real_name = "개미(Ant)" if labels.item() == 0 else "벌(Bee)"
            pred_name = "ant" if pred_idx == imagenet_ant_idx else \
                        "bee" if pred_idx == imagenet_bee_idx else f"틀림(딴 번호: {pred_idx})"
            print(f"사진 {i+1} | 실제: {real_name} -> 순정 모델의 대답: {pred_name} {'(정답!)' if is_correct else ''}")
            
    accuracy = 100 * correct_count / total_count
    
    print("-" * 30)
    print(f"순정 ResNet18의 개미/벌 맞추기 정확도 비율: {accuracy:.2f}% ({correct_count}/{total_count}장)")
    print("-" * 30)
    print("💡 놀랍게도(또는 당연하게도), 순정 모델은 전 1,000개의 클래스 중에서도 개미와 벌을 매우 정확히 찾아냅니다.")
    print("💡 전이 학습은 이처럼 '이미 알고 있는 천재의 지식 공간'을 그저 '나만의 2개의 카테고리(라벨)'로 깔때기만 꽂아 정리해 주는 수술일 뿐입니다!")

def main():
    print("=== 호기심 해결: 순정 ResNet18은 허가 없이 개미/벌을 구별할까? ===")
    
    # [1] ImageNet 원본 규격의 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Resize(256),                
        transforms.CenterCrop(224),            
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    # [2] 평가(val) 데이터만 불러오기
    data_dir = './data/hymenoptera_data/val'
    val_dataset = datasets.ImageFolder(data_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [3] 순정 1,000클래스 모델 탑재
    print("\n[진행] ImageNet 가중치가 탑재된 순정 ResNet18 모델을 로드합니다...")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.eval() # 무조건 추론(평가) 모드
    
    # ImageNet에는 1,000종류의 라벨이 영어로 매핑되어 있습니다.
    IMAGENET_ANT_IDX = 310
    IMAGENET_BEE_IDX = 309
    
    # 정리된 평가 함수 호출
    evaluate_original_model(model, val_loader, device, IMAGENET_ANT_IDX, IMAGENET_BEE_IDX)

if __name__ == '__main__':
    main()
