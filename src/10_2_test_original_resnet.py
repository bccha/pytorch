import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

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
    # 우리가 앞서 10번 스크립트에서 구축했던 동일한 데이터 폴더입니다.
    data_dir = './data/hymenoptera_data/val'
    val_dataset = datasets.ImageFolder(data_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [3] 아무 개조도 하지 않은 "순정(Original)" 1,000클래스 모델 탑재
    print("\n[진행] ImageNet 가중치가 탑재된 순정 ResNet18 모델을 로드합니다...")
    # weights 매개변수에 미리 학습된 진짜 가중치를 끼워넣음
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.eval() # 무조건 추론(평가) 모드
    
    # ImageNet에는 1,000종류의 라벨이 영어로 매핑되어 있습니다.
    # 그 중 개미(ant)와 관련된 번호와, 벌(bee)와 관련된 번호를 우리가 알아내야 합니다!
    # ImageNet 1000개 클래스 인덱스 중에:
    # - 310번: ant, emmet, pismire (개미)
    # - 309번: bee (벌)
    IMAGENET_ANT_IDX = 310
    IMAGENET_BEE_IDX = 309
    
    correct_count = 0
    total_count = 0
    
    print("\n테스트를 시작합니다... (총 153장)")
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader): # batch_size=1 이므로 한 장씩
            inputs = inputs.to(device)
            # labels의 0번은 우리 폴더의 'ants', 1번은 'bees'
            
            # 순정 모델에 사진 투입! -> 1,000개의 클래스(영어 과목) 확률 점수가 나옴
            outputs = model(inputs)
            
            # 1,000개 중에 가장 확률이 높은 단어 1개의 '인덱스 번호'를 뽑음 (Top-1 예측)
            _, pred_idx = torch.max(outputs, 1)
            pred_idx = pred_idx.item()
            
            # 예측한 번호가 실제 정답과 맞는지 검사!
            # 실제 정답이 '개미(0)'일 때, 모델이 '310번(ant)'이라 외쳤으면 정답 처리!
            # 실제 정답이 '벌(1)'일 때, 모델이 '309번(bee)'이라 외쳤으면 정답 처리!
            is_correct = False
            if labels.item() == 0 and pred_idx == IMAGENET_ANT_IDX:
                is_correct = True
            elif labels.item() == 1 and pred_idx == IMAGENET_BEE_IDX:
                is_correct = True
                
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # 첫 5장만 모델이 뭐라고 입을 열었는지 직접 구경해봅시다.
            if i < 5:
                real_name = "개미(Ant)" if labels.item() == 0 else "벌(Bee)"
                pred_name = "ant" if pred_idx == IMAGENET_ANT_IDX else \
                            "bee" if pred_idx == IMAGENET_BEE_IDX else f"틀림(딴 번호: {pred_idx})"
                print(f"사진 {i+1} | 실제: {real_name} -> 순정 모델의 대답: {pred_name} {'(정답!)' if is_correct else ''}")
                
    accuracy = 100 * correct_count / total_count
    
    print("-" * 30)
    print(f"순정 ResNet18의 개미/벌 맞추기 정확도 비율: {accuracy:.2f}% ({correct_count}/{total_count}장)")
    print("-" * 30)
    print("💡 놀랍게도(또는 당연하게도), 순정 모델은 전 1,000개의 클래스 중에서도 개미와 벌을 매우 정확히 찾아냅니다.")
    print("💡 전이 학습은 이처럼 '이미 알고 있는 천재의 지식 공간'을 그저 '나만의 2개의 카테고리(라벨)'로 깔때기만 꽂아 정리해 주는 수술일 뿐입니다!")

if __name__ == '__main__':
    main()
