import os
import urllib.request
import zipfile
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def download_hymenoptera_data(data_dir='./data'):
    """PyTorch 공식 튜토리얼용 개미/벌(hymenoptera) 실제 사진 데이터셋을 다운로드합니다."""
    url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
    zip_path = os.path.join(data_dir, 'hymenoptera_data.zip')
    extract_dir = os.path.join(data_dir, 'hymenoptera_data')
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(extract_dir):
        print(f"[1] 인터넷에서 '개미와 벌' 실제 사진 데이터를 다운로드합니다... (약 45MB)")
        urllib.request.urlretrieve(url, zip_path)
        print("    다운로드 완료! 압축을 해제합니다...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"    압축 해제 완료! 경로: {extract_dir}\n")
    else:
        print(f"[1] 이미 데이터가 존재합니다. 경로: {extract_dir}\n")
        
    return extract_dir

def main():
    print("=== 내 사진 폴더(Custom Image)를 Dataset으로 가져오기 ===\n")
    
    # 0. 실제 개미(ants) vs 벌(bees) 데이터셋 다운로드
    base_dir = download_hymenoptera_data()
    
    # 데이터는 이미 train과 val(테스트) 폴더 구조로 나뉘어 있습니다.
    # 구조: hymenoptera_data/
    #           ├── train/
    #           │      ├── ants/
    #           │      └── bees/
    #           └── val/
    train_dir = os.path.join(base_dir, 'train')

    # 1. 이미지 전처리 (Transform) 파이프라인 만들기
    # 스마트폰 복셀, DSLR 모두 크기가 다르므로 모델 입력 크기(224x224)로 강제 통일합니다.
    custom_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 모든 사진을 가로세로 224x224 로 리사이즈 
        transforms.ToTensor(),         # 이미지 픽셀(0~255)을 파이토치 텐서(0.0~1.0)로 변환
    ])

    # 2. ImageFolder 마법의 모듈 사용하기
    # 폴더 구조가 [경로]/[클래스명]/[사진들] 형태라면 폴더명으로 알아서 정답 라벨링을 해줍니다!
    print(f"[2] ImageFolder 로 '{train_dir}' 경로를 읽어옵니다.")
    train_dataset = ImageFolder(root=train_dir, transform=custom_transform)
    
    print(f" - 총 이미지 개수: {len(train_dataset)}장")
    print(f" - 폴더명이 알아서 정답 클래스가 됩니다! -> {train_dataset.classes}")
    print(f" - 클래스-인덱스 매핑: {train_dataset.class_to_idx}\n")

    # 3. 배달 트럭(DataLoader)에 태우기
    # batch_size=4 니까, 트럭 1대에 사진 4장씩 실어서 배달합니다.
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    
    print("[3] DataLoader에서 직접 데이터를 꺼내봅니다 (Batch Size=4)")
    # DataLoader를 반복문(for)에 넣어서 첫 번째 트럭(배치)만 딱 꺼내봅니다.
    for batch_images, batch_labels in train_loader:
        print(f" - 이미지 텐서 모양(Shape): {batch_images.shape}") # [배치 4장, 채널 3(RGB), 높이 224, 너비 224]
        print(f" - 이번 배치의 정답 라벨들: {batch_labels} (0: 개미, 1: 벌)")
        
        # 첫 번째 배치를 확인했으니 바로 반복문을 탈출합니다.
        break

if __name__ == '__main__':
    main()
