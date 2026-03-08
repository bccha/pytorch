import torch
from torch.utils.data import Dataset, DataLoader

# 1. 나만의 데이터셋(Dataset) 클래스 만들기
# PyTorch의 Dataset을 상속받아야 하며, 필수적으로 거쳐야 할 3가지 매직 메서드가 있습니다.
class MyCustomDataset(Dataset):
    def __init__(self, data_size=100):
        # [필수 1] 전처리나 데이터를 불러오는(Load) 초기화 작업을 하는 곳
        # 여기서는 임의의 1D 숫자 배열과 그에 대한 짝수/홀수 정답 라벨을 만듭니다.
        self.x_data = torch.randn(data_size, 5) # 100행 5열의 임의 데이터
        # 첫 번째 열의 값이 양수면 1, 음수면 0으로 진짜 정답(Label) 생성
        self.y_label = (self.x_data[:, 0] > 0).to(torch.long) 
        
    def __len__(self):
        # [필수 2] 이 데이터셋의 총 데이터 개수를 반환하는 곳
        return len(self.x_data)
    
    def __getitem__(self, idx):
        # [필수 3] 인덱스(idx)를 입력받아 그 순서에 맞는 데이터 1개를 반환하는 곳
        # DataLoader가 이 함수를 반복해서 호출해 데이터를 긁어갑니다.
        x = self.x_data[idx]
        y = self.y_label[idx]
        return x, y

def dataloader_example():
    print("\n--- 파이토치 Dataset과 DataLoader 실습 ---")
    
    # 1. 우리가 정의한 데이터셋 객체 생성
    my_dataset = MyCustomDataset(data_size=105)
    print(f"총 데이터 개수: {len(my_dataset)}개")
    
    # 데이터셋에서 0번째 데이터 하나만 뽑아보기
    sample_x, sample_y = my_dataset[0] # __getitem__(0) 이 내부적으로 호출됨
    print(f"0번째 샘플\n - 데이터: {sample_x}\n - 정답: {sample_y}")

    # 2. DataLoader 객체 생성
    # Dataset이 '데이터를 1개씩 꺼내주는 기계'라면, 
    # DataLoader는 '그 기계를 돌려서 원하는 배치(묶음) 크기로 잘라서 섞어주는(Shuffle) 트럭'입니다.
    batch_size = 10
    my_loader = DataLoader(
        dataset=my_dataset, 
        batch_size=batch_size, 
        shuffle=True,       # 에폭마다 데이터를 섞을지 여부 (학습 시 필수!)
        drop_last=False     # 마지막에 10개짜리 묶음이 안 떨어져도(나머지 5개) 그대로 쓸 것인가 
    )

    # 3. DataLoader를 순회하며 묶음(Batch) 확인하기
    for batch_idx, (batch_x, batch_y) in enumerate(my_loader):
        print(f"\n[배치 {batch_idx+1}]")
        print(f"X 데이터 형태: {batch_x.shape}")
        print(f"Y 라벨 형태: {batch_y.shape}")
        print(f"Y 라벨 값: {batch_y}")
        
        # 실제 학습에서는 이렇게 DataLoader로 받은 batch_x, batch_y를 
        # 아까 만든 5단계 학습 루프의 1단계(inputs, labels)에 그대로 집어넣으면 됩니다!
        if batch_idx == 2: # 결과가 너무 길어지니 3묶음만 보고 탈출
            print("... (이하 생략) ...")
            break

if __name__ == "__main__":
    dataloader_example()
