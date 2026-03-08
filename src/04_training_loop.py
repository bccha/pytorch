import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.fc1 = nn.Linear(6 * 10 * 10, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

def training_loop_example():
    print("\n--- 5단계 표준 학습 루프 실습 ---")
    
    # 1. 모델, 손실 함수, 옵티마이저 준비
    net = SimpleNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01) # lr: 학습률
    
    # 모델 내 학습 가능한 파라미터의 '이름'과 '형태(Shape)' 확인하기
    print("--- 모델 파라미터 정보 ---")
    for name, param in net.named_parameters():
        print(f"이름: {name} | 형태: {param.shape}")
    print("--------------------------\n")
   
    # 2. 더미 입력 데이터와 정답 라벨 생성
    # 예: 배치 사이즈 4, 채널 1, 24x24 이미지 데이터
    inputs = torch.randn(4, 1, 24, 24)
    # 예: 4개 이미지의 각각의 정답 (0~9 사이 랜덤한 정답 라벨 부여)
    labels = torch.randint(0, 10, (4,)) 
    print(f"정답 타겟 (Labels): {labels}")

    # 3. 10 Epoch 학습 루프 진행
    for epoch in range(1, 11):
        # [Step 1] 미분값(Gradient) 초기화
        optimizer.zero_grad() 

        # [Step 2] 순전파 (Forward) 수행
        outputs = net(inputs)
        
        # [Step 3] 손실(Loss) 계산
        loss = loss_fn(outputs, labels)

        # [Step 4] 역전파 (Backward) -> 미분값 계산
        loss.backward()

        # [Step 5] 가중치 업데이트 (Step) 파라미터 갱신
        optimizer.step()

        # 학습이 진행될수록 Loss 값이 떨어지는지 확인
        print(f"Epoch [{epoch}/10], Loss: {loss.item():.4f}")
    
    # 학습 완료 후, 학습된 모델의 최종 예측 결과 (가장 확률이 높은 답)
    final_outputs = net(inputs)
    final_preds = torch.argmax(final_outputs, dim=1)
    print(f"최종 모델의 예측값: {final_preds} | 진짜 정답: {labels}")

if __name__ == "__main__":
    training_loop_example()
