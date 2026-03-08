import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 1개의 입력 이미지 채널, 6개의 출력 채널, 5x5 Conv 커널
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 선형 연산 (y = Wx + b)
        self.fc1 = nn.Linear(6 * 10 * 10, 10)

    def forward(self, x):
        # Convolution -> ReLU -> Max Pooling 적용
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = torch.flatten(x, 1) # 미니배치 차원(0)은 유지하고 나머지 공간 차원을 평탄화
        x = self.fc1(x)
        return x

def neural_net_example():
    net = SimpleNet()
    print("모델 구조:\n", net)

    # 모델의 입력 텐서 생성 (배치 크기=1, 채널=1, 높이=24, 너비=24)
    # PyTorch의 입력 데이터는 기본적으로 (Batch, Channel, Height, Width) 형태를 가집니다.
    input_tensor = torch.randn(1, 1, 24, 24)
    input_tensor.requires_grad = True
    print("\n입력 텐서 모양:", input_tensor.shape)

    # 네트워크에 입력을 통과시킵니다 (순전파)
    # 특별한 함수 호출 없이 net(input_tensor) 처럼 객체 자체를 함수처럼 호출하면 forward()가 실행됩니다.
    out = net(input_tensor)
    
    print("\n출력 텐서 결과:\n", out)
    print("출력 텐서 모양:", out.shape) # 결과: 배치크기 1, 출력 노드수 10
    print("\n소프트맥스 적용 결과:\n", torch.softmax(out, dim=1))
    print("\n가장 큰 값의 인덱스:\n", torch.argmax(out, dim=1))

    loss = F.cross_entropy(out, torch.tensor([8]))  
    print("\n손실 값:\n", loss)
    loss.backward()
    print("\n입력 텐서의 기울기:\n", input_tensor.grad) 

if __name__ == "__main__":
    print("=== Neural Network Example ===")
    neural_net_example()
