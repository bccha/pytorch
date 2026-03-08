import torch
import numpy as np

def tensor_example():
    # 1. 일반 리스트로부터 생성
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)

    # 2. NumPy 배열로부터 생성
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    # 3. 무작위 값 또는 상수 텐서 생성
    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(x_data)
    print(x_np)
    print(rand_tensor)
    print(ones_tensor)
    print(zeros_tensor)

    rand_tensor *= 100

    # 4. 텐서의 주요 속성 (Attributes)
    rand_tensor = rand_tensor.to(torch.int8)
    print(rand_tensor)
    print(f"Shape: {rand_tensor.shape}")
    print(f"Datatype: {rand_tensor.dtype}")
    print(f"Device: {rand_tensor.device}") # cpu 또는 cuda:0 등

def operation_example():
    tensor = torch.tensor([[i * 4 + j + 1 for j in range(4)] for i in range(4)])
    print(f"첫 행: {tensor[0]}")
    print(f"첫 열: {tensor[:, 0]}")
    print(f"마지막 열: {tensor[..., -1]}")
    print(f"마지막 행: {tensor[-1]}")

    print(tensor)

    # 특정 값 변경(in-place)
    tensor[:, 1] = 0

    x = torch.randn(4, 4)       # 4x4 행렬 (총 16개 요소)
    y = x.view(16)              # 1차원 벡터로 변환
    z = x.view(-1, 8)           # -1은 다른 차원을 보고 알아서 추론 (여기선 2x8이 됨)

    # 1차원 방향(가로)으로 이어 붙이기
    t1 = torch.cat([tensor, tensor, tensor], dim=1)

    # 4x4 텐서 3개를 하나로 합쳐서 (3, 4, 4) 텐서로 만듦
    t2 = torch.stack([tensor, tensor, tensor])

    agg = tensor.sum()       # 모든 원소의 합계 텐서 (크기 1)
    agg_item = agg.item()    # 파이썬 숫자로 변환
    print(agg_item, type(agg_item))

if __name__ == "__main__":
    print("=== 1. Tensor Basics ===")
    tensor_example()
    print("\n=== 2. Tensor Operations ===")
    operation_example()
