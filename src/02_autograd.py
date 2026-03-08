import torch

def compute_graph():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()

    out.backward() # 역전파(Backpropagation) 수행
    print("순전파 최종 결과:", out)
    print("x의 기울기(x.grad):", x.grad)  # d(out)/dx의 결과가 출력됨

if __name__ == "__main__":
    print("=== Autograd & Compute Graph ===")
    compute_graph()
