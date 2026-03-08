import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 09_3. 순정 nn.Conv2d vs 핵심 가속 im2col 완벽 비교
# 동일한 가중치(Seed)를 주고 수학적 결괏값과 속도를 비교합니다.
# ==========================================

# 1. 시드 고정 (동일한 난수 발생 보장)
torch.manual_seed(42)

# --- A. 순정 PyTorch Conv2d ---
class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(StandardConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


# --- B. 우리가 직접 짠 im2col (F.unfold) + matmul ---
class Im2colConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(Im2colConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.out_channels = out_channels
        
        # 가중치 행렬 선언
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels * kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        out_height = height + 2 * self.padding - self.kernel_size + 1
        out_width = width + 2 * self.padding - self.kernel_size + 1
        
        # 1. 차원 펴기 (im2col)
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        # 2. 거대 행렬 곱셈 1방울!
        out_unfold = self.weight.matmul(x_unfold) + self.bias.view(1, -1, 1)
        # 3. 다시 이미지로 접기
        return out_unfold.view(batch_size, self.out_channels, out_height, out_width)


def main():
    print("=== PyTorch nn.Conv2d VS Custom im2col GEMM 비교 실증 ===")
    
    # [1] 테스트 데이터/조건 생성 (128장 배치, 224x224 RGB 이미지)
    batch_size = 128
    in_c = 3
    out_c = 64
    k_size = 3
    
    print(f"테스트 환경: 배치 {batch_size}장, 입력채널 {in_c}, 출력채널 {out_c}, 필터크기 {k_size}x{k_size}")
    
    # 동일한 입력 데이터 생성
    x = torch.randn(batch_size, in_c, 224, 224)
    
    # [2] 두 모델 인스턴스화
    net_standard = StandardConv(in_c, out_c, k_size)
    net_im2col = Im2colConv(in_c, out_c, k_size)
    
    # 🚨 [가장 중요] 두 모델의 가중치를 수학적으로 완전히 100% 동일하게 복사!
    # standard Conv의 파라미터를 강제로 im2col 형태로 펴서 덮어씌웁니다.
    with torch.no_grad():
        net_im2col.weight.copy_(net_standard.conv.weight.view(out_c, -1))
        net_im2col.bias.copy_(net_standard.conv.bias)

    # [3] 속도 측정 준비
    # 경고: CPU 환경에서는 F.unfold 메모리 복사비용 때문에 오히려 느릴 수 있습니다.
    # GPU(CUDA) 환경이 병렬 행렬 곱셈의 진짜 압도적 힘을 보여줍니다.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=> 벤치마크 실행 기기: {device}")
    
    net_standard.to(device)
    net_im2col.to(device)
    x = x.to(device)

    # (A) Standard Conv2d 측정
    print("\n[ 1. 기본 nn.Conv2d 실행... ]")
    start_time = time.time()
    out_standard = net_standard(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    time_standard = time.time() - start_time
    print(f"-> 소요 시간: {time_standard:.4f}초")

    # (B) Im2col Custom 측정
    print("\n[ 2. Custom im2col + Matmul 행렬곱 연산 실행... ]")
    start_time = time.time()
    out_im2col = net_im2col(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    time_im2col = time.time() - start_time
    print(f"-> 소요 시간: {time_im2col:.4f}초")

    # [4] 검증 (수학적 결과값이 동일한지 확인)
    # 텐서 간의 차이 절댓값의 최댓값이 1e-5.0 근처의 오차(실수 연산 한계)라면 완전히 동일한 연산임.
    diff = torch.max(torch.abs(out_standard - out_im2col)).item()
    print("\n[ 결론 및 평가 ]")
    print(f"두 방식의 결과물 최대 오차 (차이값): {diff:.8e}")
    
    if diff < 1e-4:
        print("✅ 검증 성공: nn.Conv2d는 내부적으로 im2col(행렬 곱셈)과 완벽하게 동일한 수학적 결과값을 반환합니다!")
    else:
        print("❌ 실패: 결괏값이 다릅니다.")
        
    print("\n💡 성능 해석:")
    print("- CUDA(GPU) 환경이라면 nn.Conv2d는 Nvidia의 극도로 최적화된 cuDNN 라이브러리를 통하므로 가장 빠릅니다.")
    print("- 하지만 그 cuDNN의 내부 깊숙한 곳에서 동작하는 하드웨어 가속의 핵심 C++ 코드가 바로 우리가 방금 Python으로 시뮬레이션한 'im2col + 행렬곱(GEMM)' 아키텍처입니다.")

if __name__ == '__main__':
    main()
