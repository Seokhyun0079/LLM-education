import torch
import torch.nn as nn
from casual_attention import CasualAttention

class MultiHeadAttentionWrapper(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    self.heads = nn.ModuleList(
      [CasualAttention(d_in, d_out, context_length, dropout, qkv_bias)for _ in range(num_heads)]
    )
    self.num_heads = num_heads
    self.d_out = d_out
    
  def forward(self, x):
    return torch.cat([head(x) for head in self.heads], dim=-1)


# 테스트
torch.manual_seed(123)

# 설정값 지정 (d_in=3, d_out=2는 고정)
d_in = 3        # 입력 차원 3 (고정)
d_out = 2        # 각 헤드의 출력 차원 2 (고정)
num_heads = 2    # 헤드 수 2
context_length = 6  # 컨텍스트 길이 6

# 멀티헤드 어텐션 모델 생성
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.1, num_heads=num_heads, qkv_bias=False)

# 입력 텐서 생성
batch_size = 2  # 배치 크기 2
batch = torch.randn(batch_size, context_length, d_in)  # 형태: [2, 6, 3]

# Forward 패스
contextt_vecs = mha(batch)  # 출력: [2, 6, 4]

# 출력 shape 분석:
# [2, 6, 4] 형태는 다음과 같이 계산됩니다:
# - 첫 번째 차원 (2): 배치 크기(batch_size)에서 유래, 입력 배치의 샘플 수
# - 두 번째 차원 (6): 컨텍스트 길이(context_length)에서 유래, 시퀀스의 길이
# - 세 번째 차원 (4): 각 헤드의 출력 차원(d_out=2) * 헤드 수(num_heads=2) = 4
#
# 구체적인 계산 과정:
# 1. 각 어텐션 헤드는 입력 [2, 6, 3]에서 [2, 6, 2] 형태의 출력을 생성
#    - CasualAttention 클래스 내부 동작:
#      a) 입력 텐서 [2, 6, 3]에서 Query, Key, Value 행렬을 생성
#         - 각 행렬은 선형 변환(Linear layer)을 통해 생성됨
#         - Query: [2, 6, 3] → Linear(3, 2) → [2, 6, 2]
#         - Key: [2, 6, 3] → Linear(3, 2) → [2, 6, 2]
#         - Value: [2, 6, 3] → Linear(3, 2) → [2, 6, 2]
#      b) Query와 Key의 내적 계산: [2, 6, 2] @ [2, 2, 6] → [2, 6, 6] (어텐션 스코어)
#      c) 소프트맥스 적용하여 어텐션 가중치 계산: softmax([2, 6, 6]) → [2, 6, 6]
#      d) 어텐션 가중치와 Value의 행렬 곱: [2, 6, 6] @ [2, 6, 2] → [2, 6, 2]
#      e) 최종적으로 각 헤드는 [2, 6, 2] 형태의 출력을 생성
#
# 2. forward 메서드에서 torch.cat([head(x) for head in self.heads], dim=-1)를 통해
#    마지막 차원을 따라 각 헤드의 출력을 연결하므로 
#    [2, 6, 2] + [2, 6, 2] = [2, 6, 4] 형태가 됨

print(contextt_vecs)

print(batch.shape)       # 출력: torch.Size([2, 6, 3])
print(contextt_vecs.shape)  # 출력: torch.Size([2, 6, 4])