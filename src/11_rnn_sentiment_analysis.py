import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. 초소형 커스텀 데이터셋 준비 (영화 리뷰)
# ==========================================
# RNN은 '순서'가 중요하므로, 문장을 단어 단위로 쪼개서 넣습니다.
# 실무에서는 nn.Embedding과 거대한 단어장(Vocab)을 쓰지만,
# 원리 이해를 위해 한 문장당 정해진 길이(시퀀스)를 가진 초소형 데이터를 만듭니다.

# 단어장(Vocabulary) 번호 매기기 (예시)
# 0:<PAD>(빈칸 채우기), 1:이, 2:영화, 3:정말, 4:최고, 5:추천, 6:최악, 7:돈, 8:아까워, 9:시간, 10:낭비
vocab_size = 11

# 문장 데이터 (각 숫자는 위 단어장의 번호를 의미함)
# 모든 문장의 길이를 똑같이(예: 4단어) 맞춰야 GPU가 병렬 처리를 할 수 있습니다. (Padding)
x_data = [
    [1, 2, 3, 4],    # "이 영화 정말 최고" (긍정)
    [1, 2, 5, 0],    # "이 영화 추천 <PAD>" (긍정)
    [1, 2, 3, 6],    # "이 영화 정말 최악" (부정)
    [7, 8, 0, 0],    # "돈 아까워 <PAD> <PAD>" (부정)
    [3, 9, 10, 0]    # "정말 시간 낭비 <PAD>" (부정)
]

# 정답(Label): 1은 긍정(Positive), 0은 부정(Negative)
y_data = [1, 1, 0, 0, 0]

# PyTorch 텐서로 변환
inputs = torch.tensor(x_data, dtype=torch.long)
labels = torch.tensor(y_data, dtype=torch.float32) # 손실함수를 위해 float으로 변환

# ==========================================
# 2. RNN 모델 구조 정의
# ==========================================
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(SimpleRNNClassifier, self).__init__()
        
        # 1. 임베딩 층 (Embedding Layer)
        # 단어 번호(예: 4번 '최고')를 아무 의미 없는 정수가 아니라, 
        # 컴퓨터가 연산하기 좋은 의미를 가진 '밀집 벡터 크기(embedding_dim)'로 변환합니다.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # 2. RNN 층 (LSTM, GRU 등도 있지만 기본 RNN 사용)
        # input_size: 임베딩된 단어 벡터의 크기
        # hidden_size: RNN이 문맥을 누적해서 기억할 메모리 공간의 크기 (메모장 크기)
        # batch_first=True: 입력 데이터의 첫 번째 차원이 '배치(Batch)' 개수임을 명시
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        
        # 3. 분류기 층 (Classifier)
        # RNN이 모든 문장을 다 읽고 나서 최종적으로 작성완료한 메모장(hidden)을 보고
        # "긍정 확률(1개 숫자)"을 예측하는 단일 출력 뉴런입니다.
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid() # 결과값을 0.0 ~ 1.0 (0~100%) 비율로 압축

    def forward(self, x):
        # x 형태: [배치 수, 문장 길이] = [5, 4]
        
        # 1. 단어 숫자를 벡터로 변환
        # embedded 형태: [배치 수, 문장 길이, 임베딩 차원]
        embedded = self.embedding(x) 
        
        # 2. RNN에 순서대로 밀어 넣기
        # rnn_out: 모든 시점(단어)마다의 메모장 기록
        # hidden: 문장을 전부 다 읽고 난 뒤의 **최종 문맥 메모장** (이게 핵심입니다!)
        rnn_out, hidden = self.rnn(embedded)
        
        # hidden의 형태는 [층 개수=1, 배치 수, 메모장 크기] 이므로 
        # 분류기에 넣기 위해 [배치 수, 메모장 크기]로 차원을 축소(-1)합니다.
        final_context = hidden.squeeze(0) 
        
        # 3. 최종 결론 예측
        out = self.fc(final_context)
        out = self.sigmoid(out)
        
        # 출력 형태 차원을 [배치 수] 로 쫙 펴줍니다.
        return out.squeeze() 

# ==========================================
# 3. 훈련 루프 (Training)
# ==========================================
def main():
    print("=== 11. RNN을 이용한 영화 리뷰 감성(긍정/부정) 분류 기초 실습 ===\n")
    
    # 하이퍼파라미터 설정
    embedding_dim = 8  # 단어 1개를 8개의 숫자로 표현 (예: [0.1, -0.4, ...])
    hidden_size = 16   # RNN 메모장의 용량
    
    model = SimpleRNNClassifier(vocab_size, embedding_dim, hidden_size)
    print("생성된 RNN 모델 뼈대:\n", model)
    
    # 이진 분류(0 아니면 1) 문제이므로 BCELoss(Binary Cross Entropy)를 사용합니다.
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 100 # 데이터가 극도로 적으므로 에폭을 많이 돌립니다.
    
    print("\n[ 훈련 시작 ]")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 모델 순전파 (RNN이 단어를 순서대로 꿀꺽꿀꺽 삼키며 최종 예측 반환)
        outputs = model(inputs)
        
        # 오차 계산 및 역전파
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    print("\n[ 훈련 단어장 ]")
    print("1:이, 2:영화, 3:정말, 4:최고, 5:추천, 6:최악, 7:돈, 8:아까워, 9:시간, 10:낭비\n")
            
    print("[ 예측 테스트 (Inference) ]")
    # 훈련된 모델 평가 모드
    model.eval()
    with torch.no_grad():
        test_preds = model(inputs)
        
        for i, sentence in enumerate(x_data):
            # 긍정 확률이 0.5(50%)를 넘으면 긍정(1), 아니면 부정(0)
            pred_label = "긍정 😍" if test_preds[i].item() > 0.5 else "부정 😡"
            prob = test_preds[i].item() * 100
            
            # 원래 숫자를 보기 좋게 텍스트로 변환 (Pad는 무시)
            print(f"문장 데이터 {sentence} \t=> 예측: {pred_label} (긍정 확률: {prob:.1f}%) | 정답: {'긍정' if y_data[i]==1 else '부정'}")

if __name__ == '__main__':
    main()
