# PyTorch 기초 학습 프로젝트 (PyTorch Learning Tutorial)

이 저장소는 PyTorch를 처음 접하는 분들이 쉽고 체계적으로 학습할 수 있도록 단계별로 구성된 기초 튜토리얼 프로젝트입니다. 기초 텐서(Tensor) 조작부터 표준 5단계 학습 루프, 커스텀 데이터셋(Dataset) 작성법까지 딥러닝 실전에 필요한 핵심 핵심 개념들이 포함되어 있습니다.

## 📁 디렉토리 구조 (Repository Structure)

```text
pytorch/
├── README.md        # 이 파일: 프로젝트 개요 및 가이드
├── doc/             # [학습 문서] 핵심 개념, 설계, 로드맵, 트러블슈팅 정리
│   ├── STUDY.md     # ⭐️ 기초 개념/문법 요약 (Loss, Optimizer, Tensor, Autograd)
│   ├── DESIGN.md    # 프로젝트 설계 원칙
│   ├── ROADMAP.md   # PyTorch 학습 로드맵
│   └── ...
└── src/             # [실습 코드] 단계별로 하나씩 실행 가능한 파이썬 소스
    ├── 01_tensor_basics.py         # 텐서 기본 및 연산 실습
    ├── 02_autograd.py              # 자동 미분 및 연산 그래프 실습
    ├── 03_neural_network.py        # nn.Module 기반 모델 설계 실습
    ├── 04_training_loop.py         # Optimizer 선언 및 5단계 학습 루프 실습
    └── 05_dataset_dataloader.py    # Custom Dataset 제작 및 DataLoader 실습
```

## 📚 문서 가이드 (Documentation Links)
프로젝트 내의 목적별 문서를 아래 링크를 통해 바로 확인하실 수 있습니다.

* [📖 **BOOK.md**](doc/BOOK.md): (추천) 코드와 함께 읽는 O'Reilly 스타일 실전 튜토리얼 입문서
* [⭐️ **STUDY.md**](doc/STUDY.md): 딥러닝 필수 개념과 비유, 상세한 트러블슈팅 꿀팁 총망라 노트
* [🏗️ **DESIGN.md**](doc/DESIGN.md): 모델 아키텍처 구조 및 프로젝트 설계 원칙
* [🗺️ **ROADMAP.md**](doc/ROADMAP.md): PyTorch 학습 커리큘럼 및 향후 마스터 로드맵 
* [📊 **RESULT.md**](doc/RESULT.md): 신경망 (FC vs CNN) 성능 비교 및 시험 평가 결과 분석

## 🚀 학습 가이드 (How to Study)

본 프로젝트는 아래 순서대로 뼈대를 잡아나가시길 권장합니다.

1. **환경 설정**: 가상 환경(`venv`)을 켜고 `pip install torch`를 통해 PyTorch를 설치합니다.
2. **이론 읽기**: `doc/STUDY.md` 파일을 먼저 가볍게 읽어보시면서 텐서, Softmax, CrossEntropyLoss, DataLoader 등의 개념을 숙지합니다.
3. **코드 실습**: `src/` 폴더 안의 스크립트 파일을 `01`번부터 `05`번까지 차례대로 실행하며 동작 원리를 이해합니다.
   - 예시: `python src/01_tensor_basics.py`
4. **응용하기**: 각 파일 안에 있는 코드(숫자 텐서 차원, 노드 수 등)를 깨뜨려보고 나타나는 에러 메시지를 확인하는 방법을 통해 파이토치에 익숙해집니다.

## 💡 1부 핵심 학습 내용 총정리 (Part 1 Summary)

본 저장소의 1부(1장~11장) 스터디를 완주하시면 아래의 핵심 파이프라인을 완전히 마스터할 수 있습니다.

1. **기본 뼈대와 엔진 (1장 ~ 5장)**
   * **Tensor & Autograd**: 데이터를 담는 다차원 행렬(GPU)과, 손실(Loss) 방향을 역추적해주는 자동 미분 네비게이션
   * **학습 루프 5단계**: `초기화(Zero_grad) -> 순전파(Forward) -> 오차(Loss) -> 역전파(Backward) -> 업데이트(Step)`의 불변 패턴
2. **컴퓨터 비전과 전이 학습 (6장 ~ 10장)**
   * **CNN**: 돋보기 필터로 이미지의 공간적 특징(모서리, 질감 등)을 추출하는 '눈' 역할의 네트워크
   * **전이 학습(Transfer Learning)**: ResNet 등 거대 사전 학습 모델(Pre-trained)을 가져와 내 데이터셋에 맞게 머리(Classifier)만 뜯어고쳐 파인 튜닝(Fine-Tuning)하는 실무 기법
3. **순차 데이터와 자연어 처리 입문 (11장)**
   * **RNN**: 단순 주사위 놀이(N-gram)나 고정된 창문(Sliding Window)의 한계를 깨고, 자신의 이전 출력(Hidden State)을 다시 입력으로 넣는 **피드백(Feedback) 루프**를 통해 문맥과 시간의 흐름을 누적하는 모델

---
*Happy PyTorch Coding!* 😊
