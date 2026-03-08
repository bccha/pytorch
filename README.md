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

## 🚀 학습 가이드 (How to Study)

본 프로젝트는 아래 순서대로 뼈대를 잡아나가시길 권장합니다.

1. **환경 설정**: 가상 환경(`venv`)을 켜고 `pip install torch`를 통해 PyTorch를 설치합니다.
2. **이론 읽기**: `doc/STUDY.md` 파일을 먼저 가볍게 읽어보시면서 텐서, Softmax, CrossEntropyLoss, DataLoader 등의 개념을 숙지합니다.
3. **코드 실습**: `src/` 폴더 안의 스크립트 파일을 `01`번부터 `05`번까지 차례대로 실행하며 동작 원리를 이해합니다.
   - 예시: `python src/01_tensor_basics.py`
4. **응용하기**: 각 파일 안에 있는 코드(숫자 텐서 차원, 노드 수 등)를 깨뜨려보고 나타나는 에러 메시지를 확인하는 방법을 통해 파이토치에 익숙해집니다.

## 💡 주요 학습 내용 요약

* PyTorch의 텐서는 `.shape`, `.dtype`, `.device` 속성을 가집니다.
* 분류 모델의 오차(Loss)를 계산할 때, 직접 만든 Output에 Softmax를 씌우지 않고 `nn.CrossEntropyLoss`에 통째로 넘기면 자동으로 계산해 줍니다!
* 모델 학습 시에는 파라미터를 업데이트(옵티마이저 `step`) 하기 전에 반드시 `zero_grad()`로 **이전 기울기를 초기화**해주어야 합니다.
* 큰 데이터를 쪼개서 학습시키기 위해 `Dataset`과 `DataLoader`가 필수적으로 사용됩니다. (자판기와 배달 트럭의 원리)

---
*Happy PyTorch Coding!* 😊
