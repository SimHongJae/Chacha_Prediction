# ChaCha4 딥러닝 암호분석

딥러닝으로 ChaCha (축소 라운드) CSPRNG의 다음 출력을 예측하는 실험.

## 구조

```
Chacha4 cryptanalysis/
├── my_chacha/src/main.rs    # ChaCha 데이터 생성기 (Rust)
├── dataset/                 # 생성된 바이너리 데이터
├── chacha_dataset.py        # 공통 데이터 로딩 유틸
├── train_mlp.py             # MLP baseline
├── train_lstm.py            # LSTM
└── train_transformer.py     # Transformer (GPT-style)
```

## 실험 방법

이전 N개의 u32 워드(비트 표현)를 보고 다음 u32의 32비트를 예측하는 next-token prediction 태스크.
각 비트를 독립적인 이진 분류로 처리하며, 랜덤 baseline 정확도는 50%.

## 요구사항

- **Rust** (데이터 생성)
- **Python 3.8+**
- PyTorch, NumPy, Matplotlib

## 사용법

### 1. 데이터 생성

```bash
cd my_chacha
cargo run --release -- <라운드수> <u32개수>
```

```bash
# ChaCha4, 400만 워드 (~16MB)
cargo run --release -- 4 4000000

# ChaCha6
cargo run --release -- 6 4000000
```

`dataset/chacha{라운드수}_seq.bin`에 u32 리틀엔디언 바이너리로 저장됨.

### 2. 모델 학습

```bash
# MLP baseline
python train_mlp.py

# LSTM
python train_lstm.py

# Transformer
python train_transformer.py
```

다른 라운드 수의 데이터를 사용할 경우:

```bash
python train_mlp.py --data dataset/chacha6_seq.bin --rounds 6
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data` | `dataset/chacha4_seq.bin` | 데이터 경로 |
| `--rounds` | 4 | 라운드 수 (라벨링용) |
| `--window` | 4 | 슬라이딩 윈도우 크기 |
| `--batch-size` | 1024 (MLP/LSTM), 512 (Transformer) | 배치 크기 |
| `--epochs` | 20 (MLP/LSTM), 30 (Transformer) | 에폭 수 |
| `--lr` | 1e-3 (MLP/LSTM), 1e-4 (Transformer) | 학습률 |

## 모델 요약

| 모델 | 입력 형태 | 구조 | 파라미터 |
|------|-----------|------|----------|
| MLP | (N×32,) flat | 128→256→256→128→32, ReLU | ~130K |
| LSTM | (N, 32) seq | 2-layer LSTM h=128 → FC 32 | ~200K |
| Transformer | (N, 32) seq | 4-layer, 4-head, d=128, d_ff=512 | ~800K |

## 평가 지표

- **Bit Accuracy**: 32비트 각각의 예측 정확도 (50% 초과 = 패턴 발견)
- **Exact Match**: 32비트 전부 맞춘 비율
- **Per-Bit Heatmap**: 비트 위치별 정확도 시각화

## 출력물

각 모델 학습 후 생성:
- `{모델}_chacha{라운드}_results.png` — Loss 곡선, Bit Accuracy 곡선, Per-Bit Heatmap
- `{모델}_chacha{라운드}.pt` — 모델 가중치
