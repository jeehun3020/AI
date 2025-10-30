# 🧩 Generative Adversarial Networks (GAN, cGAN, DCGAN)

본 프로젝트는 Fashion-MNIST 데이터셋을 기반으로  
GAN, cGAN, DCGAN 세 가지 모델을 구현하고,  
학습 결과 및 손실 그래프를 비교 분석한  코드입니다.

---

## 📂 프로젝트 구성
AI/
├── GAN/                     # 본 브랜치
│    ├── gan.py              # 기본 MLP 기반 GAN
│    ├── cgan.py             # Conditional GAN (label conditioning)
│    ├── dcgan.py            # Deep Convolutional GAN
│    ├── images/             # 학습 중 생성된 이미지 저장
│    ├── loss_graph/         # 학습 손실 그래프 저장
│    └── saved_model/        # 학습된 모델(.h5) 저장
└── README.md

---

## ⚙️ 개발 환경

- **Framework**: TensorFlow 2.15.0 (macOS Metal backend)
- **Language**: Python 3.9+
- **Dataset**: Fashion-MNIST
- **GPU**: Apple Silicon (M1/M2)  
- **IDE**: PyCharm / Jupyter Notebook

---

## 🧩 모델 설명

### 1️⃣ GAN (Vanilla GAN)
- 완전연결층(Dense Layer)으로 구성된 기본 구조  
- 생성자(Generator)와 판별자(Discriminator)를 각각 MLP로 설계  
- 입력: `latent vector (100)`  
- 출력: `(28×28×1)` Fashion-MNIST 이미지  
- Loss: Binary Crossentropy  
- Optimizer: Adam(learning rate=0.0002, β1=0.5)

---

### 2️⃣ cGAN (Conditional GAN)
- 기존 GAN에 **클래스 레이블(0~9)** 조건을 추가  
- Generator, Discriminator 모두 `label embedding` 사용  
- Generator 입력: `[noise, label]`  
- Discriminator 입력: `[image, label]`  
- 클래스별로 제어된 이미지 생성 가능  
- 결과적으로 더 구체적이고 안정된 이미지 품질 확보

---

### 3️⃣ DCGAN (Deep Convolutional GAN)
- Dense 기반 구조를 CNN 기반으로 변경  
- Generator: Conv2DTranspose(UpSampling2D) 구조 사용  
- Discriminator: Conv2D 기반의 특징 추출 구조  
- 학습 안정성 향상 및 시각적 품질 개선  
- Conv-BatchNorm-LeakyReLU 반복 구조로 구현

---

## 📊 학습 결과

| Model | Epoch | 이미지 예시 | 손실 그래프 |
|:------|:------:|:-------------:|:-------------:|
| GAN | 5000 | ![GAN result](images/gan_5000.png) | ![GAN loss](loss_graph/gan_loss_epoch_5000.png) |
| cGAN | 5000 | ![cGAN result](images/cgan_5000.png) | ![cGAN loss](loss_graph/cgan_loss_epoch_5000.png) |
| DCGAN | 5000 | ![DCGAN result](images/dcgan_5000.png) | ![DCGAN loss](loss_graph/dcgan_loss_epoch_5000.png) |

---

## 📈 분석 및 평가

- GAN
  - 단순한 구조로 인해 이미지가 흐릿하고 잡음이 많음.  
  - 판별자와 생성자의 균형 잡힌 학습이 어렵고, mode collapse 발생 가능.

- cGAN
  - 클래스 조건(label)을 추가함으로써 명확한 형태의 이미지 생성 가능.  
  - 특정 클래스별 특성이 잘 반영되어 분별력 있는 출력 결과를 보임.

- DCGAN 
  - CNN 기반 구조로 시각적 품질이 가장 우수.  
  - BatchNorm과 Conv 층 조합으로 학습 안정화.  
  - 손실 그래프에서 진동이 줄고 수렴 패턴이 명확하게 나타남.

---

## 💾 실행 방법

```bash
# 1. 가상환경 생성 및 패키지 설치
pip install tensorflow-macos tensorflow-metal matplotlib numpy

# 2. 모델 학습 실행 (예: DCGAN)
python dcgan.py

# 3. 생성된 이미지 확인
open images/new_5000.png
