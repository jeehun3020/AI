# 🎨 Gradio Video/Semantic Segmentation Demo  
> 실전 학습용 프로젝트 | 포트폴리오용 영상 분할 인터페이스 구현  

## 🔗 데모 링크  
- [Segmentation 데모 – Gradio Space](https://huggingface.co/spaces/jirtor/gradio)  
- [Fine-Tune 데모 – Gradio Space](https://huggingface.co/spaces/jirtor/finetune)  

## 📌 프로젝트 개요  
이 프로젝트는 사전 학습된 영상 분할 모델을 활용해 **웹 기반 데모 인터페이스**를 구현함으로써,  
단순히 알고리즘을 실행하는 수준을 넘어 **사용자 경험(UI/UX)까지 고려한 실습형 애플리케이션** 제작 역량을 키우고자 한 결과물이다.  
특히 SegFormer 모델을 활용해 픽셀 단위 객체 분할 기능을 제공하며, 배포 플랫폼으로는 Hugging Face Spaces + Gradio를 채택했다.

## 🛠 주요 기능 및 특징  
- 업로드한 영상 또는 이미지에 대해 실시간으로 픽셀 단위 분할(mask) 결과 표시  
- **Overlay Transparency 슬라이더**를 통해 원본 이미지와 분할 결과의 투명도 조정 가능  
- 감지된 클래스 목록을 시각화하여 어떤 객체가 분할됐는지 직관적으로 확인 가능  
- 깔끔하고 일관된 UI 디자인으로 사용자 친화적인 인터페이스 구현  
- 폴더 구조: `Segmentation/`, `finetune/`, `gradio/`로 기능별 코드 및 자료 정리  

## 🧑‍💻 기술 스택  
- **프로그래밍 언어**: Python 3.x  
- **모델/라이브러리**: Transformers (Hugging Face), PyTorch  
- **모델명**: `nvidia/segformer-b0-finetuned-ade-512-512`  
- **배포·UI**: Gradio, Hugging Face Spaces  
- **이미지 처리**: PIL, NumPy, Matplotlib  

## 🚀 설치 및 실행 방법  
```bash
git clone https://github.com/jeehun3020/AI.git
cd AI/gradio/Segmentation
pip install -r requirements.txt
python app.py

📁 폴더 구조
AI/
└─ gradio/
   ├─ Segmentation/     # Gradio 데모 코드 및 이미지 예제
   ├─ finetune/         # 모델 미세조정 실습 코드
   └─ gradio/           # 인터페이스 구성 및 배포 관련 코드
