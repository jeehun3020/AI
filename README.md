LoRA-Based Image Editing & Efficient Fine-Tuning Project

Diffusers + LoRA + UNet Partial Fine-Tuning 실험

본 프로젝트는 Stable Diffusion 기반 이미지 편집(image editing)에서
**Full Fine-tuning 대비 LoRA(Low-Rank Adaptation)**가
얼마나 효율적이고 효과적인지 비교·분석하는 것을 목표로 한다.

📌 1. Project Overview

본 프로젝트에서는 다음 3가지 AI 편집 기능을 평가함:
	1.	Instruction-based Image Editing
	•	예: “Change it to white”, “Turn him into a cyborg”, “Change it to night”
	2.	LoRA Rank-reduced Training (LFF, LPF)
	3.	Partial LoRA Fine-tuning (LPF)
	•	UNet 내 up / mid / down 블록 중 선택적 LoRA 적용
	•	[‘up’, ‘mid’]
	•	[‘mid’]
	•	[‘down’, ‘mid
	
📌 2. Image Editing Examples

2-1. Prompt-based Editing Examples

✨ Example 1 — Change to white

 <img width="516" height="1324" alt="image" src="https://github.com/user-attachments/assets/613a91ef-1da6-4b3b-8b58-c9bc7b4b5b42" />


✨ Example 2 — Change to gold

 <img width="448" height="1306" alt="image" src="https://github.com/user-attachments/assets/ac6a2d8a-2e69-4b24-b8f4-961c73e7909f" />


✨ Example 3 — Change to night

 <img width="558" height="1328" alt="image" src="https://github.com/user-attachments/assets/acb82b53-e0cc-48e7-9c57-bd33deda0a48" />


⸻
📌 3. Inference Hyperparameter Analysis

3-1. num_inference_steps 영향

(시간 증가 vs 품질 증가의 trade-off)
<img width="976" height="312" alt="image" src="https://github.com/user-attachments/assets/73e14069-3757-49da-a8bb-e6279d27ec36" />
→ 20 steps가 속도·품질 균형 최적

3-2. guidance_scale 영향

(프롬프트 반영 강도)
<img width="976" height="414" alt="image" src="https://github.com/user-attachments/assets/4a410d00-37ec-4600-b31c-4f8715d6a8b0" />
→ 1.0 ~ 1.5 권장

📌 4. LoRA 실험 (Full FT vs LFF vs LPF)

4-1. Parameter Comparison
<img src="PARAM_GRAPH" width="700">
<img width="954" height="338" alt="image" src="https://github.com/user-attachments/assets/81a372b7-e01b-4ec9-b190-d0c730234961" />

→ LPF는 Full FT 대비 약 0.018%의 파라미터만 사용
4-2. Editing Quality Comparison

Dataset Example

Prompt: "Transform the natural image into a cartoonish version."

🔴 Full Fine-tuning (BFF)
	•	Ground Truth와 가장 유사
	•	가장 높은 품질
	•	가장 무거움

🔵 LoRA Method 1 (LFF)
	•	색 번짐, Ghosting, Hard-edge overshoot 발생
	•	품질 저하 큼

🟢 LoRA Method 2 (LPF)
	•	원본 구조 보전
	•	자연스러운 cartoon style 반영
	•	Full FT 대비 매우 가벼움
	•	가장 효율적 모델

⸻

📌 5. LoRA Partial Finetuning (LPF) — Block 조합 비교
<img width="1042" height="324" alt="image" src="https://github.com/user-attachments/assets/beb44712-40e8-46b6-bc0d-5eab8c6ca522" />
→ [‘up’, ‘mid’] 조합이 품질/시간 균형에서 가장 우수

📌 6. Experiment Conclusion

✔ Full FT
	•	최고의 품질
	•	하지만 가장 느리고 가장 무거움
	•	실사용에는 부적합

✔ LoRA Method 1 (LFF)
	•	파라미터 적지만
	•	색 번짐, Artifact 발생
	•	안정성 문제 있어 실사용 어려움

✔ LoRA Method 2 (LPF)
	•	가장 가벼움(0.16M)
	•	품질은 Full FT보다 낮지만 LFF보다 훨씬 안정적
	•	특히 'up' + 'mid' 조합이 최적
	•	실시간·저비용 환경에 매우 적합

⸻

📌 7. Final Recommendation

본 프로젝트의 최종 선정 모델은
→ LoRA Method 2 (LPF) — Partial UNet LoRA with [‘up’, ‘mid’] 블록 적용

이 방식이
	•	가장 적은 파라미터
	•	가장 빠른 학습
	•	가장 안정적인 출력
	•	실사용 적용 가능성 높음
	
📌 8. Code Structure
IP2P_LoRA_FT/
│
├── lora_utils.py          # LoRA layer, partial LoRA, full LoRA 적용 유틸 함수
│
├── prac_1.ipynb           # 실습 1: 기본 Image Editing (Prompt-based Editing)
│
├── prac_3.ipynb           # 실습 3: LoRA 구현 및 Full LoRA Fine-tuning
│
├── prac_4.ipynb           # 실습 4: Partial LoRA Fine-tuning (LPF) + 비교 실험
│
└── README.md              # 프로젝트 설명 문서 (작성 예정)
