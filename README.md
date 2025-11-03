# AdaBoost 구현 및 성능 분석

이 프로젝트는 '기계학습' 과제로, AdaBoost 알고리즘을 처음부터(from scratch) 구현한 것입니다.

## 1. 프로젝트 목표
* 강의 자료를 기반으로 AdaBoost 알고리즘 구현
* Breast Cancer Wisconsin 데이터셋 활용
* 약한 학습기가 추가됨에 따른 테스트 정확도 변화 관찰

## 2. 핵심 구현
`MyAdaBoost` 클래스는 다음의 핵심 로직을 포함합니다.
* 샘플 가중치(w) 업데이트
* 분류기 계수(alpha_t) 계산
* 가중 오류(err_t) 계산

## 3. 실험 결과
### 기본 실험: 모델 수(T)에 따른 정확도
![기본 실험 결과](adaboost_accuracy.png)

### 추가 실험: 약한 학습기 복잡도 비교
![비교 실험 결과](adaboost_comparison.png)
