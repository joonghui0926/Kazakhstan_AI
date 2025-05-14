# Kazakhstan_AI

# KAIST AI Camp Code (Kazakhstan NIS, Summer 2025)

본 저장소는 KAIST 글로벌 봉사단 AI팀이 카자흐스탄 알마티 나자르바예프 영재학교(NIS)를 대상으로 진행하는 AI 교육 프로그램의 실습 코드 모음입니다. 각 파일은 8차시 수업 구성에 맞춰 순서대로 정리되어 있으며, Google Colab 환경에서 실행 가능하도록 작성되었습니다.

## 수업 개요

- 교육 대상: 고등학생 (AI 기초 ~ 중급)
- 수업 방식: 이론 + 실습 + 프로젝트 기반 학습
- 개발 환경: Python, Google Colab, TensorFlow, PyTorch, HuggingFace

## 파일 구성

### Session1_Perceptron_Backprop.py
- XOR 문제를 해결하며 퍼셉트론과 다층 퍼셉트론(MLP)의 작동 원리와 학습 알고리즘(backpropagation)을 실습합니다.

### Session2_CNN_Visualization.py
- MNIST 데이터를 기반으로 간단한 CNN 구조를 구현하고, 합성곱 필터 시각화를 통해 이미지 특징 추출 원리를 이해합니다.

### Session3_UserImageClassifier.py
- 학생이 직접 수집한 이미지 데이터를 활용해 사용자 정의 이미지 분류기를 설계하고 학습시킵니다.

### Session4_RNN_LSTM_Sentiment.py
- 시계열 데이터와 텍스트 데이터를 다루기 위한 RNN 및 LSTM 구조를 구현하고, 감정 분류 문제를 실습합니다.

### Session5_NLP_Token_Embedding.py
- 토큰화(Tokenization)와 임베딩(Embedding)의 개념을 실습하고, 문장을 벡터로 변환하는 방법을 체험합니다.

### Session6_Transformer_Attention.py
- Transformer 구조의 핵심인 Scaled Dot-Product Attention을 직접 구현하고, 입력 벡터 간의 연관성을 계산하는 과정을 실험합니다.

### Session7_GPT2_TextGeneration.py
- HuggingFace 라이브러리를 활용하여 GPT-2 기반 텍스트 생성 모델을 실행하고, 다양한 프롬프트에 따른 출력 결과를 실험합니다.

### Session8_Project_Template.py
- 최종 프로젝트 진행을 위한 템플릿으로, 모델 설계 및 학습, 결과 시각화, 발표 준비 흐름을 안내합니다.

## 실행 방법

1. Google Colab에 접속합니다.
2. 각 세션별 `.py` 파일 내용을 복사하여 새로운 Colab 노트북에 붙여넣습니다.
3. 코드 실행 전에 필요한 라이브러리가 설치되어 있는지 확인하고, 데이터 경로를 수정합니다.

## 참고 사항

- 본 수업은 Stanford CS231n 강의를 기반으로 구성하되, 고등학생 대상 교육에 맞추어 실습 중심으로 재구성하였습니다.
- 데이터셋은 공개 데이터 또는 현장에서 직접 수집한 데이터를 사용할 수 있습니다.
- 모든 코드는 Colab 기준으로 작성되었으며, 로컬 실행 시 환경 설정이 필요할 수 있습니다.
