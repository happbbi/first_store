import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time

# --- [1] 데이터 준비 및 AI 모델 학습 ---
@st.cache_resource
def setup_ai_model():
    # 1. 가상 데이터 생성 (우수, 기초부족, 심화부족 3유형)
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'solve_time': np.concatenate([
            np.random.normal(100, 20, 100),  # 우수
            np.random.normal(200, 40, 100),  # 기초부족
            np.random.normal(350, 60, 100)   # 심화부족
        ]),
        'hint_count': np.concatenate([
            np.random.randint(0, 2, 100), 
            np.random.randint(2, 5, 100), 
            np.random.randint(6, 10, 100)
        ]),
        'accuracy_rate': np.concatenate([
            np.random.uniform(0.8, 1.0, 100),
            np.random.uniform(0.4, 0.7, 100),
            np.random.uniform(0.1, 0.4, 100)
        ]),
        'retry_count': np.concatenate([
            np.random.randint(1, 3, 100),
            np.random.randint(3, 7, 100),
            np.random.randint(8, 15, 100)
        ]),
        'label': np.array([0]*100 + [1]*100 + [2]*100)
    }
    
    df = pd.DataFrame(data)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 2. XGBoost 모델 학습
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, objective='multi:softprob')
    model.fit(X, y)
    return model

# 모델 로드 및 유형 정의
model = setup_ai_model()
types = {0: "자기주도 우수형", 1: "기초 개념 보완형", 2: "학습 의욕 저하 및 포기형"}

# --- [2] Streamlit 웹 UI 구성 ---
st.set_page_config(page_title="수학 AI 진단 서비스", layout="centered")

st.title("📊 수학 학습 유형 AI 진단 시스템")
st.markdown("학생의 **문제 해결 과정 데이터**를 분석하여 최적의 학습 유형을 분류합니다.")
st.divider()

# 사용자 입력창 배치
st.subheader("📝 학생 데이터 입력")
col1, col2 = st.columns(2)

with col1:
    solve_time = st.number_input("평균 문제 풀이 시간 (초)", min_value=0, value=120, help="한 문제를 푸는 데 걸리는 평균 시간")
    hint_count = st.slider("힌트 사용 횟수", 0, 15, 2)

with col2:
    accuracy = st.slider("최근 정답률 (%)", 0, 100, 75)
    retry_count = st.number_input("평균 재시도 횟수", min_value=0, value=1)

# --- [3] 분석 실행 및 결과 출력 ---
if st.button("AI 분석 시작", use_container_width=True):
    with st.spinner('사용자의 학습 패턴을 알고리즘으로 분석 중입니다...'):
        # 입력 데이터 변환 (시간, 힌트, 정답률(0~1), 재시도)
        user_data = np.array([[solve_time, hint_count, accuracy/100, retry_count]])
        
        # AI 예측
        prediction = model.predict(user_data)[0]
        result_name = types[prediction]
        time.sleep(1.5) # 연출용 대기 시간

    # 결과 대시보드
    st.success("✅ 분석이 완료되었습니다!")
    st.divider()
    
    # 결과 요약
    st.subheader(f"당신의 학습 유형은 :blue[[{result_name}]] 입니다.")
    
    # 시각화 데이터 설정 (유형에 따른 능력치 차별화)
    if prediction == 0:
        scores = [95, 90, 85]
    elif prediction == 1:
        scores = [50, 70, 45]
    else:
        scores = [20, 30, 25]

    chart_data = pd.DataFrame({
        '역량 항목': ['개념 이해도', '연산 정확도', '문제 해석력'],
        '점수': scores
    })

    # 막대 차트 출력
    st.bar_chart(data=chart_data, x='역량 항목', y='점수')

    # 하단 피드백 박스
    with st.expander("📌 유형별 맞춤 학습 전략 보기"):
        if prediction == 0:
            st.write("이미 훌륭한 학습 습관을 갖추고 있습니다. 고난도 응용 문제와 심화 탐구 프로젝트에 집중해 보세요.")
        elif prediction == 1:
            st.write("개념의 정의를 다시 확인하는 과정이 필요합니다. 오답 노트를 활용해 반복되는 실수 패턴을 잡아보세요.")
        else:
            st.write("학습에 대한 부담을 줄이는 것이 급선무입니다. 아주 쉬운 단계부터 성공 경험을 쌓는 '스몰 스텝' 전략을 추천합니다.")

st.sidebar.info("본 시스템은 XGBoost 알고리즘을 활용한 학생 맞춤형 교육 솔루션 프로토타입입니다.")