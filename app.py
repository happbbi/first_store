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

if 'start_times' not in st.session_state:
    st.session_state.start_times = [None, None, None] # 각 문제 시작 시간
if 'solved' not in st.session_state:
    st.session_state.solved = [False, False, False]   # 각 문제 해결 여부
if 'solve_times' not in st.session_state:
    st.session_state.solve_times = [0, 0, 0]          # 각 문제 소요 시간

# --- [문제 데이터 설정] ---
problems = [
    {"img": "math_problem.png", "ans": "3", "hint": "이 문제에서는 먼저 각 좌표의 값을 구하는 것이 가장 중요합니다!! 좌표를 각자 구한 뒤 같은 크기를 가지는 도형을 찾아주세요!!"},
    {"img": "math_problem1.png", "ans": "20", "hint": "이 문제에서는 양변에 밑이 같은 로그를 취하여 풀어나가야 합니다!! 최댓값을 그대로 쓰지 않고, 이는 로그를 취한 수라는 것을 잊지 않는 것이 중요합니다!!!"},
    {"img": "math_problem2.png", "ans": "12", "hint": "이 문제에서는 접힌 활꼴의 호의 중심각의 크기를 파악하는 것이 중요합니다!! 호의 길이 공식을 이용해서 구해주세요!"}
]

st.title("✍️ 수학 실전 테스트 (3문제)")
st.write("모든 문제를 풀어야 AI 정밀 진단 결과를 볼 수 있습니다.")

# --- [문제 반복 생성] ---
for i in range(3):
    st.divider()
    st.subheader(f"📝 문제 {i+1}번")
    st.image(problems[i]['img'], use_container_width=True)
    
    # 1. 시작 버튼
    if not st.session_state.solved[i]:
        if st.button(f"⏱️ {i+1}번 문제 풀이 시작", key=f"start_{i}"):
            st.session_state.start_times[i] = time.time()
            st.info(f"{i+1}번 문제 측정을 시작합니다.")

        # 2. 힌트
        with st.expander(f"🔍 {i+1}번 힌트 보기"):
            st.write(problems[i]['hint'])

        # 3. 정답 입력
        user_ans = st.text_input(f"{i+1}번 정답 입력", key=f"input_{i}")

        if st.button(f"✅ {i+1}번 정답 확인", key=f"check_{i}"):
            if st.session_state.start_times[i] is None:
                st.warning("먼저 '풀이 시작' 버튼을 눌러주세요.")
            elif user_ans == problems[i]['ans']:
                elapsed = int(time.time() - st.session_state.start_times[i])
                st.session_state.solve_times[i] = elapsed
                st.session_state.solved[i] = True
                st.success(f"정답입니다! 소요 시간: {elapsed}초")
                st.balloons()
            else:
                st.error("오답입니다. 다시 시도해보세요!")
    else:
        st.success(f"✅ 완료! (소요 시간: {st.session_state.solve_times[i]}초)")

# --- [최종 AI 진단 연결] ---
st.divider()

# 1. 진단 모드 상태 관리 변수 초기화
if 'diagnosis_mode' not in st.session_state:
    st.session_state.diagnosis_mode = False

if all(st.session_state.solved):
    total_time = sum(st.session_state.solve_times)
    avg_time = total_time / 3
    
    st.subheader("🎉 모든 문제를 풀었습니다!")
    st.write(f"총 소요 시간: {total_time}초 (평균 {avg_time:.1f}초)")
    
    # 버튼을 누르면 진단 모드를 True로 바꿈
    if st.button("🚀 AI 정밀 진단 결과 보기"):
        st.session_state.diagnosis_mode = True

    # 진단 모드가 True일 때만 아래 내용 표시
    if st.session_state.diagnosis_mode:
        st.subheader("📝 학생 데이터 입력")
        col1, col2 = st.columns(2)

        with col1:
            # 수동 입력 대신 자동으로 측정된 평균 시간을 기본값(value)으로 넣어주면 더 좋습니다!
            solve_time = st.number_input("평균 문제 풀이 시간 (초)", min_value=0, value=int(avg_time))
            hint_count = st.slider("힌트 확인 횟수", 0, 15, 2)

        with col2:
            accuracy = st.slider("최근 정답률 (%)", 0, 100, 75)
            retry_count = st.number_input("평균 재시도 횟수", min_value=0, value=1)

        # 분석 버튼 (이제 독립적으로 작동합니다)
        if st.button("AI 분석 시작", use_container_width=True):
            with st.spinner('사용자의 학습 패턴을 알고리즘으로 분석 중입니다...'):
                user_data = np.array([[solve_time, hint_count, accuracy/100, retry_count]])
                prediction = model.predict(user_data)[0]
                result_name = types[prediction]
                time.sleep(1.5)

            st.success("✅ 분석이 완료되었습니다!")
            st.divider()
            st.subheader(f"당신의 학습 유형은 :blue[[{result_name}]] 입니다.")

            # (이후 그래프 및 피드백 코드는 동일하게 유지)
            if prediction == 0: scores = [95, 90, 85]
            elif prediction == 1: scores = [50, 70, 45]
            else: scores = [20, 30, 25]

            chart_data = pd.DataFrame({'역량 항목': ['개념 이해도', '연산 정확도', '문제 해석력'], '점수': scores})
            st.bar_chart(data=chart_data, x='역량 항목', y='점수')

            with st.expander("📌 유형별 맞춤 학습 전략 보기"):
                if prediction == 0: st.write("이미 훌륭한 학습 습관을 갖추고 있습니다! 심화 문제를 바탕으로 실력을 키워보세요!!")
                elif prediction == 1: st.write("개념의 정의를 다시 확인하는 과정이 필요합니다! 차근차근 하다보면 실력이 상승할 수 있어요!")
                else: st.write("학습에 대한 부담을 줄이는 것이 급선무입니다. 수학은 생각보다 재밌는 과목일 수도 있어요!!")

        st.sidebar.info("본 시스템은 XGBoost 알고리즘을 활용한 프로토타입입니다.")     
else:
    st.warning("모든 문제를 풀어야 AI 진단 버튼이 활성화됩니다.")