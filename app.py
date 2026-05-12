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
            np.random.normal(350, 100, 100),  # 기초부족
            np.random.normal(200, 50, 100)   # 심화부족
        ]),
        'hint_count': np.concatenate([
            np.random.randint(0, 2, 100), 
            np.random.randint(6, 10, 100), 
            np.random.randint(2, 5, 100)
        ]),
        'accuracy_rate': np.concatenate([
            np.random.uniform(0.8, 1.0, 100),
            np.random.uniform(0.1, 0.4, 100),
            np.random.uniform(0.4, 0.7, 100)
        ]),
        'retry_count': np.concatenate([
            np.random.randint(1, 3, 100),
            np.random.randint(8, 15, 100),
            np.random.randint(3, 7, 100)
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

st.set_page_config(page_title="수학 AI 진단 서비스", layout="centered")

if 'start_times' not in st.session_state:
    st.session_state.update({
        'start_times': [None] * 3,
        'solved': [False] * 3,
        'solve_times': [0] * 3,
        'retry_counts': [0] * 3,
        'hint_checks': [0] * 3,
        'gave_up': [False] * 3,
        'total_attempts': 0,
        'diagnosis_mode': False
    })

# --- [3] 문제 데이터 설정 ---
problems = [
    {"img": "math_problem.png", "ans": "3", "hint": "각 좌표의 값을 먼저 구하고 같은 크기의 도형을 찾으세요!"},
    {"img": "math_problem1.png", "ans": "20", "hint": "지수를 통합한 뒤 양변에 밑이 같은 로그를 취하여 풀어보세요!"},
    {"img": "math_problem2.png", "ans": "12", "hint": "중심각의 크기를 삼각형을 통해 구한 뒤, 호의 길이 공식을 이용해보세요!"}
]

st.title("📊 수학 학습 유형 AI 진단 시스템")
st.write("문제를 풀면 AI가 당신의 학습 태도와 실력을 분석합니다.")

# --- [4] 실전 문제 풀이 섹션 ---
for i in range(3):
    st.divider()
    st.subheader(f"📝 문제 {i+1}번")
    
    # 이미지 출력 (파일이 없을 경우 대비 예외처리)
    try:
        st.image(problems[i]['img'], use_container_width=True)
    except:
        st.error(f"'{problems[i]['img']}' 파일을 찾을 수 없습니다.")

    if st.session_state.solved[i] or st.session_state.gave_up[i]:
        status = "✅ 정답" if st.session_state.solved[i] else "❌ 미해결(넘어감)"
        st.info(f"{status} | 소요시간: {st.session_state.solve_times[i]}초 | 시도: {st.session_state.retry_counts[i]}회")
        continue

    # 시작 버튼
    if st.session_state.start_times[i] is None:
        if st.button(f"⏱️ {i+1}번 풀이 시작", key=f"start_{i}"):
            st.session_state.start_times[i] = time.time()
            st.rerun()
    else:
        # 힌트 및 정답 입력
        with st.expander(f"🔍 {i+1}번 힌트 확인"):
            st.write(problems[i]['hint'])
            if st.button("힌트 사용 기록하기", key=f"h_count_{i}"):
                st.session_state.hint_checks[i] += 1
                st.toast("힌트 사용이 기록되었습니다.")

        user_ans = st.text_input(f"{i+1}번 정답", key=f"in_{i}", placeholder="숫자만 입력")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"✅ 정답 확인", key=f"chk_{i}", use_container_width=True):
                st.session_state.total_attempts += 1
                if user_ans == problems[i]['ans']:
                    st.session_state.solve_times[i] = int(time.time() - st.session_state.start_times[i])
                    st.session_state.solved[i] = True
                    st.success("정답입니다!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.retry_counts[i] += 1
                    st.error(f"오답! (시도: {st.session_state.retry_counts[i]}회)")
        
        with c2:
            if st.session_state.retry_counts[i] >= 2:
                if st.button("❓ 모르겠어요", key=f"skip_{i}", use_container_width=True):
                    st.session_state.solve_times[i] = int(time.time() - st.session_state.start_times[i])
                    st.session_state.gave_up[i] = True
                    st.warning("오답 처리되었습니다.")
                    time.sleep(1)
                    st.rerun()

# --- [5] 결과 분석 및 AI 진단 ---
if all(st.session_state.solved[i] or st.session_state.gave_up[i] for i in range(3)):
    st.divider()
    
    # 데이터 자동 계산
    correct_count = sum(st.session_state.solved)
    total_attempts = max(1, st.session_state.total_attempts)
    accuracy_val = (correct_count / total_attempts) * 100
    avg_time = sum(st.session_state.solve_times) / 3
    total_hints = sum(st.session_state.hint_checks)
    total_retries = sum(st.session_state.retry_counts)

    st.subheader("📊 테스트 요약")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("정답률(시도대비)", f"{accuracy_val:.1f}%")
    col_b.metric("평균 시간", f"{avg_time:.1f}초")
    col_c.metric("총 힌트 사용", f"{total_hints}회")

    if st.button("🚀 AI 정밀 진단 시작", use_container_width=True):
        st.session_state.diagnosis_mode = True

    if st.session_state.diagnosis_mode:
        with st.spinner('XGBoost 모델이 학습 데이터를 분석 중...'):
            # 자동 수집된 데이터를 AI 모델 입력값으로 사용
            user_data = np.array([[avg_time, total_hints, accuracy_val/100, total_retries]])
            prediction = model.predict(user_data)[0]
            result_name = types[prediction]
            time.sleep(1.5)

        st.success(f"분석 완료! 당신은 **[{result_name}]** 입니다.")
        
        # 그래프 시각화
        scores = [95, 90, 85] if prediction == 0 else ([50, 70, 45] if prediction == 1 else [20, 40, 30])
        chart_data = pd.DataFrame({'항목': ['개념', '연산', '해석'], '점수': scores})
        st.bar_chart(data=chart_data, x='항목', y='점수')

        with st.expander("📌 맞춤형 처방전"):
            st.write(f"현재 당신의 정답률은 {accuracy_val:.1f}%입니다. {result_name} 유형에 맞는 학습이 필요합니다.")
            st.write("<자기주도 우수형> 현재 학습방식이 매우 좋으며, 심화 학습을 통해 실력 향상을 추진할 것을 추천함")
            st.write("<기초 개념 보완형> 현재 기초개념이 부족한 것으로 보이며, 개념원리와 같은 개념서를 통해 개념을 보충해 나가는 것을 추천함")
            st.write("<학습 의욕 저하 및 포기형> 현재 심화문제에서 어려움을 느끼고 있는 것으로 보이며, 수학에 대한 거부감을 줄이기는 것이 최우선으로 보임")
