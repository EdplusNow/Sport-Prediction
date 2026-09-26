import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="IPL Win Predictor", page_icon="# ", layout="centered")

st.title(" IPL Live Win Probability Predictor")

# ---------- DEBUG INFO (temporary, helps us see what's wrong if it fails) ----------
with st.expander("🔧 Debug Info (click if app looks broken)"):
    st.write("Files in current directory:", os.listdir('.'))
    st.write("Streamlit version:", st.__version__)

# ---------- LOAD MODEL & DATA (with visible error handling) ----------
try:
    with open('win_predictor_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except Exception as e:
    st.error(f" Could not load model file: {e}")
    st.stop()

try:
    with open('teams_cities.pkl', 'rb') as f:
        data = pickle.load(f)
    teams = data['teams']
    cities = data['cities']
except Exception as e:
    st.error(f" Could not load teams/cities file: {e}")
    st.stop()

st.markdown("""
This app uses a **Machine Learning model** trained on real IPL ball-by-ball data 
to predict the live win probability of the chasing team based on the current 
match situation.
""")
st.divider()

# ---------- INPUT SECTION ----------
st.subheader(" Enter Match Situation")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team (Chasing)", sorted(teams))
with col2:
    bowling_team = st.selectbox("Bowling Team", sorted([t for t in teams if t != batting_team]))

city = st.selectbox("Match City / Venue", sorted(cities))

col3, col4 = st.columns(2)
with col3:
    target = st.number_input("Target Score", min_value=1, max_value=300, value=180, step=1)
with col4:
    current_score = st.number_input("Current Score", min_value=0, max_value=300, value=100, step=1)

col5, col6 = st.columns(2)
with col5:
    overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=19.5, value=12.0, step=0.1)
with col6:
    wickets_fallen = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=3, step=1)

# ---------- PREDICT BUTTON ----------
if st.button(" Predict Win Probability", use_container_width=True):
    if current_score >= target:
        st.success(f" {batting_team} has already won this match!")
    else:
        runs_left = target - current_score
        balls_completed = int(overs_completed) * 6 + round((overs_completed % 1) * 10)
        balls_left = 120 - balls_completed
        wickets_left = 10 - wickets_fallen

        if balls_left <= 0:
            st.error(f" {batting_team} lost the match (overs completed).")
        else:
            crr = (current_score * 6) / balls_completed if balls_completed > 0 else 0
            rrr = (runs_left * 6) / balls_left

            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'target': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            try:
                proba = pipeline.predict_proba(input_df)[0]
            except Exception as e:
                st.error(f" Prediction failed: {e}")
                st.stop()

            win_prob = round(proba[1] * 100, 1)
            lose_prob = round(proba[0] * 100, 1)

            st.divider()
            st.subheader(" Prediction Result")

            st.markdown(f"**{batting_team}: {win_prob}%**")
            st.progress(int(win_prob))
            st.markdown(f"**{bowling_team}: {lose_prob}%**")
            st.progress(int(lose_prob))

            m1, m2, m3 = st.columns(3)
            m1.metric("Runs Needed", runs_left)
            m2.metric("Balls Left", balls_left)
            m3.metric("Required Run Rate", f"{rrr:.2f}")

            st.divider()
            if win_prob >= 50:
                st.success(f" **{batting_team}** is favored to win with **{win_prob}%** probability!")
            else:
                st.warning(f" **{bowling_team}** is favored to win — {batting_team} needs {rrr:.2f} runs/over!")

st.divider()
st.caption("Built with Python, Scikit-learn/XGBoost & Streamlit | Educational ML Project")
