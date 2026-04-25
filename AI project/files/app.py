"""
Student Grade Predictor — Streamlit GUI

Run: streamlit run app.py
"""

import streamlit as st
import joblib, numpy as np, os
from pathlib import Path

st.set_page_config(
    page_title="Grade Predictor — Made by Abdul Sami",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #f0f4f8; }

.hero {
    background: linear-gradient(135deg, #1a3a5c 0%, #0e5f8a 50%, #1a3a5c 100%);
    border-radius: 18px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(14,95,138,0.25);
}
.hero h1 { color: white; font-size: 2rem; font-weight: 800; margin: 0 0 0.3rem 0; }
.hero p  { color: #a8d4f0; font-size: 0.9rem; margin: 0; }

.card {
    background: white; border-radius: 14px; padding: 1.5rem 1.8rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 1rem;
    border-left: 4px solid #0e5f8a;
}
.card-title { font-size: 1rem; font-weight: 700; color: #1a3a5c; margin-bottom: 0.8rem; }

.result-box {
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}
.grade-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4rem; font-weight: 700; line-height: 1;
}
.grade-letter { font-size: 1.4rem; font-weight: 700; margin-top: 0.3rem; }
.grade-msg    { font-size: 0.88rem; margin-top: 0.5rem; opacity: 0.85; }

.metric-card {
    background: white; border-radius: 12px; padding: 1rem;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-val { font-size: 1.6rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #0e5f8a; }
.metric-lbl { font-size: 0.73rem; color: #6b7c93; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

.factor-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.4rem 0; border-bottom: 1px solid #f0f0f0; font-size: 0.88rem;
}
.factor-label { color: #4a5568; }
.factor-impact { font-weight: 700; }
.pos { color: #27ae60; }
.neg { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent
    model = joblib.load(base / "models" / "best_model.pkl")
    metrics = joblib.load(base / "models" / "metrics.pkl")
    return model, metrics

model, metrics_data = load_artifacts()

def grade_letter_color(g):
    if g >= 85: return "A", "#27ae60", "#e8f8f0"
    if g >= 70: return "B", "#2980b9", "#e8f4fd"
    if g >= 55: return "C", "#f39c12", "#fef9e7"
    if g >= 40: return "D", "#e67e22", "#fef5e7"
    return "F", "#e74c3c", "#fdedec"

def predict(inputs):
    arr = np.array([[
        inputs["study_hours"], inputs["attendance_pct"], inputs["prev_gpa"],
        inputs["assignments_done"], inputs["sleep_hours"], inputs["part_time_job"],
        inputs["internet_access"], inputs["tutoring"], inputs["family_support"],
        inputs["stress_level"], inputs["gender_enc"], inputs["major_enc"]
    ]])
    return float(np.clip(model.predict(arr)[0], 0, 100))

# ── Sidebar — Model Info ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Project Info")
    st.markdown("""
    <div style='background:#e8f4fd;border:1px solid #a8d4f0;border-radius:10px;padding:1rem;font-size:0.83rem;color:#1a3a5c;'>
    <b>Course:</b> Artificial Intelligence<br>
    <b>Semester:</b> Spring 2026<br>
    <b>Model:</b> Linear Regression<br>
    <b>Features:</b> 12 student attributes
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    best = metrics_data["best_model_name"]
    r    = metrics_data["results"][best]
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{r['r2']:.3f}</div><div class='metric-lbl'>R² Score</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{r['mae']:.1f}</div><div class='metric-lbl'>MAE</div></div>", unsafe_allow_html=True)
    st.markdown(f"<br><div class='metric-card'><div class='metric-val'>{r['cv_mean']:.3f}</div><div class='metric-lbl'>5-Fold CV R²</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔬 All Models")
    for name, mr in metrics_data["results"].items():
        star = " ⭐" if name == best else ""
        st.markdown(f"**{name}{star}**  \nR²={mr['r2']:.3f} | MAE={mr['mae']:.1f}")
        st.markdown("")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🎓 Student Grade Predictor</h1>
  <p>AI-powered final grade prediction using 12 student performance attributes · Sukkur IBA University · Spring 2026</p>
</div>
""", unsafe_allow_html=True)

col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:
    st.markdown("<div class='card'><div class='card-title'>📚 Academic Factors</div>", unsafe_allow_html=True)
    study_hours      = st.slider("Daily Study Hours", 0.0, 10.0, 4.0, 0.5)
    attendance_pct   = st.slider("Attendance (%)", 40, 100, 80)
    prev_gpa         = st.slider("Previous GPA (out of 4.0)", 1.5, 4.0, 3.0, 0.05)
    assignments_done = st.slider("Assignments Completed (out of 10)", 0, 10, 8)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>🏠 Lifestyle Factors</div>", unsafe_allow_html=True)
    sleep_hours   = st.slider("Sleep Hours per Night", 4.0, 10.0, 7.0, 0.5)
    stress_level  = st.select_slider("Stress Level", options=[1,2,3,4,5],
                                     value=3, format_func=lambda x: ["Very Low","Low","Medium","High","Very High"][x-1])
    part_time_job = st.radio("Has Part-time Job?", ["No","Yes"], horizontal=True) == "Yes"
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>🌐 Support & Resources</div>", unsafe_allow_html=True)
    internet_access = st.radio("Internet Access at Home?", ["Yes","No"], horizontal=True) == "Yes"
    tutoring        = st.radio("Receives Tutoring?",         ["No","Yes"], horizontal=True) == "Yes"
    family_support  = st.select_slider("Family Support Level", options=[1,2,3,4,5],
                                       value=3, format_func=lambda x: ["Very Low","Low","Medium","High","Very High"][x-1])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>👤 Student Profile</div>", unsafe_allow_html=True)
    gender = st.radio("Gender", ["Male","Female"], horizontal=True)
    major  = st.selectbox("Major", ["CS","EE","BBA","Mathematics","B.Ed"])
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔮 Predict Grade", type="primary", use_container_width=True)

with col_result:
    major_map = {"CS":0,"EE":1,"BBA":2,"Mathematics":3,"B.Ed":4}
    inputs = dict(
        study_hours=study_hours, attendance_pct=float(attendance_pct),
        prev_gpa=prev_gpa, assignments_done=float(assignments_done),
        sleep_hours=sleep_hours, part_time_job=int(part_time_job),
        internet_access=int(internet_access), tutoring=int(tutoring),
        family_support=float(family_support), stress_level=float(stress_level),
        gender_enc=int(gender == "Female"), major_enc=major_map[major]
    )

    # Live preview (always shown)
    grade = predict(inputs)
    letter, color, bg = grade_letter_color(grade)
    msg = {"A":"Excellent! Outstanding performance predicted.",
           "B":"Good standing. Keep up the solid effort.",
           "C":"Satisfactory. There is room to improve.",
           "D":"Below average. Consider seeking support.",
           "F":"At risk. Immediate action recommended."}[letter]

    st.markdown("#### 📊 Predicted Grade")
    st.markdown(f"""
    <div class='result-box' style='background:{bg};border:2px solid {color};'>
        <div class='grade-num' style='color:{color};'>{grade:.1f}</div>
        <div class='grade-letter' style='color:{color};'>Grade {letter}</div>
        <div class='grade-msg' style='color:#4a5568;'>{msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Grade breakdown bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(int(grade))
    st.markdown(f"<p style='text-align:center;font-size:0.82rem;color:#6b7c93;'>Score: {grade:.1f} / 100</p>", unsafe_allow_html=True)

    # Key factor analysis
    st.markdown("#### 🔍 Key Factors Analysis")
    factors = [
        ("Previous GPA",       prev_gpa/4.0,       True,  "Most influential feature"),
        ("Study Hours/Day",    study_hours/10.0,    True,  "Strong positive impact"),
        ("Attendance",         (attendance_pct-40)/60.0, True, "Attendance strongly matters"),
        ("Assignments Done",   assignments_done/10.0, True, "Completion rate"),
        ("Stress Level",       stress_level/5.0,    False, "High stress hurts performance"),
        ("Part-time Job",      int(part_time_job),  False, "Reduces study time"),
    ]
    for label, val, positive, tip in factors:
        bar_color = "#27ae60" if positive else "#e74c3c"
        direction = "▲ Positive" if positive else "▼ Negative"
        dir_color = "pos" if positive else "neg"
        st.markdown(f"""
        <div class='factor-row'>
            <span class='factor-label'>{label}</span>
            <span class='factor-impact {dir_color}'>{direction} ({val*100:.0f}%)</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(val)

    # Recommendations
    st.markdown("#### 💡 Recommendations")
    recs = []
    if study_hours < 3:    recs.append("📖 Increase daily study hours to at least 3–4 hours.")
    if attendance_pct < 75: recs.append("🏫 Improve attendance — aim for above 75%.")
    if assignments_done < 7: recs.append("✏️ Complete more assignments; aim for 8+/10.")
    if stress_level >= 4:  recs.append("🧘 Manage stress levels through exercise or counselling.")
    if sleep_hours < 6:    recs.append("😴 Get at least 7 hours of sleep for better retention.")
    if not tutoring:       recs.append("👨‍🏫 Consider enrolling in tutoring sessions.")
    if not recs:           recs.append("✅ Great habits! Maintain your current routine.")
    for r in recs[:4]:
        st.info(r)

st.markdown("---") # Keep the horizontal line
st.markdown("""
<div style='text-align: center; color: #6b7c93; font-size: 0.8rem; padding-bottom: 20px;'>
    <b>AI Semester Project · Spring 2026</b><br>
    Developed by <b>Abdul Sami</b> | CMS ID: 023-23-0087<br>
    Sukkur IBA University
</div>
""", unsafe_allow_html=True)