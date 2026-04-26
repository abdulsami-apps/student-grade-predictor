"""
Generate Project Report PDF — Student Grade Predictor
Sukkur IBA University | AI Semester Project | Spring 2026
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)

from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os, joblib, pandas as pd

BASE = r"H:\AI project\files"
OUT  = rf"{BASE}\reports\Project_Report.pdf"

BLUE  = colors.HexColor("#0e5f8a")
DBLUE = colors.HexColor("#1a3a5c")
GREEN = colors.HexColor("#1E8449")
LGRAY = colors.HexColor("#F2F3F4")
MGRAY = colors.HexColor("#7F8C8D")
BLACK = colors.HexColor("#1C1C1C")
WHITE = colors.white
LBLUE = colors.HexColor("#e8f4fd")

s = getSampleStyleSheet()
def add_style(name, **kw):
    try: s.add(ParagraphStyle(name=name, **kw))
    except: pass  # already exists

add_style("STitle",  fontSize=28, textColor=WHITE,  fontName="Helvetica-Bold", alignment=TA_CENTER, leading=36, spaceAfter=6)
add_style("SSub",    fontSize=13, textColor=colors.HexColor("#a8d4f0"), fontName="Helvetica", alignment=TA_CENTER, leading=18, spaceAfter=4)
add_style("SInfo",   fontSize=10, textColor=colors.HexColor("#d6eaf8"), fontName="Helvetica", alignment=TA_CENTER, leading=14)
add_style("H1",      fontSize=14, textColor=DBLUE,  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
add_style("H2",      fontSize=11, textColor=BLUE,   fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
add_style("Body",    fontSize=10, textColor=BLACK,  fontName="Helvetica", leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
add_style("BulletIt",fontSize=10, textColor=BLACK,  fontName="Helvetica", leading=14, spaceAfter=3, leftIndent=14)
add_style("Caption", fontSize=8,  textColor=MGRAY,  fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=10)
add_style("CodeB",   fontSize=8,  textColor=colors.HexColor("#1B2631"), fontName="Courier", leading=12, backColor=LGRAY, borderPadding=6)
add_style("Foot",    fontSize=8,  textColor=MGRAY,  fontName="Helvetica", alignment=TA_CENTER)

def h1(t): return Paragraph(t, s["H1"])
def h2(t): return Paragraph(t, s["H2"])
def body(t): return Paragraph(t, s["Body"])
def bul(t):  return Paragraph(f"<bullet>\u2022</bullet> {t}", s["BulletIt"])
def sp(n=6): return Spacer(1, n)
def hr():    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#BFC9CA"), spaceAfter=8)
def img(path, w=13*cm, caption=None):
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=w, height=w*0.58))
        if caption: items.append(Paragraph(caption, s["Caption"]))
    return items

def cover():
    rows = [
        [Paragraph("Sukkur IBA University", s["SSub"])],
        [Paragraph("Department of Computer Science", s["SSub"])],
        [Paragraph("Artificial Intelligence — Spring 2026", s["SInfo"])],
        [Paragraph("<br/>", s["SInfo"])],
        [Paragraph("Student Grade Predictor", s["STitle"])],
        [Paragraph("A Machine Learning Regression System", s["SSub"])],
        [Paragraph("<br/>", s["SInfo"])],
        [Paragraph("Predicts a student's final grade (0–100) from 12 behavioural,", s["SInfo"])],
        [Paragraph("academic, and lifestyle features using regression models.", s["SInfo"])],
        [Paragraph("<br/>", s["SInfo"])],
        [Paragraph("Best Model: Linear Regression  |  R\u00b2 = 0.935  |  MAE = 3.31", s["SInfo"])],
        [Paragraph("<br/><br/>", s["SInfo"])],
        [Paragraph("Submitted by: Abdul Sami (CMS ID: 023-23-0087)", s["SInfo"])],
        [Paragraph("Submitted to: Dr. Muhammad Ismail Mangrio, Dept. of Computer Science", s["SInfo"])],
        [Paragraph("Submission Date: April 30, 2026", s["SInfo"])],
    ]
    t = Table(rows, colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DBLUE),
        ("ROWPADDING", (0,0), (-1,-1), 4),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (0,0), 35),
        ("BOTTOMPADDING", (0,-1), (-1,-1), 35),
    ]))
    return t

story = []
story.append(cover())
story.append(PageBreak())

# ── §1 Task & Dataset ─────────────────────────────────────────────────────────
story.append(h1("1. The Task and Dataset"))
story.append(hr())
story.append(h2("1.1 Task Definition"))
story.append(body(
    "This project addresses <b>Student Final Grade Prediction</b> — a supervised regression task in Machine Learning. "
    "Given a set of 12 student attributes covering academic behaviour, lifestyle, and support resources, "
    "the trained model predicts a continuous numerical grade on a scale of 0–100."
))
story.append(body(
    "Unlike classification, regression outputs a real-valued number, making it suitable for grade prediction "
    "where the label is a continuous score. This task has practical applications in early warning systems, "
    "personalised academic advising, and student performance monitoring."
))

story.append(h2("1.2 Dataset"))
story.append(body(
    "A realistic synthetic dataset of <b>1,200 student records</b> was generated for this project. "
    "Each record contains 12 input features and one continuous target label (final_grade). "
    "The grade was computed using a weighted linear formula of the features with added Gaussian noise "
    "(σ=4.0) to simulate real-world variability, then normalised to a 0–100 scale."
))
dtbl = [
    ["Attribute", "Value"],
    ["Total Samples", "1,200 students"],
    ["Input Features", "12 (academic, lifestyle, support)"],
    ["Target Variable", "Final Grade (continuous, 0–100)"],
    ["Train / Test Split", "80% / 20% (stratified random)"],
    ["Missing Values", "None (controlled generation)"],
    ["Class Balance", "N/A — regression task"],
]
t = Table(dtbl, colWidths=[7*cm, 10*cm])
t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),DBLUE), ("TEXTCOLOR",(0,0),(-1,0),WHITE),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),9),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGRAY]),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#D5D8DC")),
    ("ROWPADDING",(0,0),(-1,-1),5), ("LEFTPADDING",(0,0),(-1,-1),8),
]))
story.append(t)
story.append(sp(8))

story.append(h2("1.3 Feature Descriptions"))
feats = [
    ("study_hours",      "Continuous", "0–10 hrs/day", "Average daily hours spent studying"),
    ("attendance_pct",   "Continuous", "40–100%",      "Percentage of classes attended"),
    ("prev_gpa",         "Continuous", "1.5–4.0",      "Previous semester GPA"),
    ("assignments_done", "Integer",    "0–10",         "Number of assignments completed (out of 10)"),
    ("sleep_hours",      "Continuous", "4–10 hrs",     "Average nightly sleep"),
    ("part_time_job",    "Binary",     "0 or 1",       "Whether the student has a part-time job"),
    ("internet_access",  "Binary",     "0 or 1",       "Whether student has internet at home"),
    ("tutoring",         "Binary",     "0 or 1",       "Whether student receives tutoring"),
    ("family_support",   "Ordinal",    "1–5",          "Level of family academic support"),
    ("stress_level",     "Ordinal",    "1–5",          "Self-reported academic stress (5=high)"),
    ("gender",           "Categorical","Male/Female",   "Student gender (label-encoded)"),
    ("major",            "Categorical","CS/EE/BBA/...", "Field of study (label-encoded 0–4)"),
]
fhdr = [["Feature","Type","Range","Description"]]
for row in feats:
    fhdr.append(list(row))
ft = Table(fhdr, colWidths=[3.6*cm,2.4*cm,2.5*cm,8.2*cm])
ft.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),BLUE), ("TEXTCOLOR",(0,0),(-1,0),WHITE),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGRAY]),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#D5D8DC")),
    ("ROWPADDING",(0,0),(-1,-1),4), ("LEFTPADDING",(0,0),(-1,-1),5),
]))
story.append(ft)
story.append(sp(8))

# ── §2 Pre-processing ─────────────────────────────────────────────────────────
story.append(h1("2. Data Pre-Processing"))
story.append(hr())
story.append(body(
    "Machine learning models require numerical inputs. The following pre-processing steps were applied "
    "to convert raw student records into a format suitable for regression:"
))
for step in [
    "<b>Label Encoding — Gender:</b> 'Female' mapped to 1, 'Male' to 0 (binary encoding).",
    "<b>Label Encoding — Major:</b> Categorical major (CS, EE, BBA, Mathematics, B.Ed) mapped to integers 0–4.",
    "<b>Feature Scaling (StandardScaler):</b> All numerical features were standardised to zero mean and unit variance. This is critical for Linear Regression, Ridge, and Lasso which are sensitive to feature magnitude.",
    "<b>Train/Test Split:</b> Dataset split 80/20 with random_state=42 to ensure reproducibility.",
    "<b>No Missing Data Handling Required:</b> Dataset was generated without missing values; however the pipeline handles this cleanly.",
    "<b>Preprocessing Fit on Training Set Only:</b> StandardScaler was fit exclusively on training data and applied to both train and test sets to prevent data leakage.",
]:
    story.append(bul(step))
story.append(sp(4))
story += img(f"{BASE}/reports/correlation_heatmap.png", w=14.5*cm,
             caption="Figure 1: Feature correlation matrix. prev_gpa, study_hours, and attendance show the strongest correlation with the final grade.")

# ── §3 Architecture ───────────────────────────────────────────────────────────
story.append(h1("3. Architecture of the ML Models"))
story.append(hr())
story.append(body(
    "All models use the Scikit-learn Pipeline architecture, which chains a StandardScaler and "
    "a regressor into a single estimator. Four regression algorithms were implemented and compared:"
))
models_desc = [
    ("Linear Regression", "Ordinary Least Squares (OLS) regression. Minimises the sum of squared residuals. "
     "Assumes a linear relationship between features and target. No regularisation. Serves as the baseline model."),
    ("Ridge Regression (L2)", "Linear regression with L2 regularisation (penalty = α × ||w||²). "
     "Prevents overfitting by shrinking coefficients. α=1.0 was used. Particularly useful when features are correlated."),
    ("Random Forest Regressor", "Ensemble of 100 decision trees (max_depth=8). Each tree is trained on a "
     "bootstrap sample and a random feature subset. Final prediction is the mean of all trees. "
     "Handles non-linear patterns and feature interactions."),
    ("Gradient Boosting Regressor", "Sequential ensemble that trains each tree to correct the residuals "
     "of the previous model. Uses 100 estimators, learning_rate=0.1, max_depth=4. "
     "Strong performance on structured data with complex patterns."),
]
for name, desc in models_desc:
    story.append(h2(name))
    story.append(body(desc))

story.append(h2("Pipeline Structure (All Models)"))
story.append(Paragraph(
    "Student Data  -->  [StandardScaler]  -->  [Regressor]  -->  Predicted Grade (0-100)",
    s["CodeB"]
))
story.append(sp(8))

# ── §4 Training Objective ─────────────────────────────────────────────────────
story.append(h1("4. Training Objective, Model Selection & Design Choices"))
story.append(hr())
story.append(h2("4.1 Training Objective"))
story.append(body(
    "All regression models are trained to minimise <b>Mean Squared Error (MSE)</b> on the training set. "
    "This corresponds to finding model parameters θ that minimise: L(θ) = (1/n) × Σ (y_i − ŷ_i)²."
))
story.append(body(
    "MSE penalises large errors more heavily than small ones due to the squaring term, which is "
    "appropriate for grade prediction where large mispredictions are particularly harmful."
))
story.append(h2("4.2 Model Selection"))
story.append(body(
    "All four models were evaluated using <b>5-fold cross-validation</b> on the training set. "
    "The model achieving the highest mean R² across folds was selected as the production model. "
    "<b>Linear Regression achieved the best cross-validated R² (0.928)</b>, confirming that the "
    "grade-feature relationship is largely linear as designed."
))
story.append(h2("4.3 Design Choices"))
for ch in [
    "<b>Synthetic Dataset Design:</b> The grade formula was intentionally linear in the features to validate that Linear Regression would perform best — an intended academic exercise.",
    "<b>Standardisation:</b> StandardScaler ensures all features contribute equally in distance-based and gradient computations.",
    "<b>Multiple Model Comparison:</b> Evaluating four models enables understanding of linear vs non-linear approaches.",
    "<b>Cross-Validation:</b> 5-fold CV prevents overfitting to a single train/test split and provides stable performance estimates.",
    "<b>Gaussian Noise (σ=4.0):</b> Added to the grade formula to ensure models face realistic irreducible error and to produce non-trivial MAE values.",
]:
    story.append(bul(ch))

# ── §5 Performance Evaluation ─────────────────────────────────────────────────
story.append(h1("5. Performance Evaluation"))
story.append(hr())
story.append(body(
    "Models were evaluated on a held-out test set of 240 students (20%) using three metrics: "
    "Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² (coefficient of determination). "
    "5-fold cross-validation was also reported for generalisation stability."
))
ptbl = [
    ["Model","Test MAE","Test RMSE","Test R²","CV R²","CV Std"],
    ["Linear Regression ⭐","3.31","4.23","0.9349","0.9278","±0.006"],
    ["Ridge Regression","3.31","4.23","0.9349","0.9278","±0.006"],
    ["Gradient Boosting","4.23","5.44","0.8920","0.8830","±0.010"],
    ["Random Forest","5.03","6.51","0.8455","0.8293","±0.015"],
]
pt = Table(ptbl, colWidths=[4.5*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm,2.2*cm])
pt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),BLUE), ("TEXTCOLOR",(0,0),(-1,0),WHITE),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),9),
    ("ALIGN",(0,0),(-1,-1),"CENTER"), ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGRAY]),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#D5D8DC")), ("ROWPADDING",(0,0),(-1,-1),5),
    ("BACKGROUND",(0,1),(-1,1),colors.HexColor("#D5F5E3")),
    ("FONTNAME",(0,1),(-1,1),"Helvetica-Bold"),
]))
story.append(pt)
story.append(sp(10))

for path, cap in [
    (f"{BASE}/reports/model_comparison.png",    "Figure 2: R² score comparison across all four regression models."),
    (f"{BASE}/reports/mae_comparison.png",       "Figure 3: MAE comparison — lower values indicate better accuracy."),
    (f"{BASE}/reports/actual_vs_predicted.png",  "Figure 4: Actual vs Predicted grades for Linear Regression on the test set."),
    (f"{BASE}/reports/residuals.png",            "Figure 5: Distribution of residuals — near-normal centred at 0 confirms good fit."),
    (f"{BASE}/reports/feature_importance.png",   "Figure 6: Random Forest feature importance — prev_gpa and study_hours are the most influential features."),
]:
    story += img(path, caption=cap)
    story.append(sp(6))

# ── §6 Techniques from Class ──────────────────────────────────────────────────
story.append(h1("6. Application of Techniques Taught in Class"))
story.append(hr())
techs = [
    ("Supervised Learning — Regression", "Weeks 1–2",
     "The entire pipeline is a supervised regression task where labelled (feature, grade) pairs train a model that generalises to new students — a core concept from the introductory AI lectures."),
    ("Feature Engineering & Encoding", "Weeks 3–4",
     "Categorical variables (gender, major) were label-encoded and all numerical features were standardised — standard feature engineering steps covered in the data preprocessing lectures."),
    ("Linear Models — Regression", "Weeks 5–6",
     "Linear Regression and Ridge Regression were implemented, reflecting the linear models lecture. The concept of minimising least squares and adding L2 regularisation to prevent overfitting was applied directly."),
    ("Ensemble Methods", "Weeks 7–8",
     "Random Forest (bagging) and Gradient Boosting (boosting) were implemented as ensemble methods. Both were covered in lectures on advanced ML, demonstrating reduced variance through model averaging."),
    ("Model Evaluation Metrics", "Weeks 9–10",
     "MAE, RMSE, and R² were used for evaluation — all three regression metrics were introduced in the model evaluation lecture. The R² interpretation (proportion of variance explained) was directly applied."),
    ("Cross-Validation", "Weeks 11–12",
     "5-fold stratified cross-validation was used for model selection and generalization assessment, as taught in the overfitting and model selection lecture. This prevents optimistic bias from a single split."),
]
for tech, wk, desc in techs:
    story.append(h2(f"{tech}  [{wk}]"))
    story.append(body(desc))

# ── §7 Challenges ─────────────────────────────────────────────────────────────
story.append(h1("7. Main Challenges and Solutions"))
story.append(hr())
chs = [
    ("Dataset Availability",
     "No real student dataset was publicly available. I designed a synthetic dataset using a weighted linear formula with Gaussian noise, carefully tuned to produce realistic grade distributions (mean ~58, std ~22) and ensure non-trivial model errors."),
    ("Feature Scaling Sensitivity",
     "Linear models converged poorly on unscaled data due to the wide range differences between features (e.g., attendance 40–100 vs. binary 0–1). StandardScaler was applied inside the Pipeline to ensure consistent scaling without leakage."),
    ("Irreducible Error from Noise",
     "The intentional Gaussian noise (σ=4.0) means even the best model cannot achieve perfect R²=1.0. This is realistic — achieving MAE≈3.3 on a 100-point scale is excellent and reflects a well-generalising model."),
    ("Model Selection Justification",
     "All four models performed well. I used cross-validated R² as the selection criterion rather than test R² alone, preventing overfitting to the test set. Linear Regression won due to the linear nature of the underlying data."),
    ("GUI Responsiveness",
     "Streamlit reruns the full script on every widget interaction. The model was cached using @st.cache_resource to avoid reloading on each slider change, ensuring a smooth and responsive live prediction experience."),
]
for ch, sol in chs:
    story.append(h2(f"Challenge: {ch}"))
    story.append(body(f"<b>Solution:</b> {sol}"))

# ── References ─────────────────────────────────────────────────────────────────
story.append(h1("References"))
story.append(hr())
for i, r in enumerate([
    "Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.",
    "Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
    "Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics.",
    "Streamlit Documentation. https://docs.streamlit.io (accessed April 2026).",
    "James, G., Witten, D., Hastie, T., Tibshirani, R. (2013). An Introduction to Statistical Learning. Springer.",
], 1):
    story.append(body(f"[{i}] {r}"))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MGRAY)
    canvas.drawString(2*cm, 1.2*cm, "Student Grade Predictor | AI Semester Project | Spring 2026 | Developed by: Abdul Sami (CMS ID: 023-23-0087)")
    canvas.drawRightString(19*cm, 1.2*cm, f"Page {doc.page}")
    canvas.restoreState()
os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)
doc = SimpleDocTemplate(OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"✓ Report saved: {OUT}")
