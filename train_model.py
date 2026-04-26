"""
Student Grade Predictor — Training Script
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
warnings.filterwarnings('ignore')

BASE = "H:/AI project/files"
np.random.seed(42)

# ─── 1. DATASET GENERATION ─────────────────────────────────────────────────
def generate_dataset(n=1200):
    rng = np.random.default_rng(42)

    study_hours     = rng.uniform(0, 10, n)
    attendance_pct  = rng.uniform(40, 100, n)
    prev_gpa        = rng.uniform(1.5, 4.0, n)
    assignments_done= rng.integers(0, 11, n)   # out of 10
    sleep_hours     = rng.uniform(4, 10, n)
    part_time_job   = rng.choice([0, 1], n, p=[0.65, 0.35])
    internet_access = rng.choice([0, 1], n, p=[0.20, 0.80])
    tutoring        = rng.choice([0, 1], n, p=[0.70, 0.30])
    family_support  = rng.integers(1, 6, n)   # 1–5 scale
    stress_level    = rng.integers(1, 6, n)   # 1–5 (5=high stress)
    gender          = rng.choice(["Male","Female"], n)
    major           = rng.choice(["CS","EE","BBA","Mathematics","B.Ed"], n)

    # Grade formula with realistic noise
    grade = (
        study_hours     * 2.8
        + (attendance_pct - 40) * 0.25
        + prev_gpa      * 12.0
        + assignments_done * 1.5
        + sleep_hours   * 0.8
        - part_time_job * 4.5
        + internet_access * 2.0
        + tutoring      * 3.5
        + family_support * 1.2
        - stress_level  * 1.8
        + rng.normal(0, 4, n)   # noise
    )
    # Normalise to 0–100
    grade = np.clip((grade - grade.min()) / (grade.max() - grade.min()) * 100, 0, 100)
    grade = np.round(grade, 1)

    df = pd.DataFrame({
        "study_hours":      np.round(study_hours, 1),
        "attendance_pct":   np.round(attendance_pct, 1),
        "prev_gpa":         np.round(prev_gpa, 2),
        "assignments_done": assignments_done,
        "sleep_hours":      np.round(sleep_hours, 1),
        "part_time_job":    part_time_job,
        "internet_access":  internet_access,
        "tutoring":         tutoring,
        "family_support":   family_support,
        "stress_level":     stress_level,
        "gender":           gender,
        "major":            major,
        "final_grade":      grade,
    })
    return df

# ─── 2. PRE-PROCESSING ────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    # Encode categoricals
    df["gender_enc"] = (df["gender"] == "Female").astype(int)
    major_map = {"CS":0,"EE":1,"BBA":2,"Mathematics":3,"Physics":4}
    df["major_enc"] = df["major"].map(major_map)
    features = [
        "study_hours","attendance_pct","prev_gpa","assignments_done",
        "sleep_hours","part_time_job","internet_access","tutoring",
        "family_support","stress_level","gender_enc","major_enc"
    ]
    return df[features], df["final_grade"]

# ─── 3. MODELS ────────────────────────────────────────────────────────────────
def build_models():
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)),
        ]),
    }

# ─── 4. EVALUATION + PLOTS ───────────────────────────────────────────────────
MC = ["#3498DB","#9B59B6","#1ABC9C","#E67E22"]

def evaluate(models, Xtr, Xte, ytr, yte, out):
    os.makedirs(out, exist_ok=True)
    res = {}
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        yp  = pipe.predict(Xte)
        mae = mean_absolute_error(yte, yp)
        rmse= np.sqrt(mean_squared_error(yte, yp))
        r2  = r2_score(yte, yp)
        cv  = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="r2")
        res[name] = dict(pipeline=pipe, mae=mae, rmse=rmse, r2=r2,
                         cv_mean=cv.mean(), cv_std=cv.std(), yp=yp)
        print(f"  [{name:20s}]  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  CV_R2={cv.mean():.4f}±{cv.std():.4f}")

    names = list(res.keys())

    # Plot 1 — R2 comparison
    r2s = [res[n]["r2"] for n in names]
    fig,ax = plt.subplots(figsize=(9,4))
    bars = ax.bar(names, r2s, color=MC, edgecolor="white", linewidth=1.5, width=0.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Model Comparison — R² Score", fontsize=14, fontweight="bold")
    for b,v in zip(bars,r2s):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out}/model_comparison.png", dpi=150); plt.close()

    # Plot 2 — MAE comparison
    maes = [res[n]["mae"] for n in names]
    fig,ax = plt.subplots(figsize=(9,4))
    bars = ax.bar(names, maes, color=MC, edgecolor="white", linewidth=1.5, width=0.5)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax.set_title("Model Comparison — MAE (lower is better)", fontsize=14, fontweight="bold")
    for b,v in zip(bars,maes):
        ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out}/mae_comparison.png", dpi=150); plt.close()

    # Best model
    best = max(res, key=lambda n: res[n]["r2"])

    # Plot 3 — Actual vs Predicted
    fig,ax = plt.subplots(figsize=(7,6))
    ax.scatter(yte, res[best]["yp"], alpha=0.4, color="#3498DB", edgecolors="white", linewidth=0.3, s=25)
    mn,mx = yte.min(), yte.max()
    ax.plot([mn,mx],[mn,mx],"r--",linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Grade", fontsize=12); ax.set_ylabel("Predicted Grade", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {best}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out}/actual_vs_predicted.png", dpi=150); plt.close()

    # Plot 4 — Residuals
    residuals = yte.values - res[best]["yp"]
    fig,ax = plt.subplots(figsize=(8,4))
    ax.hist(residuals, bins=30, color="#1ABC9C", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#E74C3C", linewidth=1.5, linestyle="--", label="Zero error")
    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=12); ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Residual Distribution — {best}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out}/residuals.png", dpi=150); plt.close()

    # Plot 5 — Feature Importance (Random Forest)
    rf_pipe = res["Random Forest"]["pipeline"]
    rf_model = rf_pipe.named_steps["reg"]
    feat_names = [
        "Study Hours","Attendance %","Prev GPA","Assignments",
        "Sleep Hours","Part-time Job","Internet","Tutoring",
        "Family Support","Stress Level","Gender","Major"
    ]
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)
    fig,ax = plt.subplots(figsize=(8,6))
    colors_fi = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(idx)))
    ax.barh([feat_names[i] for i in idx], importances[idx], color=colors_fi, edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out}/feature_importance.png", dpi=150); plt.close()

    # Plot 6 — Correlation heatmap
    df_raw = generate_dataset()
    X_all, y_all = preprocess(df_raw)
    corr_df = X_all.copy()
    corr_df.columns = feat_names
    corr_df["Final Grade"] = y_all.values
    fig,ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.5, annot_kws={"size":7})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{out}/correlation_heatmap.png", dpi=150); plt.close()

    return res, best

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  Student Grade Predictor — AI Semester Project")
    print("="*55)

    df = generate_dataset(1200)
    df.to_csv(f"{BASE}/data/student_dataset.csv", index=False)
    print(f"\nDataset: {len(df)} students | Grade range: {df['final_grade'].min():.1f}–{df['final_grade'].max():.1f}\n")

    X, y = preprocess(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training models...")
    res, best = evaluate(build_models(), Xtr, Xte, ytr, yte, f"{BASE}/reports")

    joblib.dump(res[best]["pipeline"], f"{BASE}/models/best_model.pkl")
    joblib.dump({"best_model_name": best,
                 "results": {n: {"mae":r["mae"],"rmse":r["rmse"],"r2":r["r2"],"cv_mean":r["cv_mean"],"cv_std":r["cv_std"]}
                              for n,r in res.items()}},
                f"{BASE}/models/metrics.pkl")

    pd.DataFrame([{"Model":n,"MAE":round(r["mae"],2),"RMSE":round(r["rmse"],2),
                   "R2":round(r["r2"],4),"CV R2":round(r["cv_mean"],4),"CV Std":round(r["cv_std"],4)}
                  for n,r in res.items()]).to_csv(f"{BASE}/reports/metrics_summary.csv", index=False)
    print(f"\n✓ Best model: {best}  (R2={res[best]['r2']:.4f})")
    print("✓ All artifacts saved.")
