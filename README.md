# Student Grade Predictor

**[ Try the Live Demo Here](https://student-grade-predictor-r3bcze4j7jgqu5trnda3io.streamlit.app/)**

---
### Project Structure
```
grade_predictor/
├── app.py                   ← Streamlit GUI (run this)
├── requirements.txt
├── src/
│   ├── train_model.py       ← ML training + evaluation + plots
│   └── generate_report.py   ← PDF report generator
├── data/
│   └── student_dataset.csv  ← 1,200 student records
├── models/
│   ├── best_model.pkl       ← Trained Linear Regression pipeline
│   └── metrics.pkl          ← Model metrics summary
└── reports/
    ├── Project_Report.pdf   ← Full 8-page project report
    ├── model_comparison.png
    ├── mae_comparison.png
    ├── actual_vs_predicted.png
    ├── residuals.png
    ├── feature_importance.png
    └── correlation_heatmap.png
```

### How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (generates all plots + saves model)
python src/train_model.py

# 3. Launch GUI
streamlit run app.py

# 4. (Optional) Regenerate PDF report
python src/generate_report.py
```

### Model Performance
| Model              | R²    | MAE  | RMSE |
|--------------------|-------|------|------|
| Linear Regression  | 0.935 | 3.31 | 4.23 |
| Ridge Regression   | 0.935 | 3.31 | 4.23 |
| Gradient Boosting  | 0.892 | 4.23 | 5.44 |
| Random Forest      | 0.846 | 5.03 | 6.51 |

Best model: **Linear Regression** (R²=0.935, CV R²=0.928)
