# Student Academic Performance Analytics — Future Enhancements Demo

## Added functionality
- Random Forest regression demo to estimate final marks from academic inputs.
- Test-set MAE and R² metrics for a basic demonstration.
- Early-warning flag based on predicted marks and attendance thresholds.
- Rule-based personalized study recommendations.
- Recommendation CSV export.
- Semester field in the sample data for future progress-tracking work.

## Run in VS Code (Windows)
1. Extract this ZIP.
2. Open the `student_performance_analytics_future_enhancements` folder in VS Code.
3. Open Terminal > New Terminal.
4. Install packages:
   ```powershell
   py -m pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   py -m streamlit run app.py
   ```
6. Open the local URL shown, usually `http://localhost:8501`.

If `py` is not recognized, use `python` instead.

## CSV columns required
Gender, Attendance, Study_Hours, Assignment_Score, Internal_Marks, Previous_Marks, Final_Marks, Extracurricular.
Age, Student_ID, and Semester are optional.

## Important limitations
The included CSV is synthetic demo data. The model estimates are educational examples, not validated predictions for real students. The early-warning rules are simple thresholds and should be reviewed by educators. PDF report generation, authentication/roles, and cloud deployment are not implemented in this version.
