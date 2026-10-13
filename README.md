# Student Risk Advisor — Cohort App

A Streamlit research prototype that scores a CSV cohort, ranks records by estimated at-risk probability, flags a top-risk share (10%, 20%, or 30%), and exports the ranked list.

## Included files

- `app.py` — Streamlit app entry point
- `requirements.txt` — pinned Python dependencies
- `student_risk_pipeline.joblib` — fitted preprocessing + Random Forest pipeline used for inference
- `cohort_template.csv` — blank CSV header with all 30 required predictors
- `sample_cohort.csv` — small example cohort without G1/G2/G3
- `student-por.csv` — original public UCI dataset, included to make the model rebuild reproducible
- `build_model.py` — rebuilds the inference artifact from the dataset

## Deploy on Streamlit Community Cloud

1. Download and extract this ZIP.
2. Create a **new public GitHub repository** named `student-risk-advisor` (or another name you prefer).
3. Upload the files and folders from this extracted directory to the repository root. Do not upload the ZIP file itself as the only repository file.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
5. Select **Create app** and choose your new repository, branch `main`, and entrypoint file `app.py`.
6. Optionally choose an app subdomain, then click **Deploy**.
7. When deployment finishes, open the generated URL and test with `sample_cohort.csv`.

The new cohort app has its own URL and does not replace or modify the existing single-student app.

## Test locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the model (optional)

```bash
python build_model.py
```

The included model artifact is already built; rebuilding is not required for deployment.

## CSV format

The uploaded file must contain these columns exactly:

`school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences`

An optional `student_id` column is supported. `G1`, `G2`, and `G3` are excluded from inputs and removed from exported records.

## Responsible-use note

This is an educational research prototype, not a validated institutional decision system. Probabilities are estimates, not guarantees. Use only to support human-reviewed, supportive outreach; never for punishment or automated high-stakes decisions. The inference artifact is refitted on all 649 records for demonstration, so its predictions are not the held-out test predictions reported in the paper. Validate externally before any real use.

Dataset: Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T (CC BY 4.0).
