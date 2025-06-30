Below is a two-part pipeline for ACA, fully spelled out step-by-step with clear definitions of every concept and why we chose each approach.

Part A: Offline Data & Model Training Pipeline
mermaid
Copy
Edit
flowchart TD
  A[Raw Data Ingestion] --> B[ETL & Validation]
  B --> C[Feature Engineering]
  C --> D[Model Training]
  D --> E[Model Artifacts]
  E --> F[Parquet Store]
1. Raw Data Ingestion
What: Pull in your one‐file CSV/Excel export from CDHR (all PNOCs for the fiscal year)
Why: This single source has everything—issued/closed dates, critical flag, comment counts, centroid IDs, etc.

2. ETL & Validation
Trim & parse all column names and dates → uniform datetime64[ns].

Filter to the date range and projects you care about.

Classify PNOCs:

Closed if both R&C‐close and RM‐close dates exist.

In-flight otherwise.

Compute business-day durations using NumPy’s busday_count (skips weekends & holidays):

python
Copy
Edit
CI_Dur = busday_count(Date_Issued, Date_Closed_CI)
RM_Dur = busday_count(Date_Closed_CI, Date_Closed_RM)
Flag SLA adherence (static rules):

R&C must finish in ≤ 20 business-days

RM must finish in ≤ 10 business-days

3. Feature Engineering
Create the predictor variables (“features”) that feed the ML models:

BaselineDays (if you have it)

Meta fields: Critical vs Routine (Y/N), Project, Group, Branch

Lag counts: WorkingDays_Since_Request, CalendarDays_Since_Issue

Comment churn: RC_Comment_Count, RM_Comment_Count, comment-span days

Clustering: Dominant Centroid_ID (remark cluster) as a categorical

Optional text embeddings: TF-IDF vectors of all comments

All numeric features get median‐filled for missing; categoricals get an “Unknown” label.

4. Model Training
4.1 CatBoost Regressor → Dynamic Baselines
Why CatBoost?

Handles categorical features natively (no one-hot explosion)

Very strong on tabular data “out of the box”

Built-in support for quantile (P50/P90) loss

Provides fast SHAP explanations

P50 / P90:

P50 = 50th percentile (median) prediction of phase duration. Treat this as your data-driven baseline.

P90 = 90th percentile prediction, i.e. a “worst-case” buffer—90 % of similar PNOCs finish sooner.

4.2 CatBoost Regressor → Early-Warning Cushion
Retrain a simple regressor to predict days‐remaining until SLA break (SLA_days − days_elapsed).

Negative values mean “already late.” Use this live to flag at-risk PNOCs.

4.3 Prophet / SARIMAX → KPI Trend Forecasts
Fit on monthly aggregates of SLA % and BSA %

Output a 90-day forward projection for executive planning

4.4 SHAP → Driver Insights
Compute SHAP values on the P50 model

Translate top features into English bullets:

RC_Comment_Count adds ~3.2 days on average
Cluster = Electrical adds ~2.1 days

5. Model Artifacts & Storage
Save trained models (.cbm files) and feature metadata to S3 or your Parquet dataset

Archive historical runs for audit and monthly retraining

Part B: Online Application & Inference Pipeline
mermaid
Copy
Edit
flowchart TD
  U[User Uploads CSV/Excel] --> V[ETL & Validation]
  V --> W[Feature Build]
  W --> R[Load Models & Predict]
  R --> S[Compute KPIs & SHAP]
  S --> T[Render Streamlit UI]
  T -->|Download| D[CSV / Model Files]
1. User Upload
Single file containing all PNOC columns (same schema as training).

Sidebar filters let you pick date range, project(s), critical/routine, and cluster.

2. ETL & Validation
Same ETL as offline, but both closed and in-flight PNOCs flow through:

Closed → KPI display & charts

In-flight → days-remaining predictions & “High-risk” table

3. Feature Build
Apply identical transformations (median fill, label encode) as in training.

4. Load Models & Predict
Load CatBoost P50/P90 model → predict per-phase durations

Load days-remaining model → predict cushion for R&C and RM

Execute SHAP explainer to score feature impacts

Compute static SLA flags (CI_Dur ≤ 20, RM_Dur ≤ 10)

5. Compute KPIs & SHAP Insights
SLA Adherence = % Closed PNOCs meeting static SLA

Baseline Schedule Accuracy (BSA) = mean(1–|Actual–Baseline|/Baseline)

Comment-cycle KPIs = mean(RC_span_days), mean(RM_span_days)

Driver bullets from SHAP

6. Render Streamlit UI
Home page: KPI cards & trend charts

Combo‐charts: histograms of CI_Dur & RM_Dur with:

Black rule at static SLA (20/10)

Gold rule at historical mean

Blue rule at P50 model baseline

Scatter: Planned vs Actual, colored by cluster

Residual funnel: CI_Dur–Baseline vs Baseline

Cluster heat-map & stacked bars for open vs late

Driver Insight: bar chart + bullet list

Early-Warning: table of live PNOCs sorted by days-remaining

7. Downloads & Alerts
Export CSVs of metrics, predictions, SHAP values

Download model artifacts for offline audit

Slack/Teams integration: ping owners when days-remaining ≤ 0

Key Terminology
P50 / P90: percentiles of the model’s predictive distribution.

P50 (median) = 50 % of past PNOCs were faster, 50 % slower.

P90 = 90 % of past PNOCs were faster → use as a conservative buffer.

CatBoost: a gradient-boosting library that

natively encodes categorical vars

resists overfitting

supports quantile regression and SHAP explainability.

SLA Adherence: the fraction of PNOCs that finish on time relative to the fixed standard (20 days for R&C, 10 days for RM).

Baseline Schedule Accuracy (BSA):

BSA
=
1
−
∣
 
ActualDays
−
BaselineDays
 
∣
BaselineDays
BSA=1− 
BaselineDays
∣ActualDays−BaselineDays∣
​
 
averaged over PNOCs (a value closer to 1 means your baselines were accurate).

SHAP: an explanation method that assigns each feature a “credit” for pushing the model prediction up or down, enabling the bullet-list driver insight.

Why This Matters
Compliance via static SLA lines (20 d, 10 d) keeps you honest.

Optimization via P50/P90 baselines tightens schedules to data-driven targets.

Proactivity via days-remaining flags stops SLA breaches before they happen.

Transparency via SHAP bullets turns “machine learning” into clear process levers.

Scalability: one file in → entire ACA dashboard auto-refreshes.

With this pipeline in place, your team goes from manual spreadsheets to a fully automated, ML-powered control tower—driving both precision and continuous improvement in every PNOC.
