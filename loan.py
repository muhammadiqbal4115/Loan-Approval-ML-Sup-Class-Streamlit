######################################################################################################
# Importing Libraries
######################################################################################################
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

######################################################################################################
# Streamlit Page Setup
######################################################################################################
st.set_page_config(page_title="💵 LOAN PREDICTION", layout="wide", initial_sidebar_state="expanded")

######################################################################################################
# NEON RED THEME — Times New Roman — Global CSS Injection
######################################################################################################
st.markdown("""
<style>

/* ── Hide ALL Streamlit Chrome ───────────────────────────────── */
#MainMenu                            { visibility: hidden !important; }
footer                               { visibility: hidden !important; }
header                               { visibility: hidden !important; }
.stDeployButton                      { display: none !important; }
[data-testid="stToolbar"]            { display: none !important; }
[data-testid="stDecoration"]         { display: none !important; }
[data-testid="stStatusWidget"]       { display: none !important; }
[data-testid="manage-app-button"]    { display: none !important; }
.viewerBadge_container__1QSob        { display: none !important; }
.styles_viewerBadge__1yB5_           { display: none !important; }
button[title="View fullscreen"]      { display: none !important; }
button[title="Download"]             { display: none !important; }

/* ── Root Variables ───────────────────────────────────────────── */
:root {
    --red:        #FF0033;
    --red-dim:    #CC0022;
    --red-deep:   #990020;
    --red-glow:   rgba(255, 0, 51, 0.5);
    --red-faint:  rgba(255, 0, 51, 0.07);
    --red-border: rgba(255, 0, 51, 0.28);
    --bg:         #07090B;
    --bg2:        #0C0F12;
    --bg3:        #121619;
    --text:       #EEF0F2;
    --text-dim:   #6E7A85;
}

/* ── Global Font: Times New Roman ────────────────────────────── */
*, *::before, *::after,
html, body,
.stApp, .stMarkdown, .stText,
p, h1, h2, h3, h4, h5, h6,
label, span, div, input, select, textarea,
button, .stButton > button,
[class*="st-"], [data-testid] {
    font-family: 'Times New Roman', Times, serif !important;
}

/* ── App Background — subtle red grid ────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #07090B !important;
    background-image:
        repeating-linear-gradient(0deg,
            transparent, transparent 39px,
            rgba(255,0,51,0.04) 39px, rgba(255,0,51,0.04) 40px),
        repeating-linear-gradient(90deg,
            transparent, transparent 39px,
            rgba(255,0,51,0.04) 39px, rgba(255,0,51,0.04) 40px) !important;
}

/* ── Main block ───────────────────────────────────────────────── */
[data-testid="stMain"],
.main .block-container {
    background: transparent !important;
    padding-top: 1.5rem !important;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0C0F12 !important;
    border-right: 1px solid rgba(255,0,51,0.28) !important;
    box-shadow: 4px 0 30px rgba(255,0,51,0.12) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Times New Roman', Times, serif !important;
    color: #EEF0F2 !important;
}

[data-testid="stSidebar"] h2 {
    color: #FF0033 !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-variant: small-caps !important;
}

/* ── Page Title ───────────────────────────────────────────────── */
.stApp h1 {
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    color: #FF0033 !important;
    text-shadow: 0 0 18px rgba(255,0,51,0.5), 0 0 40px rgba(255,0,51,0.2) !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid rgba(255,0,51,0.28) !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 0.2rem !important;
}

/* ── Subheaders ───────────────────────────────────────────────── */
.stApp h2, .stApp h3 {
    font-family: 'Times New Roman', Times, serif !important;
    color: #FF0033 !important;
    font-variant: small-caps !important;
    letter-spacing: 0.05em !important;
    text-shadow: 0 0 10px rgba(255,0,51,0.3) !important;
}

/* ── Caption ──────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"],
.stCaption {
    color: #6E7A85 !important;
    font-style: italic !important;
}

/* ── General text ─────────────────────────────────────────────── */
.stApp p, .stApp li, .stApp span {
    color: #EEF0F2 !important;
    font-family: 'Times New Roman', Times, serif !important;
}

/* ── Input Labels ─────────────────────────────────────────────── */
.stApp label {
    color: #6E7A85 !important;
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Text / Number Inputs ─────────────────────────────────────── */
.stTextInput input,
.stNumberInput input {
    font-family: 'Times New Roman', Times, serif !important;
    background: #121619 !important;
    border: 1px solid rgba(255,0,51,0.28) !important;
    border-radius: 3px !important;
    color: #EEF0F2 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #FF0033 !important;
    box-shadow: 0 0 0 2px rgba(255,0,51,0.2), 0 0 12px rgba(255,0,51,0.15) !important;
    outline: none !important;
}

/* ── Selectbox ────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #121619 !important;
    border: 1px solid rgba(255,0,51,0.28) !important;
    border-radius: 3px !important;
    color: #EEF0F2 !important;
    font-family: 'Times New Roman', Times, serif !important;
}

/* ── Slider thumb + fill ──────────────────────────────────────── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #FF0033 !important;
    box-shadow: 0 0 10px rgba(255,0,51,0.5) !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #EEF0F2 !important;
    background: transparent !important;
    border: 1.5px solid #FF0033 !important;
    border-radius: 3px !important;
    padding: 0.5rem 1.8rem !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 0 12px rgba(255,0,51,0.2) !important;
}

.stButton > button:hover {
    color: #fff !important;
    background: #FF0033 !important;
    box-shadow: 0 0 24px rgba(255,0,51,0.5), 0 0 50px rgba(255,0,51,0.2) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Dataframes ───────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,0,51,0.28) !important;
    border-radius: 4px !important;
    box-shadow: 0 0 20px rgba(255,0,51,0.08) !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] * {
    font-family: 'Times New Roman', Times, serif !important;
    color: #EEF0F2 !important;
}

[data-testid="stDataFrame"] th {
    background: rgba(255,0,51,0.07) !important;
    color: #FF0033 !important;
    border-bottom: 1px solid rgba(255,0,51,0.28) !important;
    font-variant: small-caps !important;
    letter-spacing: 0.04em !important;
}

/* ── Metric cards ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0C0F12 !important;
    border: 1px solid rgba(255,0,51,0.28) !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 0 18px rgba(255,0,51,0.07) !important;
}

[data-testid="stMetricLabel"] {
    color: #6E7A85 !important;
    font-variant: small-caps !important;
    letter-spacing: 0.08em !important;
    font-family: 'Times New Roman', Times, serif !important;
}

[data-testid="stMetricValue"] {
    color: #FF0033 !important;
    font-size: 1.8rem !important;
    text-shadow: 0 0 12px rgba(255,0,51,0.4) !important;
    font-family: 'Times New Roman', Times, serif !important;
}

/* ── Success / Error banners ──────────────────────────────────── */
.stSuccess > div {
    background: rgba(0,255,80,0.06) !important;
    border-left: 4px solid #00FF50 !important;
    color: #00FF50 !important;
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 0 20px rgba(0,255,80,0.12) !important;
    border-radius: 4px !important;
}

.stError > div {
    background: rgba(255,0,51,0.07) !important;
    border-left: 4px solid #FF0033 !important;
    color: #FF0033 !important;
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 0 20px rgba(255,0,51,0.15) !important;
    border-radius: 4px !important;
}

/* ── Divider ──────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,0,51,0.28) !important;
    box-shadow: 0 0 8px rgba(255,0,51,0.2) !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbars ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0C0F12; }
::-webkit-scrollbar-thumb { background: #CC0022; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #FF0033; }

/* ── Number input +/- buttons ─────────────────────────────────── */
.stNumberInput button {
    background: #121619 !important;
    border-color: rgba(255,0,51,0.28) !important;
    color: #FF0033 !important;
}
.stNumberInput button:hover {
    background: rgba(255,0,51,0.07) !important;
    border-color: #FF0033 !important;
}

/* ── Sidebar success badge ────────────────────────────────────── */
[data-testid="stSidebar"] .stSuccess > div {
    background: rgba(255,0,51,0.06) !important;
    border-left: 3px solid #FF0033 !important;
    color: #FF0033 !important;
    font-family: 'Times New Roman', Times, serif !important;
    border-radius: 3px !important;
}

</style>
""", unsafe_allow_html=True)

######################################################################################################
# Data Loading (cached)
######################################################################################################
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


######################################################################################################
# Model Training (cached resource)
######################################################################################################
@st.cache_resource
def train_model(df: pd.DataFrame):
    target = "approved"
    drop_cols = [target]
    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")

    X = df.drop(columns=drop_cols)
    y = df[target]

    cat_cols = [c for c in ["gender", "city", "employment_type", "bank"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return clf, metrics, X_train.columns.tolist()


######################################################################################################
# Sidebar — (1) Load Dataset
######################################################################################################
st.sidebar.header("(1) Load Dataset")

csv_path = st.sidebar.text_input(
    "CSV Path",
    value="loan_dataset.csv",
    help="Path to the dataset CSV. If running from same folder, keep as-is."
)

try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

st.sidebar.success(f"✔ Loaded {len(df):,} rows")


######################################################################################################
# Sidebar — (2) Train Model
######################################################################################################
st.sidebar.header("(2) Train Model")
train_now = st.sidebar.button("⚡ Train / Re-Train")

if train_now:
    st.cache_resource.clear()

clf, metrics, feature_order = train_model(df)


######################################################################################################
# Page Title
######################################################################################################
st.title("💵 Loan Approval Prediction")
st.caption("Machine Learning Classification Project · Logistic Regression · For Practice Purposes Only")


######################################################################################################
# Main Layout — Data Preview + Metrics
######################################################################################################
colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

with colB:
    st.subheader("Model Metrics — Holdout Test Set")

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    m1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
    m2.metric("Precision", f"{metrics['precision']:.4f}")
    m3.metric("Recall",    f"{metrics['recall']:.4f}")
    m4.metric("F1 Score",  f"{metrics['f1']:.4f}")

    st.caption("Confusion Matrix — rows: actual [0,1] · cols: predicted [0,1]")
    cm = np.array(metrics["confusion_matrix"])
    st.dataframe(
        pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]),
        use_container_width=True
    )

st.divider()


######################################################################################################
# Try a Prediction
######################################################################################################
st.subheader("Try a Prediction")

c1, c2, c3, c4 = st.columns(4)

with c1:
    applicant_name  = st.text_input("Applicant Name", value="Muhammad Ali")
    gender          = st.selectbox("Gender", ["M", "F"], index=0)
    age             = st.slider("Age", 21, 60, 30)

with c2:
    city            = st.selectbox("City", sorted(df["city"].unique().tolist()))
    employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique().tolist()))
    bank            = st.selectbox("Bank", sorted(df["bank"].unique().tolist()))

with c3:
    monthly_income_pkr = st.number_input("Monthly Income (PKR)", min_value=1500, max_value=500000, value=120000, step=1000)
    credit_score       = st.slider("Credit Score", 300, 900, 680)

with c4:
    loan_amount_pkr    = st.number_input("Loan Amount (PKR)", min_value=50000, max_value=3500000, value=800000, step=5000)
    loan_tenure_months = st.selectbox("Tenure (months)", [6, 12, 18, 24, 36, 48, 60], index=3)
    existing_loans     = st.selectbox("Existing Loans", [0, 1, 2, 3], index=0)
    default_history    = st.selectbox("Default History", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", index=0)
    has_credit_card    = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", index=0)


######################################################################################################
# Build Input Row + Prediction
######################################################################################################
input_row = pd.DataFrame([{
    "gender"             : gender,
    "age"                : age,
    "city"               : city,
    "employment_type"    : employment_type,
    "bank"               : bank,
    "monthly_income_pkr" : monthly_income_pkr,
    "credit_score"       : credit_score,
    "loan_amount_pkr"    : loan_amount_pkr,
    "loan_tenure_months" : loan_tenure_months,
    "existing_loans"     : existing_loans,
    "default_history"    : default_history,
    "has_credit_card"    : has_credit_card
}])

input_row = input_row[feature_order]

st.write("")
if st.button("🔍 Predict Approval"):
    prob = float(clf.predict_proba(input_row)[:, 1][0])
    pred = int(prob >= 0.5)

    if pred == 1:
        st.success(f"✅  {applicant_name} — APPROVED  ·  Probability: {prob:.2%}")
    else:
        st.error(f"❌  {applicant_name} — REJECTED  ·  Probability: {prob:.2%}")
