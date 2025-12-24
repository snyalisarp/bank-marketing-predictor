import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        
        X_temp = X.copy()
        X_temp = self._do_transform(X_temp)
        self.feature_names_out_ = X_temp.columns.tolist()
        return self
    
    def transform(self, X):
        return self._do_transform(X)

    def _do_transform(self, X):
        X = X.copy()
        yes_no_map = {'no': 0, 'yes': 1}
        
        
        for col in ['default', 'housing', 'loan']:
            X[col] = X[col].map(yes_no_map)
            
        
        X['is_non_negative_balance'] = (X['balance'] >= 0).astype(int)
        X['new_client'] = (X['pdays'] == -1).astype(int)
        
        
        month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 
                     'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        
        m_idx = X['month'].map(month_map)
        X['month_sin'] = np.sin(2 * np.pi * m_idx / 12)
        X['month_cos'] = np.cos(2 * np.pi * m_idx / 12)
        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 31)
        
        
        cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
        for col in cat_cols:
            X[col] = X[col].astype('category')
            
        return X.drop(['month', 'day'], axis=1)

    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)



@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

st.set_page_config(
    page_title="Bank Telemarketing Prediction",
    layout="centered"
)

st.title("📞 Bank Telemarketing")
st.write("Müşterinin vadeli mevduata **abone olma olasılığını** tahmin eder.")


st.header("👤 Müşteri Bilgileri")

age = st.number_input("Age", min_value=18, max_value=100, value=35)

job = st.selectbox(
    "Job",
    [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed",
        "services", "student", "technician", "unemployed", "unknown"
    ]
)

marital = st.selectbox("Marital Status", ["single", "married", "divorced"])

education = st.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"]
)

default = st.selectbox("Has Credit in Default?", ["yes", "no"])
balance = st.number_input("Account Balance", value=0)

housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])

contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])

day = st.slider("Last Contact Day of Month", 1, 31, 15)

month = st.selectbox(
    "Month",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)

campaign = st.number_input("Number of Contacts (This Campaign)", min_value=1, value=1)
pdays = st.number_input("Days Since Last Contact (-1 = never)", value=-1)
previous = st.number_input("Previous Contacts", min_value=0, value=0)

poutcome = st.selectbox(
    "Outcome of Previous Campaign",
    ["success", "failure", "other", "unknown"]
)


if st.button("🔮 Tahmin Et"):

    input_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    prediction = model.predict(input_df)[0]

    
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
        st.metric(
            label="Abonelik Olasılığı",
            value=f"%{probability * 100:.2f}"
        )

    if prediction in [1, "yes", True]:
        st.success("✅ Müşteri **ABONE OLABİLİR**")
    else:
        st.error("❌ Müşteri **ABONE OLMAYABİLİR**")

    st.subheader("📊 Girilen Veriler")
    st.dataframe(input_df)