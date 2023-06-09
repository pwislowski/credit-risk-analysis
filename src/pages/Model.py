from typing import Dict

import pandas as pd
import streamlit as st

# local libs
from data import process_data, get_cleaned_data
import model
from model import ModelPerformance

st.set_page_config(page_title="XGBClassifier ML Model", layout="wide")

st.title("Model Page")
st.header("Metrics")
st.divider()


# * model init

LABEL_MAPPING = model.LABEL_MAPPING
SCALER_MAPPING = model.SCALER_MAPPING


df_clean = process_data(get_cleaned_data("../credit_customers.csv"))
df = process_data(get_cleaned_data("../credit_customers.csv"))

# keys: ["X_train", "X_test", "y_train", "y_test"]
DF: Dict[str, pd.DataFrame] = model.process_df(df)

MODEL = model.xgbclassifier_model(DF["X_train"], DF["y_train"])

y_pred = MODEL.predict(DF["X_test"])

STATS = ModelPerformance(MODEL, y_pred, **DF)

# * dashboard set-up

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Accuracy", value=STATS.fmt_float_to_str(STATS.accuracy))

with col2:
    st.metric(label="Recall", value=STATS.fmt_float_to_str(STATS.recall))

with col3:
    st.metric(label="F1-Score", value=STATS.fmt_float_to_str(STATS.f1_score))

with col4:
    st.metric(label="Train AUC", value=STATS.fmt_float_to_str(STATS.train_auc))

with col5:
    st.metric(label="Test AUC", value=STATS.fmt_float_to_str(STATS.test_auc))

st.divider()

st.header("Visualization")
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curve")
    st.plotly_chart(
        model.plot_roc(MODEL, DF["X_test"], DF["y_test"]),
        use_container_width=True,
    )

with col2:
    st.subheader("Feature Importance")
    st.plotly_chart(
        model.plot_feature_importance(MODEL, DF["X_train"]),
        use_container_width=True,
    )

st.divider()

# * model forecast
st.header("Assess Creditor")

col1, col2 = st.columns(2)

with col1:
    with st.form("Assess Creditor"):
        fcheck_status = st.selectbox(
            label="Checking Status", options=df_clean["checking_status"].unique()
        )
        finstall_commit = st.selectbox(
            label="Installment Commitment",
            options=sorted(df_clean["installment_commitment"].unique()),
        )
        fexisting_credits = st.selectbox(
            label="Existing Credits",
            options=sorted(df_clean["existing_credits"].unique()),
        )
        fother_parties = st.selectbox(
            label="Other Parties", options=df_clean["other_parties"].unique()
        )
        fother_pay_plans = st.selectbox(
            label="Other Payment Plans",
            options=df_clean["other_payment_plans"].unique(),
        )
        fown_telephone = st.selectbox(
            label="Own Telephone", options=df_clean["own_telephone"].unique()
        )
        fduration = st.selectbox(
            label="Credit Duration", options=sorted(df_clean["duration"].unique())
        )
        fjob = st.selectbox(label="Job Type", options=df_clean["job"].unique())
        fsaving_status = st.selectbox(
            label="Savings Status", options=df_clean["savings_status"].unique()
        )
        fpurpose = st.selectbox(
            label="Purpose", options=sorted(df_clean["purpose"].unique())
        )
        femployment = st.selectbox(
            label="Employment Duration", options=df_clean["employment"].unique()
        )
        fresidence_since = st.selectbox(
            label="Residence Since",
            options=sorted(df_clean["residence_since"].unique()),
        )
        fage_agg = st.selectbox(
            label="Age Bracket", options=sorted(df_clean["age_agg"].unique())
        )
        fhousing = st.selectbox(label="Housing", options=df_clean["housing"].unique())
        fcredit_history = st.selectbox(
            label="Credit History", options=df_clean["credit_history"].unique()
        )
        fcredit_amount = st.number_input(label="Credit Amount", step=100)
        fforeign_worker = st.selectbox(
            label="Foreign Worker", options=df_clean["foreign_worker"].unique()
        )
        fproperty_magintude = st.selectbox(
            label="Collateral", options=df_clean["property_magnitude"].unique()
        )
        fsex = st.selectbox(label="Sex", options=df_clean["sex"].unique())
        fmartial = st.selectbox(
            label="Martial Status", options=df_clean["martial"].unique()
        )
        fcredit_util: float = fcredit_amount / fexisting_credits
        fdit: float = fcredit_amount / finstall_commit

        submitted = st.form_submit_button("Predict")

        if submitted:
            form_data = {
                "checking_status": fcheck_status,
                "duration": fduration,
                "credit_history": fcredit_history,
                "purpose": fpurpose,
                "credit_amount": fcredit_amount,
                "savings_status": fsaving_status,
                "employment": femployment,
                "installment_commitment": finstall_commit,
                "other_parties": fother_parties,
                "residence_since": fresidence_since,
                "property_magnitude": fproperty_magintude,
                "other_payment_plans": fother_pay_plans,
                "housing": fhousing,
                "existing_credits": fexisting_credits,
                "job": fjob,
                "own_telephone": fown_telephone,
                "foreign_worker": fforeign_worker,
                "sex": fsex,
                "martial": fmartial,
                "dti": fdit,
                "age_agg": fage_agg,
                "credit_util": fcredit_util,
            }

            with col2:
                transformed = model.transform_for_pred(
                    SCALER_MAPPING, LABEL_MAPPING, form_data
                )
                local_pred = MODEL.predict(transformed)

                local_pred_proba = MODEL.predict_proba(transformed)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Creditor Assessment",
                        value="Good" if local_pred else "Bad",
                    )

                with col2:
                    st.metric(
                        label= 'Model\'s Confidence',
                        value= f'{local_pred_proba[0][1]:.2%}'
                    )
