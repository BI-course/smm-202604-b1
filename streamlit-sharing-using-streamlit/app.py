import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------
# Load all four persisted artifacts ONCE, at module import time. Streamlit
# re-runs this whole script top-to-bottom on every widget interaction, so
# @st.cache_resource keeps joblib.load() from re-reading the files from
# disk on every single click -- without it, every tab switch and every
# form submission would reload all four artifacts again.
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    MODEL_DIR = "./model"
    regression_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "20261020_084339_ridge_regressor_for_exam_score.joblib")
    )
    classification_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "20261020_084942_logisticregression_tuned_classifier_for_insurance_claim_fraud.joblib")
    )
    clustering_artifact = joblib.load(
        os.path.join(MODEL_DIR, "20261020_085241_kmeans_bank_wealth_segmentation.joblib")
    )
    arm_artifact = joblib.load(
        os.path.join(MODEL_DIR, "pharmacy_recommender.joblib")
    )
    return regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact


regression_pipeline, classification_pipeline, clustering_artifact, arm_artifact = load_artifacts()

clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
# The 5 columns the clustering model was trained on, in the order it expects
clustering_features = clustering_artifact["features"]

recommendation_lookup = arm_artifact["rules"]
sku_lookup = arm_artifact["sku_lookup"]

st.set_page_config(page_title="Group B Model Validation Demos", page_icon="\U0001F4CA", layout="centered")
st.title("Group B -- Model Validation Demos")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "Student Exam Score Regressor", "Insurance Claim Fraud Classifier",
    "Bank Customer Segmenter", "Pharmacy Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression
# -----------------------------------------------------------------------
with tab_regression:
    with st.form("regression_form"):
        # These categories are the exact set the pipeline's encoders
        # were fitted on.
        major_subject = st.selectbox(
            "Major Subject", ["Business", "Engineering", "Humanities", "Sciences"], index=1)
        parental_education_level = st.selectbox(
            "Parental Education Level", ["High School", "Bachelor's", "Master's", "Doctorate"], index=1)
        internet_access_quality = st.selectbox(
            "Internet Access Quality", ["Poor", "Average", "Good", "Excellent"], index=2)
        part_time_job = st.selectbox("Part-time Job", ["No", "Yes"], index=0)
        study_group_participation = st.selectbox("Study Group Participation", ["No", "Yes"], index=1)
        study_hours_per_week = st.number_input("Study Hours per Week", value=15.0)
        attendance_rate_pct = st.number_input("Attendance Rate (%)", value=90.0)
        previous_gpa = st.number_input("Previous GPA", value=3.2)
        sleep_hours_per_night = st.number_input("Sleep Hours per Night", value=7.0)
        extracurricular_activities_count = st.number_input("Extracurricular Activities Count", value=2.0)
        stress_level_score = st.number_input("Stress Level Score", value=5.0)
        tutoring_hours_per_week = st.number_input("Tutoring Hours per Week", value=2.0)
        distance_from_home_km = st.number_input("Distance from Home (km)", value=5.0)
        part_time_job_hours_per_week = st.number_input("Part-time Job Hours per Week", value=0.0)
        submitted = st.form_submit_button("Predict Exam Score")

    if submitted:
        payload = {
            "major_subject": major_subject,
            "parental_education_level": parental_education_level,
            "internet_access_quality": internet_access_quality,
            "part_time_job": part_time_job,
            "study_group_participation": study_group_participation,
            "study_hours_per_week": float(study_hours_per_week),
            "attendance_rate_pct": float(attendance_rate_pct),
            "previous_gpa": float(previous_gpa),
            "sleep_hours_per_night": float(sleep_hours_per_night),
            "extracurricular_activities_count": float(extracurricular_activities_count),
            "stress_level_score": float(stress_level_score),
            "tutoring_hours_per_week": float(tutoring_hours_per_week),
            "distance_from_home_km": float(distance_from_home_km),
            "part_time_job_hours_per_week": float(part_time_job_hours_per_week),
        }
        input_row = pd.DataFrame([payload])
        predicted_score = float(regression_pipeline.predict(input_row)[0])
        bounded_score = round(float(np.clip(predicted_score, 0, 100)), 1)
        st.success(f"Predicted final exam score: {bounded_score} / 100")

# -----------------------------------------------------------------------
# 2. Classification
# -----------------------------------------------------------------------
with tab_classification:
    with st.form("classification_form"):
        policy_type = st.selectbox("Policy Type", ["Auto", "Health", "Life", "Property"], index=0)
        claim_type = st.selectbox(
            "Claim Type", ["Collision", "Fire", "Liability", "Medical", "Theft"], index=0)
        incident_location_type = st.selectbox(
            "Incident Location Type", ["Rural", "Suburban", "Urban"], index=2)
        witnesses_present = st.selectbox("Witnesses Present", ["No", "Yes"], index=0)
        agent_flagged_history = st.selectbox("Agent Flagged History", ["No", "Yes"], index=1)
        police_report_filed = st.selectbox("Police Report Filed", ["No", "Yes"], index=0)
        claimant_age = st.number_input("Claimant Age", value=35.0)
        policy_tenure_years = st.number_input("Policy Tenure (years)", value=2.0)
        prior_claims_count = st.number_input("Prior Claims Count", value=3.0)
        days_to_report_claim = st.number_input("Days to Report Claim", value=20.0)
        claim_amount_kes = st.number_input("Claim Amount (KES)", value=900000.0)
        estimated_repair_cost_kes = st.number_input("Estimated Repair Cost (KES)", value=400000.0)
        claim_amount_to_policy_limit_ratio = st.number_input(
            "Claim Amount to Policy Limit Ratio", value=0.9)
        customer_credit_score = st.number_input("Customer Credit Score", value=520.0)
        submitted2 = st.form_submit_button("Predict Claim Fraud")

    if submitted2:
        payload = {
            "policy_type": policy_type,
            "claim_type": claim_type,
            "incident_location_type": incident_location_type,
            "witnesses_present": witnesses_present,
            "agent_flagged_history": agent_flagged_history,
            "police_report_filed": police_report_filed,
            "claimant_age": float(claimant_age),
            "policy_tenure_years": float(policy_tenure_years),
            "prior_claims_count": float(prior_claims_count),
            "days_to_report_claim": float(days_to_report_claim),
            "claim_amount_kes": float(claim_amount_kes),
            "estimated_repair_cost_kes": float(estimated_repair_cost_kes),
            "claim_amount_to_policy_limit_ratio": float(claim_amount_to_policy_limit_ratio),
            "customer_credit_score": float(customer_credit_score),
        }
        input_row = pd.DataFrame([payload])
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_
        st.success(f"Predicted claim status: {predicted_class}")
        st.write({str(label): round(float(prob), 4)
                  for label, prob in zip(class_labels, predicted_probabilities)})

# -----------------------------------------------------------------------
# 3. Clustering
# -----------------------------------------------------------------------
with tab_clustering:
    with st.form("clustering_form"):
        account_balance_kes = st.number_input("Account Balance (KES)", value=250000.0)
        transactions_per_month = st.number_input("Transactions per Month", value=20.0)
        avg_transaction_amount_kes = st.number_input("Average Transaction Amount (KES)", value=8000.0)
        years_as_customer = st.number_input("Years as Customer", value=6.0)
        digital_banking_usage_score = st.number_input("Digital Banking Usage Score", value=70.0)
        submitted3 = st.form_submit_button("Predict Segment")

    if submitted3:
        input_row = pd.DataFrame([{
            "account_balance_kes": float(account_balance_kes),
            "transactions_per_month": float(transactions_per_month),
            "avg_transaction_amount_kes": float(avg_transaction_amount_kes),
            "years_as_customer": float(years_as_customer),
            "digital_banking_usage_score": float(digital_banking_usage_score),
        }])[clustering_features]
        scaled_input = clustering_scaler.transform(input_row)
        cluster = int(clustering_kmeans.predict(scaled_input)[0])
        st.success(f"Assigned segment: Cluster {cluster}")

# -----------------------------------------------------------------------
# 4. Association Rule Mining
# -----------------------------------------------------------------------
with tab_arm:
    with st.form("arm_form"):
        cart_text = st.text_input("Cart (comma-separated SKUs)", "D03, D05")
        top_n = st.number_input("Top N Recommendations", min_value=1, value=3)
        submitted4 = st.form_submit_button("Get Recommendations")

    if submitted4:
        cart = {sku.strip().upper() for sku in cart_text.split(",") if sku.strip()}

        # Keep only the rules whose "if" part is fully inside the cart,
        # best rules first (highest confidence, then highest lift)
        matched_rules = recommendation_lookup[
            recommendation_lookup["antecedents"].apply(lambda x: set(x).issubset(cart))
        ].sort_values(by=["confidence", "lift"], ascending=False)

        recommendations = []
        for _, rule in matched_rules.iterrows():
            for sku in rule["consequents"]:
                if sku not in cart and sku not in [r[0] for r in recommendations]:
                    recommendations.append((sku, rule["confidence"], rule["lift"]))
        recommendations = recommendations[:int(top_n)]

        if not recommendations:
            st.warning("No recommendations for this cart -- try a SKU combination seen in the training orders.")
        else:
            for sku, confidence, lift in recommendations:
                st.write(f"**{sku_lookup.get(sku, sku)}** ({sku}) "
                         f"-- confidence {confidence:.1%}, lift {lift:.2f}")