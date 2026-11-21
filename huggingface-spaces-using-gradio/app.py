import gradio as gr
import joblib
import numpy as np
import pandas as pd
import spaces

# -----------------------------------------------------------------------
# Hugging Face's free tier now provisions new Gradio Spaces on ZeroGPU
# hardware only. None of this app's four models needs a GPU, but some
# free-tier Spaces fail to start unless at least one function is
# decorated with @spaces.GPU. This function does nothing GPU-related;
# it only satisfies that platform check.
@spaces.GPU
def _warm_zerogpu():
    return None

_warm_zerogpu()

# -----------------------------------------------------------------------
# Load all four saved models ONCE, when the app starts -- the same idea
# as api.py: load once at startup, reuse for every click, never retrain.
# The four .joblib files sit in the root folder of the Space.
# -----------------------------------------------------------------------
regression_pipeline = joblib.load("20261020_084339_ridge_regressor_for_exam_score.joblib")
classification_pipeline = joblib.load(
    "20261020_084942_logisticregression_tuned_classifier_for_insurance_claim_fraud.joblib"
)

clustering_artifact = joblib.load("20261020_085241_kmeans_bank_wealth_segmentation.joblib")
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
# The 5 columns the clustering model was trained on, in the order it expects
clustering_features = clustering_artifact["features"]

arm_artifact = joblib.load("pharmacy_recommender.joblib")
recommendation_lookup = arm_artifact["rules"]
sku_lookup = arm_artifact["sku_lookup"]


# -----------------------------------------------------------------------
# 1. Regression -- predict a student's final exam score
# -----------------------------------------------------------------------
def predict_exam_score(major_subject, parental_education_level, internet_access_quality,
                       part_time_job, study_group_participation, study_hours_per_week,
                       attendance_rate_pct, previous_gpa, sleep_hours_per_night,
                       extracurricular_activities_count, stress_level_score,
                       tutoring_hours_per_week, distance_from_home_km,
                       part_time_job_hours_per_week):
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
    return f"{bounded_score} / 100"


# -----------------------------------------------------------------------
# 2. Classification -- predict the status of an insurance claim
# -----------------------------------------------------------------------
def predict_claim_fraud(policy_type, claim_type, incident_location_type,
                        witnesses_present, agent_flagged_history, police_report_filed,
                        claimant_age, policy_tenure_years, prior_claims_count,
                        days_to_report_claim, claim_amount_kes, estimated_repair_cost_kes,
                        claim_amount_to_policy_limit_ratio, customer_credit_score):
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
    probability_lines = "\n".join(
        f"  {label}: {prob:.1%}" for label, prob in zip(class_labels, predicted_probabilities)
    )
    return f"Predicted claim status: {predicted_class}\n\n{probability_lines}"


# -----------------------------------------------------------------------
# 3. Clustering -- assign a bank customer to a segment
# -----------------------------------------------------------------------
def predict_segment(account_balance_kes, transactions_per_month,
                    avg_transaction_amount_kes, years_as_customer,
                    digital_banking_usage_score):
    # Keep the 5 columns in the same order the scaler was trained on
    input_row = pd.DataFrame([{
        "account_balance_kes": float(account_balance_kes),
        "transactions_per_month": float(transactions_per_month),
        "avg_transaction_amount_kes": float(avg_transaction_amount_kes),
        "years_as_customer": float(years_as_customer),
        "digital_banking_usage_score": float(digital_banking_usage_score),
    }])[clustering_features]
    scaled_input = clustering_scaler.transform(input_row)
    cluster = int(clustering_kmeans.predict(scaled_input)[0])
    return f"Assigned segment: Cluster {cluster}"


# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend pharmacy products for a cart
# -----------------------------------------------------------------------
def recommend_products(cart_text, top_n):
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
        return "No recommendations for this cart -- try a SKU combination seen in the training orders."

    lines = [
        f"{sku_lookup.get(sku, sku)} ({sku})  --  confidence {confidence:.1%}, lift {lift:.2f}"
        for sku, confidence, lift in recommendations
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------
# One gr.Blocks() app, four gr.Tab()s -- one shared Space URL.
# -----------------------------------------------------------------------
with gr.Blocks(title="Group B1 Model Validation") as demo:
    gr.Markdown("# Group B1 Model Validation")

    with gr.Tab("Student Exam Score Regressor"):
        gr.Interface(
            fn=predict_exam_score,
            inputs=[
                # These categories are the exact set the model was trained on --
                # any other word causes an "unknown category" error.
                gr.Dropdown(label="Major Subject",
                            choices=["Business", "Engineering", "Humanities", "Sciences"],
                            value="Engineering"),
                gr.Dropdown(label="Parental Education Level",
                            choices=["High School", "Bachelor's", "Master's", "Doctorate"],
                            value="Bachelor's"),
                gr.Dropdown(label="Internet Access Quality",
                            choices=["Poor", "Average", "Good", "Excellent"], value="Good"),
                gr.Dropdown(label="Part-time Job", choices=["No", "Yes"], value="No"),
                gr.Dropdown(label="Study Group Participation", choices=["No", "Yes"], value="Yes"),
                gr.Number(label="Study Hours per Week", value=15),
                gr.Number(label="Attendance Rate (%)", value=90),
                gr.Number(label="Previous GPA", value=3.2),
                gr.Number(label="Sleep Hours per Night", value=7),
                gr.Number(label="Extracurricular Activities Count", value=2),
                gr.Number(label="Stress Level Score", value=5),
                gr.Number(label="Tutoring Hours per Week", value=2),
                gr.Number(label="Distance from Home (km)", value=5),
                gr.Number(label="Part-time Job Hours per Week", value=0),
            ],
            outputs=gr.Textbox(label="Predicted Final Exam Score"),
        )

    with gr.Tab("Insurance Claim Fraud Classifier"):
        gr.Interface(
            fn=predict_claim_fraud,
            inputs=[
                gr.Dropdown(label="Policy Type",
                            choices=["Auto", "Health", "Life", "Property"], value="Auto"),
                gr.Dropdown(label="Claim Type",
                            choices=["Collision", "Fire", "Liability", "Medical", "Theft"],
                            value="Collision"),
                gr.Dropdown(label="Incident Location Type",
                            choices=["Rural", "Suburban", "Urban"], value="Urban"),
                gr.Dropdown(label="Witnesses Present", choices=["No", "Yes"], value="No"),
                gr.Dropdown(label="Agent Flagged History", choices=["No", "Yes"], value="Yes"),
                gr.Dropdown(label="Police Report Filed", choices=["No", "Yes"], value="No"),
                gr.Number(label="Claimant Age", value=35),
                gr.Number(label="Policy Tenure (years)", value=2),
                gr.Number(label="Prior Claims Count", value=3),
                gr.Number(label="Days to Report Claim", value=20),
                gr.Number(label="Claim Amount (KES)", value=900000),
                gr.Number(label="Estimated Repair Cost (KES)", value=400000),
                gr.Number(label="Claim Amount to Policy Limit Ratio", value=0.9),
                gr.Number(label="Customer Credit Score", value=520),
            ],
            outputs=gr.Textbox(label="Predicted Claim Status"),
        )

    with gr.Tab("Bank Customer Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[
                gr.Number(label="Account Balance (KES)", value=250000),
                gr.Number(label="Transactions per Month", value=20),
                gr.Number(label="Average Transaction Amount (KES)", value=8000),
                gr.Number(label="Years as Customer", value=6),
                gr.Number(label="Digital Banking Usage Score", value=70),
            ],
            outputs=gr.Textbox(label="Prediction"),
        )

    with gr.Tab("Pharmacy Recommender"):
        gr.Interface(
            fn=recommend_products,
            inputs=[
                gr.Textbox(label="Cart (comma-separated SKUs)", value="D03, D05"),
                gr.Number(label="Top N Recommendations", value=3),
            ],
            outputs=gr.Textbox(label="Recommendations"),
        )

if __name__ == "__main__":
    demo.launch()