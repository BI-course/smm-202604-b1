import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",
    "null",
]}})

MODEL_DIR = "./model"

# Load Group B Artifacts
regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "20261020_084339_ridge_regressor_for_exam_score.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "20261020_084942_logisticregression_tuned_classifier_for_insurance_claim_fraud.joblib")
)

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "20261020_085241_kmeans_bank_wealth_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
# The 5 columns the clustering model was trained on, in the order it expects
clustering_features = clustering_artifact["features"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "pharmacy_recommender.joblib")
)
recommendation_lookup = arm_artifact["rules"]
sku_lookup = arm_artifact["sku_lookup"]

print("Group B artifacts loaded successfully. API is ready.")


# 1. REGRESSION -- Predict Student Final Exam Score
@app.route("/api/v1/models/student-exam-score/predictions", methods=["POST"])
def predict_exam_score():
    payload = request.get_json()
    input_row = pd.DataFrame([payload])

    predicted_score = float(regression_pipeline.predict(input_row)[0])
    bounded_score = round(float(np.clip(predicted_score, 0, 100)), 1)

    return jsonify({
        "predicted_final_exam_score": bounded_score,
        "model": "Tuned Ridge Regressor",
    })


# 2. CLASSIFICATION -- Predict Insurance Claim Fraud
@app.route("/api/v1/models/insurance-claim-fraud/predictions", methods=["POST"])
def predict_claim_fraud():
    payload = request.get_json()
    input_row = pd.DataFrame([payload])

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_

    return jsonify({
        "predicted_claim_status": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Tuned Logistic Regression (SMOTE)",
    })


# 3. CLUSTERING -- Assign Bank Customer Segment
@app.route("/api/v1/models/bank-customer-segmenter/predictions", methods=["POST"])
def predict_customer_segment():
    # Error handling 1: the body must be a valid, non-empty JSON object
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or len(payload) == 0:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Error handling 2: all 5 required fields must be present
    missing = [field for field in clustering_features if field not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Error handling 3: every value must be a number (true/false is not a number)
    not_numbers = [
        field for field in clustering_features
        if not isinstance(payload[field], (int, float)) or isinstance(payload[field], bool)
    ]
    if not_numbers:
        return jsonify({"error": f"These fields must be numbers: {not_numbers}"}), 400

    # Keep the 5 columns in the same order the scaler was trained on
    input_row = pd.DataFrame([payload])[clustering_features]

    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": "K-Means (Bank Customer Segmentation)",
    })


# 4. ASSOCIATION RULE MINING -- Recommend Products
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    # Error handling 1: the body must be a valid JSON object
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Error handling 2: "cart" must be a non-empty array
    cart = payload.get("cart")
    if not isinstance(cart, list) or len(cart) == 0:
        return jsonify({"error": "'cart' must be a non-empty array of product codes"}), 400

    top_n = int(payload.get("top_n", 3))

    # If recommendation_lookup is a DataFrame of rules:
    # Filter rules where antecedents are subsets of cart
    matched_rules = recommendation_lookup[
        recommendation_lookup["antecedents"].apply(lambda x: set(x).issubset(set(cart)))
    ]

    # Sort by confidence/lift and drop items already in cart
    sorted_rules = matched_rules.sort_values(by=["confidence", "lift"], ascending=False)

    recommendations = []
    for _, row in sorted_rules.iterrows():
        for consequent in row["consequents"]:
            if consequent not in cart and consequent not in [r["sku"] for r in recommendations]:
                product_name = sku_lookup.get(consequent, str(consequent))
                recommendations.append({
                    "sku": str(consequent),
                    "product_name": product_name,
                    "confidence": round(float(row["confidence"]), 3),
                    "lift": round(float(row["lift"]), 2),
                })
                if len(recommendations) >= top_n:
                    break
        if len(recommendations) >= top_n:
            break

    return jsonify({
        "cart": cart,
        "recommendations": recommendations,
        "model": "Apriori association rules",
    })


if __name__ == "__main__":
    app.run(debug=True)