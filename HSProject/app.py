from flask import Flask, request, jsonify
import pickle
from utils.predict import predict_churn_and_sentiment

app = Flask(__name__)

# Load trained models
churn_model = pickle.load(open("model/churn_model.pkl", "rb"))
sentiment_model = pickle.load(open("model/sentiment_model.pkl", "rb"))

@app.route("/")
def home():
    return "✅ Smart Churn Prediction API is Running. Use POST /predict to get predictions."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        result = predict_churn_and_sentiment(data, churn_model, sentiment_model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
