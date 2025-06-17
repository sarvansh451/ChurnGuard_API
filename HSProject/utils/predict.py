from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

def predict_churn_and_sentiment(data, churn_model, sentiment_model):
    tenure = float(data["tenure"])
    monthly_charges = float(data["MonthlyCharges"])
    contract_type = int(data["Contract"])
    feedback = data["feedback"]

    X = np.array([[tenure, monthly_charges, contract_type]])
    churn_pred = churn_model.predict(X)[0]

    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(feedback)["compound"]

    sentiment = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"

    return {
        "churn_prediction": int(churn_pred),
        "sentiment": sentiment,
        "sentiment_score": score
    }
