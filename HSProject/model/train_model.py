import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load data
df = pd.read_csv("data/customer_data.csv")

# Ensure required columns are correct
df = df.rename(columns={
    'monthly_charges': 'MonthlyCharges',
    'contract_type': 'Contract'
})  # adjust if your CSV uses lowercase

# Extract features and labels
X = df[['tenure', 'MonthlyCharges', 'Contract']]
y = df['churn']

# Train churn prediction model
model = LogisticRegression()
model.fit(X, y)

# Save churn model
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Create and save sentiment analyzer
sentiment_model = SentimentIntensityAnalyzer()
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)
