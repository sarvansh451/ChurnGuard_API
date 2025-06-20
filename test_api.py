import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "tenure": 10,
    "MonthlyCharges": 85.5,
    "Contract": 1,
    "feedback": "I am unhappy with the service"
}

response = requests.post(url, json=data)
print(response.json())
