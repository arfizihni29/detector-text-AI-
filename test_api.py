import requests

url = "http://localhost:5000/api/detect"
payload = {
    "text": "As an AI language model, I don't have personal opinions. However, many people believe that artificial intelligence will revolutionize the way we work and live.",
    "lang": "en"
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", e)
