import requests

url = "http://localhost:8008/predict"
data = {
    "instances": [
        {"filepath": "http://t0.gstatic.com/licensed-image?q=tbn:ANd9GcSdf6ERvaPyckc-PAJNBsK3MbSe-ZU57YOsxLsh0BXa1yZ2kWGhzFp4T1JnQUVsqMkb652lX0fFqtHvvjHx5ig"}
    ]
}

response = requests.post(url, json=data)
print(response.json())
