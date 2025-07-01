import requests

url = "http://localhost:8008/predict"
data = {
    "instances": [
        {
            "filepath": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxSyhtXpCD4XuWXtf7BbfC-bYMbNlKLWJjWAPmEjOGaWbkUD61Q6dvnBbhkwH57Pidg5vaOGVgFF2pIfNiIuZorg",
            "country": "POL"
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())
