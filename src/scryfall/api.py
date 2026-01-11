import requests

response = requests.get("https://api.scryfall.com")
# print(response.status_code)
print(response.json())

