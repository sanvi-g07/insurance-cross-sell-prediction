import requests

url = "http://0.0.0.0:8080/predict"

customer = {
        "gender":'Male',
        "age":35,
        "drivinglicense":1,
        "regioncode":28.0,
        "previouslyinsured":0,
        "vehicleage":'1-2 Year',
        "vehicledamage":'Yes',
        "annualpremium":32000.0,
        "policysaleschannel":26.0,
        "vintage":150,
}

response = requests.post(url, json=customer).json()
 
print(f'Response probability: {response['convert_prob']}')
print(f'Convert value: {response['convert']}')
print(f'Predicted response: {response['convert_response']}')