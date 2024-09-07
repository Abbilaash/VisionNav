import json


with open('message.json', 'r') as file:
    messages_data = json.load(file)

def messages():
    messages = messages_data['messages']
    return messages