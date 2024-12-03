import json

# Load the dataset
try:
    with open('chatbot_dataset.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: 'chatbot_dataset.json' not found in the current directory.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format. Details: {e}")
    exit()

# Validate the dataset
for intent in data["intents"]:
    tag = intent["tag"]
    patterns = intent["patterns"]
    responses = intent["responses"]

    if not patterns or not responses:
        print(f"Tag '{tag}' has missing patterns or responses.")
    elif not all(isinstance(item, str) for item in responses):
        print(f"Tag '{tag}' has invalid response format.")
print("Validation complete.")
