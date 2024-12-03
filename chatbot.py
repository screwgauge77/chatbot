import random
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

# Set up logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Global context
current_context = None

def load_dataset(file_path):
    """Load the chatbot dataset from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return {"intents": []}

# Load dataset
data = load_dataset('chatbot_dataset.json')
intents = data['intents']

# Prepare data for training
all_patterns = []
intent_labels = []
tag_to_intent = {}

for intent in intents:
    tag_to_intent[intent['tag']] = intent
    patterns = intent['patterns']
    all_patterns.extend(patterns)
    intent_labels.extend([intent['tag']] * len(patterns))

# Use SentenceTransformer for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(all_patterns)

def find_best_matching_intent(user_input, threshold=0.7):
    """Find the best matching intent for user input."""
    global current_context
    user_vec = model.encode([user_input])
    similarity_scores = cosine_similarity(user_vec, X)
    best_matching_intent_idx = similarity_scores.argmax()
    best_score = similarity_scores[0][best_matching_intent_idx]

    if best_score < threshold:
        return "fallback"

    matched_tag = intent_labels[best_matching_intent_idx]
    matched_intent = tag_to_intent.get(matched_tag, {})
    
    # Check context
    if 'context_filter' in matched_intent and matched_intent['context_filter'] != current_context:
        return "fallback"

    # Set new context if provided
    if 'context_set' in matched_intent:
        current_context = matched_intent['context_set']
    
    # Logging the match
    logging.info(f"User input: {user_input}, Matched intent: {matched_tag}, Confidence: {best_score:.2f}")
    return matched_tag

def get_random_response(tag, sentiment="neutral"):
    """Get a random response for a given intent tag, considering sentiment if applicable."""
    for intent in intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            if isinstance(responses, dict):
                return random.choice(responses.get(sentiment, responses["neutral"]))
            return random.choice(responses)
    return "I'm not sure how to respond to that."

def analyze_sentiment(user_input):
    """Analyze the sentiment of user input."""
    blob = TextBlob(user_input)
    return blob.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

def chatbot_main_loop():
    """Main loop for the chatbot interaction."""
    global current_context
    print("Hey, how can I help you? Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye! Have a great day!")
            logging.info("User exited the chat.")
            break
        
        matched_intent = find_best_matching_intent(user_input)
        sentiment_score = analyze_sentiment(user_input)

        # Determine sentiment category
        sentiment = "neutral"
        if sentiment_score < -0.5:
            sentiment = "negative"
        elif sentiment_score > 0.5:
            sentiment = "positive"

        response = get_random_response(matched_intent, sentiment)
        print(f"Chatbot: {response}")
        
        # Log sentiment and response
        logging.info(f"User: {user_input}, Chatbot: {response}, Sentiment: {sentiment_score:.2f}")

# Run the chatbot
if __name__ == "__main__":
    chatbot_main_loop()
