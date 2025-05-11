import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK data (only first time)
nltk.download('punkt')

# Sample dataset of intents and responses
intents = {
    'greeting': {
        'patterns': ['Hi', 'Hello', 'Is anyone there?', 'Hey', 'Good day'],
        'responses': ['Hello!', 'Hi there!', 'How can I help you today?']
    },
    'goodbye': {
        'patterns': ['Bye', 'See you later', 'Goodbye'],
        'responses': ['Goodbye!', 'See you soon!', 'Have a great day!']
    },
    'thanks': {
        'patterns': ['Thanks', 'Thank you', 'That helps'],
        'responses': ['Happy to help!', 'Anytime!', 'You’re welcome!']
    },
    'noanswer': {
        'patterns': [],
        'responses': ['Sorry, I did not understand that.', 'Can you please rephrase?']
    },
    'order_status': {
        'patterns': ['Where is my order?', 'Order status', 'Track my order'],
        'responses': ['Please provide your order ID to track the status.']
    },
    'refund': {
        'patterns': ['I want a refund', 'How do I get a refund?', 'Refund status'],
        'responses': ['You can request a refund from your order history page.']
    }
}

# Prepare training data
all_patterns = []
all_labels = []

for intent, data in intents.items():
    for pattern in data['patterns']:
        all_patterns.append(pattern.lower())
        all_labels.append(intent)

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(all_patterns)

# Train classifier
clf = MultinomialNB()
clf.fit(X, all_labels)

# Chatbot response function
def chatbot_response(user_input):
    user_input = user_input.lower()
    X_test = vectorizer.transform([user_input])
    pred = clf.predict(X_test)[0]

    if pred in intents:
        return random.choice(intents[pred]['responses'])
    else:
        return random.choice(intents['noanswer']['responses'])

# Run the chatbot
print("Customer Support Chatbot (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")

