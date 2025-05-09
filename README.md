from transformers import pipeline, GenerationConfig

class CustomerSupportChatbot:
    def __init__(self):
        # Load a pre-trained conversational model
        self.chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

    def chat(self):
        print("Welcome to Customer Support!")
        print("Type 'exit' to end the conversation.\n")

        conversation_history = []
        # Conversation is replaced with a list to store past interactions
        # conversation = Conversation()  

        while True:
            user_input = input("You: ")
            
