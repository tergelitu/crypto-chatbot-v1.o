import os
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from dotenv import load_dotenv


load_dotenv()

class CryptoChatbot:
    def __init__(self):
        self.api_url = "https://api.coingecko.com/api/v3/simple/price"
        self.api_key = os.getenv("COINGECKO_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.greetings = ["hi", "hello", "how are you"]
        self.default_response = "Hi, I'm Crypto Chatbot, your crypto advisor. How can I assist you today?"
        self.cryptocurrencies = ["bitcoin", "ethereum", "solana", "cardano", "ripple", "doge"]
        self.coins = {crypto: crypto for crypto in self.cryptocurrencies}
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_auth_token=self.huggingface_token)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", use_auth_token=self.huggingface_token)
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        self.send_greeting()

    def send_greeting(self):
        print("Hi, I'm Crypto Chatbot. How can I assist you with cryptocurrency today?")

    def fetch_price(self, coin):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = requests.get(self.api_url, params={"ids": coin, "vs_currencies": "usd"}, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data[coin]["usd"]

    def get_text_generation_response(self, prompt):
        response = self.generator(prompt, max_length=150, num_return_sequences=1)
        return response[0]['generated_text'].strip()

    def respond(self, user_input):
        if not user_input.strip():
            return self.default_response
        
        user_input = user_input.lower()
        
        if user_input in self.greetings:
            return "Hello! How can I help you with cryptocurrencies today?"
        
        for crypto in self.cryptocurrencies:
            if crypto in user_input:
                current_price = self.fetch_price(crypto)
                prompt = (f"The current price of {crypto.capitalize()} is ${current_price} USD. "
                          f"Provide a detailed analysis and prediction for {crypto.capitalize()}.")
                generated_response = self.get_text_generation_response(prompt)
                return generated_response
        
        if "crypto" in user_input:
            return "Sure, I can help you with information about various cryptocurrencies. What would you like to know?"
        
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

if __name__ == "__main__":
    bot = CryptoChatbot()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = bot.respond(user_input)
        print(f"Chatbot: {response}")
