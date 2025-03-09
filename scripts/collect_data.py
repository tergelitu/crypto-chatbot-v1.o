def save_interaction(user_input, bot_response, file_path="../data/interactions.txt"):
    with open(file_path, "a") as file:
        file.write(f"User: {user_input}\n")
        file.write(f"Bot: {bot_response}\n\n")

if __name__ == "__main__":
    user_input = input("You: ")
    bot_response = "This is a sample response."
    save_interaction(user_input, bot_response)
    print("Interaction saved.")
