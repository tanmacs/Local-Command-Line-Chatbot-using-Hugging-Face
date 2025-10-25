"""
Interface Module
Main CLI loop for chatbot interaction.
"""

import sys
from model_loader import ModelLoader
from chat_memory import ChatMemory


class ChatbotInterface:
    """Command-line interface for the chatbot."""
    
    def __init__(self, model_name="distilgpt2", memory_window=4):
        """
        Initialize the chatbot interface.
        
        Args:
            model_name (str): Hugging Face model to use.
            memory_window (int): Number of conversation turns to remember.
        """
        self.model_loader = ModelLoader(model_name)
        self.memory = ChatMemory(window_size=memory_window)
        self.generator = None
        
    def setup(self):
        """Load the model and prepare the chatbot."""
        print("=" * 60)
        print("Welcome to the Local Command-Line Chatbot!")
        print("=" * 60)
        self.generator = self.model_loader.load_model()
        print("Type your messages below. Use '/exit' to quit.\n")
    
    def generate_response(self, user_input):
        """
        Generate a response using the model with conversation context.
        
        Args:
            user_input (str): User's input message.
            
        Returns:
            str: Bot's response.
        """
        # Check if using T5 model (text2text) or GPT model (text-generation)
        is_t5 = "t5" in self.model_loader.model_name.lower()
        
        if is_t5:
            # T5 models need context for follow-up questions
            # Build a context-aware prompt
            context = self.memory.get_context()
            if context:
                # Rewrite follow-up questions with context for clarity
                prompt = f"Given this conversation:\n{context}\n\nAnswer this question: {user_input}"
            else:
                prompt = user_input
            
            result = self.generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=False  # Use greedy decoding for factual answers
            )
            response = result[0]['generated_text'].strip()
        else:
            # GPT-style models use conversation format
            prompt = self.memory.get_prompt(user_input)
            result = self.generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            # Extract the generated text
            full_text = result[0]['generated_text']
            
            # Parse out only the bot's response (after the last "Bot:")
            if "Bot:" in full_text:
                response = full_text.split("Bot:")[-1].strip()
                if "\nUser:" in response:
                    response = response.split("\nUser:")[0].strip()
                elif "\n" in response:
                    response = response.split("\n")[0].strip()
            else:
                response = full_text.strip()
        
        return response if response else "I'm not sure how to respond to that."
    
    def run(self):
        """Main loop for the chatbot interface."""
        self.setup()
        
        while True:
            try:
                # Get user input
                user_input = input("User: ").strip()
                
                # Check for exit command
                if user_input.lower() == "/exit":
                    print("\nExiting chatbot. Goodbye!")
                    break
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Add user message to memory
                self.memory.add_user_message(user_input)
                
                # Generate and display response
                response = self.generate_response(user_input)
                print(f"Bot: {response}\n")
                
                # Add bot response to memory
                self.memory.add_bot_message(response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting chatbot. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")


def main():
    """Entry point for the chatbot application."""
    # You can customize these parameters
    MODEL_NAME = "google/flan-t5-xl"  # XL model for best factual accuracy
    MEMORY_WINDOW = 4  # Remember last 4 exchanges
    
    chatbot = ChatbotInterface(
        model_name=MODEL_NAME,
        memory_window=MEMORY_WINDOW
    )
    chatbot.run()


if __name__ == "__main__":
    main()
