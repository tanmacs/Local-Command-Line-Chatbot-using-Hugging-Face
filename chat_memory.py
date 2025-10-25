"""
Chat Memory Module
Implements a sliding window buffer to maintain conversation history.
"""

from collections import deque


class ChatMemory:
    """Manages conversation history with a sliding window buffer."""
    
    def __init__(self, window_size=4):
        """
        Initialize the chat memory.
        
        Args:
            window_size (int): Maximum number of conversation turns to keep.
                             Default is 4 (last 4 exchanges).
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def add_user_message(self, message):
        """
        Add a user message to the conversation history.
        
        Args:
            message (str): User's input message.
        """
        self.history.append(f"User: {message}")
    
    def add_bot_message(self, message):
        """
        Add a bot response to the conversation history.
        
        Args:
            message (str): Bot's response message.
        """
        self.history.append(f"Bot: {message}")
    
    def get_context(self):
        """
        Get the current conversation context as a formatted string.
        
        Returns:
            str: Formatted conversation history.
        """
        return "\n".join(self.history)
    
    def get_prompt(self, user_input):
        """
        Generate a prompt with conversation context for the model.
        
        Args:
            user_input (str): Current user input.
            
        Returns:
            str: Complete prompt including history and current input.
        """
        context = self.get_context()
        if context:
            prompt = f"{context}\nUser: {user_input}\nBot:"
        else:
            prompt = f"User: {user_input}\nBot:"
        return prompt
    
    def clear(self):
        """Clear all conversation history."""
        self.history.clear()
    
    def __len__(self):
        """Return the number of messages in history."""
        return len(self.history)
