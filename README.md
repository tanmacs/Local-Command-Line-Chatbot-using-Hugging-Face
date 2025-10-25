# Local Command-Line Chatbot

A fully functional local chatbot interface using Hugging Face text generation models. This chatbot maintains conversation context using a sliding window memory mechanism for coherent multi-turn conversations.

## Features

- **Local Execution**: Runs entirely on your machine (GPU optional)
- **Conversational Memory**: Maintains context from recent exchanges using a sliding window buffer
- **Modular Design**: Clean, maintainable code structure
- **Fast & Lightweight**: Uses Flan T5 XL by default (can be customized)
- **Simple CLI**: Easy-to-use command-line interface

## Project Structure

```
chatbot-project/
├── model_loader.py    # Model and tokenizer loading
├── chat_memory.py     # Memory buffer logic with sliding window
├── interface.py       # CLI loop and integration
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd chatbot-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `transformers` - Hugging Face library for language models
   - `torch` - PyTorch deep learning framework

3. **First run will download the model** (approximately 8GB for Flan T5 XL)

## How to Run

### Basic Usage

Run the chatbot with default settings:

```bash
python interface.py
```

### Customization

You can modify the following parameters in `interface.py`:

- **MODEL_NAME**: Change the Hugging Face model (default: `" google/flan-t5-xl"`)
  - Alternative options: `"gpt2"`, `"gpt2-medium"`, etc.
  
- **MEMORY_WINDOW**: Adjust conversation memory size (default: 4 turns)
  - Increase for longer context retention
  - Decrease for faster responses

### Commands

- **Chat normally**: Just type your message and press Enter
- **Exit**: Type `/exit` to quit the chatbot
- **Keyboard Interrupt**: Press `Ctrl+C` to exit

## Sample Interaction

```
============================================================
Welcome to the Local Command-Line Chatbot!
============================================================
Loading model: google/flan-t5-xl...
Using device: CPU
Model loaded successfully!

Type your messages below. Use '/exit' to quit.

User: What is the capital of France?
Bot: The capital of France is Paris.

User: And what about Italy?
Bot: The capital of Italy is Rome.

User: Tell me something interesting about Rome
Bot: Rome is known as the Eternal City and has over 2,500 years of history.

User: /exit

Exiting chatbot. Goodbye!
```

## Technical Details

### Model Loader (`model_loader.py`)
- Handles Hugging Face model initialization
- Automatically detects GPU availability
- Uses the `pipeline` API for simplified generation

### Chat Memory (`chat_memory.py`)
- Implements sliding window buffer using Python's `deque`
- Stores last N conversation turns (configurable)
- Formats conversation history into prompts for the model

### Interface (`interface.py`)
- Main CLI loop with continuous input handling
- Integrates model and memory components
- Handles graceful exit and error recovery
- Parses model output to extract clean responses

## Design Decisions

1. **Flan T5 XL as Default Model**
   - Lightweight and fast
   - Good balance between performance and resource usage
   - Works well on CPU without GPU

2. **Sliding Window Memory (4 turns)**
   - Keeps context manageable for small models
   - Prevents prompt from growing too large
   - Maintains conversation coherence

3. **Modular Architecture**
   - Separation of concerns (model, memory, interface)
   - Easy to test and extend
   - Clear responsibility for each component

4. **Response Parsing**
   - Extracts only the bot's response from generated text
   - Stops at natural boundaries (newlines, next user input)
   - Provides fallback for edge cases

## Troubleshooting

### Model Download Issues
If the model fails to download:
- Check your internet connection
- Ensure you have sufficient disk space (~10GB)
- Try manually downloading from Hugging Face

### Memory Issues
If you encounter out-of-memory errors:
- Reduce `MEMORY_WINDOW` size
- Use a smaller model
- Reduce `max_new_tokens` in `generate_response()`

### Slow Performance
- First generation is slower (model loading)
- Subsequent responses should be faster
- Consider using a GPU if available

## Requirements

- Python 3.8+
- transformers >= 4.30.0
- torch >= 2.0.0

## Future Enhancements

Possible improvements:
- Add support for different conversation formats
- Implement conversation saving/loading
- Add web interface using Flask/FastAPI
- Support for instruction-tuned models
- Temperature and generation parameter controls via CLI

## License

This project is created for educational purposes as part of the ATG Technical Assignment.

---

**Developed by:** Tanmay Rahul Kalanke  
**Date:** October 2025  
**Assignment:** ATG Machine Learning Intern - Local CLI Chatbot
