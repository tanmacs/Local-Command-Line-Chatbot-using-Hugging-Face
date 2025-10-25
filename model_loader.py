"""
Model Loader Module
Handles loading and initialization of Hugging Face text generation models.
"""

from transformers import pipeline
import torch
import warnings

# Suppress the dataset warning for single-input chatbot usage
warnings.filterwarnings("ignore", message=".*use a dataset.*")


class ModelLoader:
    """Loads and manages the Hugging Face text generation model."""
    
    def __init__(self, model_name="distilgpt2"):
        """
        Initialize the model loader.
        
        Args:
            model_name (str): Name of the Hugging Face model to use.
                            Default is "distilgpt2" (small and fast).
        """
        self.model_name = model_name
        self.generator = None
        
    def load_model(self):
        """
        Load the text generation pipeline with the specified model.
        
        Returns:
            pipeline: Hugging Face text generation pipeline.
        """
        print(f"Loading model: {self.model_name}...")
        
        # Determine device (GPU if available, else CPU)
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"Using device: {device_name}")
        
        # Determine task type based on model
        task = "text2text-generation" if "t5" in self.model_name.lower() else "text-generation"
        
        # Load the pipeline
        self.generator = pipeline(
            task,
            model=self.model_name,
            device=device
        )
        
        print("Model loaded successfully!\n")
        return self.generator
    
    def get_generator(self):
        """
        Get the loaded generator pipeline.
        
        Returns:
            pipeline: The text generation pipeline.
        """
        if self.generator is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.generator
