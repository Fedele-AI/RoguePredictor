"""
Model handler for IBM-NASA Geospatial models
Handles loading and managing Hugging Face models for wave analysis

Real IBM-NASA Models Available:
- ibm-nasa-geospatial/Prithvi-100M: Geospatial foundation model trained on NASA's HLS dataset
- Additional models available at: https://huggingface.co/organizations/ibm-nasa-geospatial

Note: The previous 'wave-height-predictor' was a fictional model name.
This handler now uses the real Prithvi-100M model with intelligent fallbacks.
"""

import logging
import os
from typing import Optional, Dict, Any
import warnings

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoModel,
        pipeline
    )
    import torch
    from datasets import load_dataset
    
    # Check if AutoModelForRegression is available (newer versions may not have it)
    try:
        from transformers import AutoModelForRegression
        HAS_REGRESSION = True
    except ImportError:
        HAS_REGRESSION = False
        
except ImportError as e:
    logging.error(f"Transformers library not available: {e}")
    raise

class GeospatialModelHandler:
    """Handles IBM-NASA Geospatial models from Hugging Face"""
    
    def __init__(self, model_name: str = "ibm-nasa-geospatial/Prithvi-100M"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Fallback models if the specified one is not available
        self.fallback_models = [
            "microsoft/DialoGPT-medium",  # General purpose model
            "distilbert-base-uncased",    # Text classification
            "bert-base-uncased",          # General BERT model
            "gpt2",                       # OpenAI's GPT-2 model
            "t5-small"                    # Google's T5 model
        ]
        
        self.logger.info(f"Initializing model handler for: {model_name}")
    
    def load_model(self, use_fallback: bool = True):
        """
        Load the specified model and tokenizer
        
        Args:
            use_fallback: Whether to use fallback models if the specified one fails
            
        Returns:
            Loaded model
        """
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Try to load the specified model
            self.model, self.tokenizer = self._load_specified_model()
            
            if self.model is not None and self.tokenizer is not None:
                self.logger.info(f"Successfully loaded model: {self.model_name}")
                return self.model
            
        except Exception as e:
            self.logger.warning(f"Failed to load specified model {self.model_name}: {str(e)}")
            
            if use_fallback:
                self.logger.info("Attempting to load fallback model")
                return self._load_fallback_model()
            else:
                raise
        
        return None
    
    def _load_specified_model(self):
        """Load the specified IBM-NASA Geospatial model"""
        try:
            # Try different model types based on the task
            model_types = []
            
            # Add regression model type if available
            if HAS_REGRESSION:
                model_types.append((AutoModelForRegression, "regression"))
            
            model_types.extend([
                (AutoModelForSequenceClassification, "classification"),
                (AutoModel, "general")
            ])
            
            for model_class, model_type in model_types:
                try:
                    self.logger.info(f"Trying to load as {model_type} model")
                    
                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        cache_dir="./model_cache"
                    )
                    
                    # Load model
                    model = model_class.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        cache_dir="./model_cache"
                    )
                    
                    # Set model to evaluation mode
                    model.eval()
                    
                    self.logger.info(f"Successfully loaded {model_type} model")
                    return model, tokenizer
                    
                except Exception as e:
                    self.logger.debug(f"Failed to load as {model_type}: {str(e)}")
                    continue
            
            # If all model types fail, try creating a pipeline
            try:
                self.logger.info("Attempting to create pipeline")
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    trust_remote_code=True,
                    cache_dir="./model_cache"
                )
                
                # Create a dummy tokenizer for compatibility
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                return self.pipeline, tokenizer
                
            except Exception as e:
                self.logger.debug(f"Pipeline creation failed: {str(e)}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error loading specified model: {str(e)}")
            return None, None
    
    def _load_fallback_model(self):
        """Load a fallback model if the specified one fails"""
        for fallback_model in self.fallback_models:
            try:
                self.logger.info(f"Loading fallback model: {fallback_model}")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    fallback_model,
                    cache_dir="./model_cache"
                )
                
                # Load model (try different types based on availability)
                model = None
                model_type = "unknown"
                
                # Try classification first
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        fallback_model,
                        cache_dir="./model_cache"
                    )
                    model_type = "classification"
                except:
                    try:
                        # Try general model
                        model = AutoModel.from_pretrained(
                            fallback_model,
                            cache_dir="./model_cache"
                        )
                        model_type = "general"
                    except:
                        # Try regression if available
                        if HAS_REGRESSION:
                            try:
                                model = AutoModelForRegression.from_pretrained(
                                    fallback_model,
                                    cache_dir="./model_cache"
                                )
                                model_type = "regression"
                            except:
                                continue
                        else:
                            continue
                
                if model is not None:
                    # Set model to evaluation mode
                    model.eval()
                    
                    self.logger.info(f"Successfully loaded fallback {model_type} model: {fallback_model}")
                    self.model_name = fallback_model  # Update model name
                    return model
                
            except Exception as e:
                self.logger.warning(f"Failed to load fallback model {fallback_model}: {str(e)}")
                continue
        
        self.logger.error("All fallback models failed to load")
        return None
    
    def create_wave_analysis_pipeline(self):
        """Create a specialized pipeline for wave analysis"""
        if self.pipeline is not None:
            return self.pipeline
        
        try:
            # Create a custom pipeline for wave height prediction
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'task_type'):
                task_type = self.model.config.task_type
            else:
                task_type = "regression"  # Default to regression for wave height
            
            if task_type == "regression":
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7
                )
            else:
                # For classification models, create a text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=50,
                    do_sample=False
                )
            
            self.logger.info(f"Created {task_type} pipeline for wave analysis")
            return self.pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to create wave analysis pipeline: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "pipeline_available": self.pipeline is not None
        }
        
        if self.model is not None:
            try:
                info["model_type"] = type(self.model).__name__
                info["model_parameters"] = sum(p.numel() for p in self.model.parameters())
                info["trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                if hasattr(self.model, 'config'):
                    config = self.model.config
                    info["config"] = {
                        "hidden_size": getattr(config, 'hidden_size', 'N/A'),
                        "num_layers": getattr(config, 'num_layers', 'N/A'),
                        "vocab_size": getattr(config, 'vocab_size', 'N/A')
                    }
            except Exception as e:
                info["model_details"] = f"Error getting details: {str(e)}"
        
        return info
    
    def prepare_input_for_wave_analysis(self, wave_data: Dict[str, Any]) -> str:
        """
        Prepare wave data input for the model
        
        Args:
            wave_data: Dictionary containing wave parameters
            
        Returns:
            Formatted text input for the model
        """
        # Create a text description of the wave conditions
        input_text = f"""
        Wave Analysis Request:
        Location: {wave_data.get('latitude', 'N/A')}°N, {wave_data.get('longitude', 'N/A')}°E
        Wave Height: {wave_data.get('wave_height', 'N/A')} meters
        Wave Period: {wave_data.get('wave_period', 'N/A')} seconds
        Wind Speed: {wave_data.get('wind_speed', 'N/A')} m/s
        Wind Direction: {wave_data.get('wind_direction', 'N/A')}°
        Timestamp: {wave_data.get('timestamp', 'N/A')}
        
        Please analyze these wave conditions and provide insights about:
        1. Wave characteristics and behavior
        2. Potential hazards or unusual patterns
        3. Recommendations for maritime operations
        """
        
        return input_text.strip()
    
    def is_model_ready(self) -> bool:
        """Check if the model is ready for inference"""
        return (self.model is not None or self.pipeline is not None) and self.tokenizer is not None 