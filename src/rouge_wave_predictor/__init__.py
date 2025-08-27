"""
Rouge Wave Predictor - A comprehensive rouge wave prediction system
using machine learning and time series analysis
"""

__version__ = "0.1.0"

from .data_loader import WaveDataLoader
from .model_handler import GeospatialModelHandler
from .predictor import WavePredictor
from .visualizer import WaveVisualizer
from .utils import setup_logging, print_dependency_status, check_dependencies
from .demo import main as demo_main
from .rouge_wave_analysis import main as analysis_main

__all__ = [
    "WaveDataLoader",
    "GeospatialModelHandler", 
    "WavePredictor",
    "WaveVisualizer",
    "setup_logging",
    "print_dependency_status", 
    "check_dependencies",
    "demo_main",
    "analysis_main"
]
