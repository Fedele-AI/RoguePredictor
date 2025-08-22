"""
Utility functions for rouge wave analysis
Helper functions for logging, file operations, and common tasks
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml
from datetime import datetime

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Set up basic configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level: {level}")
    
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger

def create_output_dirs(output_dir: str, subdirs: Optional[list] = None) -> str:
    """
    Create output directories for analysis results
    
    Args:
        output_dir: Main output directory path
        subdirs: List of subdirectory names to create
        
    Returns:
        Path to the created output directory
    """
    try:
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories if specified
        if subdirs:
            for subdir in subdirs:
                subdir_path = os.path.join(output_dir, subdir)
                os.makedirs(subdir_path, exist_ok=True)
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir
        
    except Exception as e:
        logging.error(f"Error creating output directories: {str(e)}")
        # Fallback to main output directory
        return output_dir

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            logging.warning(f"Config file {config_path} not found, using defaults")
            return get_default_config()
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logging.warning(f"Unsupported config file format: {config_path.suffix}")
            return get_default_config()
        
        logging.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration values"""
    return {
        "model": {
            "name": "ibm-nasa-geospatial/wave-height-predictor",
            "cache_dir": "./model_cache",
            "trust_remote_code": True,
            "use_fallback": True
        },
        "data": {
            "max_samples": 1000,
            "batch_size": 32,
            "required_columns": [
                "timestamp", "latitude", "longitude",
                "wave_height", "wave_period", "wind_speed", "wind_direction"
            ]
        },
        "analysis": {
            "rouge_wave_thresholds": {
                "high": 0.7,
                "medium": 0.4,
                "low": 0.1
            },
            "confidence_threshold": 0.5
        },
        "output": {
            "format": ["csv", "json"],
            "include_plots": True,
            "plot_dpi": 300,
            "plot_format": "png"
        }
    }

def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    try:
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        if output_path.suffix.lower() == '.yaml' or output_path.suffix.lower() == '.yml':
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            logging.warning(f"Unsupported output format: {output_path.suffix}")
            return
        
        logging.info(f"Configuration saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error saving configuration: {str(e)}")

def validate_data_format(data: Dict[str, Any], required_fields: list) -> bool:
    """
    Validate that data contains required fields
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        True if validation passes, False otherwise
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        logging.warning(f"Missing required fields: {missing_fields}")
        return False
    
    return True

def format_timestamp(timestamp) -> str:
    """
    Format timestamp to consistent string format
    
    Args:
        timestamp: Timestamp value (datetime, string, or other)
        
    Returns:
        Formatted timestamp string
    """
    try:
        if isinstance(timestamp, str):
            # Try to parse string timestamp
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif hasattr(timestamp, 'isoformat'):
            # Handle datetime objects
            dt = timestamp
        else:
            # Fallback to current time
            dt = datetime.now()
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception as e:
        logging.warning(f"Error formatting timestamp {timestamp}: {str(e)}")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_statistics(data: list, field: str) -> Dict[str, float]:
    """
    Calculate basic statistics for a data field
    
    Args:
        data: List of dictionaries containing data
        field: Field name to calculate statistics for
        
    Returns:
        Dictionary with statistics (min, max, mean, std)
    """
    try:
        values = [item.get(field, 0) for item in data if item.get(field) is not None]
        
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}
        
        import numpy as np
        values = np.array(values)
        
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
        
    except Exception as e:
        logging.warning(f"Error calculating statistics for {field}: {str(e)}")
        return {"min": 0, "max": 0, "mean": 0, "std": 0}

def create_progress_bar(total: int, description: str = "Processing"):
    """
    Create a progress bar for long-running operations
    
    Args:
        total: Total number of items to process
        description: Description of the operation
        
    Returns:
        Progress bar object
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=description, unit="items")
    except ImportError:
        # Fallback to simple progress indicator
        class SimpleProgressBar:
            def __init__(self, total, desc="Processing"):
                self.total = total
                self.current = 0
                self.desc = desc
            
            def update(self, n=1):
                self.current += n
                if self.current % 100 == 0 or self.current == self.total:
                    progress = (self.current / self.total) * 100
                    print(f"{self.desc}: {progress:.1f}% ({self.current}/{self.total})")
            
            def close(self):
                print(f"{self.desc}: Complete!")
        
        return SimpleProgressBar(total, description)

def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are available
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {
        "transformers": False,
        "torch": False,
        "pandas": False,
        "numpy": False,
        "matplotlib": False,
        "seaborn": False,
        "scikit-learn": False
    }
    
    for dep in dependencies.keys():
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

def print_dependency_status():
    """Print the status of required dependencies"""
    deps = check_dependencies()
    
    print("Dependency Status:")
    print("=" * 50)
    
    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"{dep:15} : {status}")
    
    print("=" * 50)
    
    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
    else:
        print("\nAll dependencies are available!")

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    try:
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        return 0.0
    except Exception:
        return 0.0

def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24):
    """
    Clean up temporary files older than specified age
    
    Args:
        temp_dir: Directory containing temporary files
        max_age_hours: Maximum age in hours before cleanup
    """
    try:
        if not os.path.exists(temp_dir):
            return
        
        current_time = datetime.now()
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    os.remove(file_path)
                    logging.debug(f"Cleaned up old temp file: {filename}")
                    
    except Exception as e:
        logging.warning(f"Error during temp file cleanup: {str(e)}") 