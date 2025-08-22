#!/usr/bin/env python3
"""
Rouge Wave Analysis using IBM-NASA Geospatial Models
Main script for orchestrating the analysis pipeline
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_loader import WaveDataLoader
from model_handler import GeospatialModelHandler
from predictor import WavePredictor
from visualizer import WaveVisualizer
from utils import setup_logging, create_output_dirs

def main():
    """Main function to run the rouge wave analysis pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Rouge Wave Analysis with IBM-NASA Geospatial Models')
    parser.add_argument('--model_name', 
                       default='ibm-nasa-geospatial/wave-height-predictor',
                       help='Hugging Face model name to use')
    parser.add_argument('--data_path', 
                       default='data/sample_wave_data.csv',
                       help='Path to wave data CSV file')
    parser.add_argument('--output_dir', 
                       default='outputs',
                       help='Output directory for results')
    parser.add_argument('--batch_size', 
                       type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--max_samples', 
                       type=int, default=1000,
                       help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Rouge Wave Analysis Pipeline")
    
    try:
        # Create output directories
        create_output_dirs(args.output_dir)
        
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing wave data")
        data_loader = WaveDataLoader()
        wave_data = data_loader.load_data(args.data_path, max_samples=args.max_samples)
        processed_data = data_loader.preprocess_data(wave_data)
        
        logger.info(f"Loaded {len(processed_data)} wave data samples")
        
        # Step 2: Initialize model
        logger.info("Step 2: Initializing geospatial model")
        model_handler = GeospatialModelHandler(args.model_name)
        model = model_handler.load_model()
        
        # Step 3: Make predictions
        logger.info("Step 3: Generating predictions")
        predictor = WavePredictor(model, model_handler.tokenizer)
        predictions = predictor.predict(processed_data, batch_size=args.batch_size)
        
        # Step 4: Visualize results
        logger.info("Step 4: Creating visualizations")
        visualizer = WaveVisualizer()
        visualizer.create_wave_analysis_plots(processed_data, predictions, args.output_dir)
        
        # Step 5: Save results
        logger.info("Step 5: Saving results")
        predictor.save_predictions(predictions, args.output_dir)
        
        # Step 6: Generate report
        logger.info("Step 6: Generating analysis report")
        report = predictor.generate_report(processed_data, predictions)
        
        with open(os.path.join(args.output_dir, 'analysis_report.txt'), 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis complete! Results saved to {args.output_dir}")
        logger.info(f"Report: {os.path.join(args.output_dir, 'analysis_report.txt')}")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 