#!/usr/bin/env python3
"""
Demo script for Rouge Wave Analysis using IBM-NASA Geospatial Models
This script demonstrates the basic functionality of the system
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils import setup_logging, print_dependency_status, check_dependencies
from data_loader import WaveDataLoader
from model_handler import GeospatialModelHandler
from predictor import WavePredictor
from visualizer import WaveVisualizer

def main():
    """Run the demo analysis"""
    
    print("=" * 60)
    print("ROUGE WAVE ANALYSIS DEMO")
    print("IBM-NASA Geospatial Models Integration")
    print("=" * 60)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    print_dependency_status()
    
    deps = check_dependencies()
    if not all(deps.values()):
        print("\n⚠️  Some dependencies are missing. The demo may not work correctly.")
        print("Install missing dependencies with: pip install -r requirements.txt")
    
    # Setup logging
    print("\n2. Setting up logging...")
    logger = setup_logging(level="INFO")
    
    try:
        # Step 1: Create sample data
        print("\n3. Creating sample wave data...")
        data_loader = WaveDataLoader()
        sample_data = data_loader._create_sample_data(100)  # Create 100 sample records
        print(f"✓ Created {len(sample_data)} sample wave data records")
        
        # Step 2: Preprocess data
        print("\n4. Preprocessing wave data...")
        processed_data = data_loader.preprocess_data(sample_data)
        print(f"✓ Preprocessed data with {processed_data.shape[1]} features")
        
        # Step 3: Initialize model (with fallback)
        print("\n5. Initializing AI model...")
        model_handler = GeospatialModelHandler("ibm-nasa-geospatial/Prithvi-100M")
        model = model_handler.load_model(use_fallback=True)
        
        if model is not None:
            print(f"✓ Model loaded successfully: {model_handler.model_name}")
            model_info = model_handler.get_model_info()
            print(f"  - Model type: {model_info.get('model_type', 'Unknown')}")
            print(f"  - Parameters: {model_info.get('model_parameters', 'Unknown'):,}")
        else:
            print("⚠️  Model loading failed, using rule-based fallback")
        
        # Step 4: Generate predictions
        print("\n6. Generating predictions...")
        predictor = WavePredictor(model, model_handler.tokenizer)
        predictions = predictor.predict(processed_data, batch_size=16)
        print(f"✓ Generated {len(predictions)} predictions")
        
        # Step 5: Create visualizations
        print("\n7. Creating visualizations...")
        visualizer = WaveVisualizer()
        output_dir = "demo_outputs"
        visualizer.create_wave_analysis_plots(processed_data, predictions, output_dir)
        print(f"✓ Visualizations saved to {output_dir}/")
        
        # Step 6: Save results
        print("\n8. Saving results...")
        predictor.save_predictions(predictions, output_dir)
        print(f"✓ Results saved to {output_dir}/")
        
        # Step 7: Generate report
        print("\n9. Generating analysis report...")
        report = predictor.generate_report(processed_data, predictions)
        
        report_path = os.path.join(output_dir, 'demo_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to {report_path}")
        
        # Display summary
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if predictions:
            # Calculate summary statistics
            rouge_probs = [p.get('rouge_wave_probability', 0) for p in predictions]
            risk_levels = [p.get('risk_level', 'Unknown') for p in predictions]
            confidence_scores = [p.get('confidence_score', 0) for p in predictions]
            
            print(f"\n📊 ANALYSIS SUMMARY:")
            print(f"   • Total samples analyzed: {len(predictions):,}")
            print(f"   • Average rouge wave probability: {sum(rouge_probs)/len(rouge_probs):.3f}")
            print(f"   • High risk conditions: {risk_levels.count('High'):,}")
            print(f"   • Medium risk conditions: {risk_levels.count('Medium'):,}")
            print(f"   • Low risk conditions: {risk_levels.count('Low'):,}")
            print(f"   • Average confidence: {sum(confidence_scores)/len(confidence_scores):.3f}")
        
        print(f"\n📁 Output files:")
        print(f"   • Predictions: {output_dir}/wave_predictions.csv")
        print(f"   • Visualizations: {output_dir}/*.png")
        print(f"   • Report: {output_dir}/demo_report.txt")
        
        print(f"\n🚀 To run the full analysis:")
        print(f"   python rouge_wave_analysis.py --max_samples 1000")
        
        print(f"\n🔧 To customize the analysis:")
        print(f"   python rouge_wave_analysis.py --help")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Check the logs for more details.")
        return 1
    
    return 0

def quick_test():
    """Quick functionality test without full analysis"""
    print("Quick functionality test...")
    
    try:
        # Test data loader
        data_loader = WaveDataLoader()
        sample_data = data_loader._create_sample_data(10)
        print(f"✓ Data loader: Created {len(sample_data)} samples")
        
        # Test model handler
        model_handler = GeospatialModelHandler()
        print(f"✓ Model handler: Initialized for {model_handler.model_name}")
        
        # Test predictor
        predictor = WavePredictor(None, None)
        print(f"✓ Predictor: Initialized")
        
        # Test visualizer
        visualizer = WaveVisualizer()
        print(f"✓ Visualizer: Initialized")
        
        print("✓ All components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = quick_test()
        sys.exit(0 if success else 1)
    else:
        exit_code = main()
        sys.exit(exit_code) 