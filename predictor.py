"""
Wave prediction module using IBM-NASA Geospatial models
Handles prediction generation and analysis for rouge wave detection
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
import warnings

class WavePredictor:
    """Handles wave predictions using loaded models"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        self.predictions = []
        
        self.logger.info("Initialized WavePredictor")
    
    def predict(self, wave_data: pd.DataFrame, batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Generate predictions for wave data
        
        Args:
            wave_data: Preprocessed wave data DataFrame
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        self.logger.info(f"Generating predictions for {len(wave_data)} wave data samples")
        
        predictions = []
        
        try:
            # Process data in batches
            for i in range(0, len(wave_data), batch_size):
                batch = wave_data.iloc[i:i+batch_size]
                self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(wave_data) + batch_size - 1)//batch_size}")
                
                batch_predictions = self._predict_batch(batch)
                predictions.extend(batch_predictions)
            
            self.predictions = predictions
            self.logger.info(f"Successfully generated {len(predictions)} predictions")
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            # Generate fallback predictions
            predictions = self._generate_fallback_predictions(wave_data)
            self.predictions = predictions
        
        return predictions
    
    def _predict_batch(self, batch_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a batch of wave data"""
        batch_predictions = []
        
        for _, row in batch_data.iterrows():
            try:
                # Convert row to dictionary
                wave_params = row.to_dict()
                
                # Generate prediction for this wave data point
                prediction = self._predict_single_wave(wave_params)
                batch_predictions.append(prediction)
                
            except Exception as e:
                self.logger.warning(f"Error predicting for row {row.name}: {str(e)}")
                # Create fallback prediction
                fallback_pred = self._create_fallback_prediction(row.to_dict())
                batch_predictions.append(fallback_pred)
        
        return batch_predictions
    
    def _predict_single_wave(self, wave_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for a single wave data point"""
        
        # Extract key parameters
        timestamp = wave_params.get('timestamp', 'Unknown')
        latitude = wave_params.get('latitude', 0)
        longitude = wave_params.get('longitude', 0)
        wave_height = wave_params.get('wave_height', 0)
        wave_period = wave_params.get('wave_period', 0)
        wind_speed = wave_params.get('wind_speed', 0)
        wind_direction = wave_params.get('wind_direction', 0)
        
        # Create prediction using the model
        try:
            if hasattr(self.model, 'generate') or hasattr(self.model, '__call__'):
                # Use the model for prediction
                model_prediction = self._get_model_prediction(wave_params)
            else:
                # Fallback to rule-based prediction
                model_prediction = self._rule_based_prediction(wave_params)
            
            # Create comprehensive prediction result
            prediction = {
                'timestamp': timestamp,
                'latitude': latitude,
                'longitude': longitude,
                'input_wave_height': wave_height,
                'input_wave_period': wave_period,
                'input_wind_speed': wind_speed,
                'input_wind_direction': wind_direction,
                'predicted_wave_height': model_prediction.get('predicted_wave_height', wave_height),
                'predicted_wave_period': model_prediction.get('predicted_wave_period', wave_period),
                'rouge_wave_probability': model_prediction.get('rouge_wave_probability', 0.0),
                'risk_level': model_prediction.get('risk_level', 'Low'),
                'confidence_score': model_prediction.get('confidence_score', 0.5),
                'analysis_notes': model_prediction.get('analysis_notes', ''),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in model prediction: {str(e)}")
            return self._create_fallback_prediction(wave_params)
    
    def _get_model_prediction(self, wave_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from the loaded model"""
        try:
            # Prepare input for the model
            if hasattr(self.model, 'prepare_input_for_wave_analysis'):
                input_text = self.model.prepare_input_for_wave_analysis(wave_params)
            else:
                input_text = self._create_wave_description(wave_params)
            
            # Generate prediction using the model
            if hasattr(self.model, 'generate'):
                # For text generation models
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=200,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = self._parse_model_output(generated_text, wave_params)
                
            elif hasattr(self.model, '__call__'):
                # For models that support direct calling
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                prediction = self._parse_model_outputs(outputs, wave_params)
                
            else:
                # Fallback to rule-based prediction
                prediction = self._rule_based_prediction(wave_params)
            
            return prediction
            
        except Exception as e:
            self.logger.warning(f"Model prediction failed: {str(e)}")
            return self._rule_based_prediction(wave_params)
    
    def _rule_based_prediction(self, wave_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction using rule-based logic when model fails"""
        
        wave_height = wave_params.get('wave_height', 0)
        wave_period = wave_params.get('wave_period', 0)
        wind_speed = wave_params.get('wind_speed', 0)
        
        # Calculate wave steepness
        wave_steepness = wave_height / (wave_period + 1e-8)
        
        # Rule-based rouge wave detection
        rouge_probability = 0.0
        risk_level = 'Low'
        
        if wave_height > 8:  # Very high waves
            rouge_probability = 0.8
            risk_level = 'High'
        elif wave_height > 6:  # High waves
            rouge_probability = 0.6
            risk_level = 'Medium'
        elif wave_height > 4:  # Moderate waves
            rouge_probability = 0.3
            risk_level = 'Low'
        
        # Adjust based on wave steepness
        if wave_steepness > 0.1:  # Steep waves
            rouge_probability = min(rouge_probability + 0.2, 1.0)
            if risk_level == 'Low':
                risk_level = 'Medium'
        
        # Adjust based on wind conditions
        if wind_speed > 20:  # High winds
            rouge_probability = min(rouge_probability + 0.1, 1.0)
        
        # Generate predicted values with some variation
        predicted_wave_height = wave_height * np.random.normal(1.0, 0.1)
        predicted_wave_period = wave_period * np.random.normal(1.0, 0.05)
        
        # Create analysis notes
        analysis_notes = self._generate_analysis_notes(wave_params, rouge_probability, risk_level)
        
        return {
            'predicted_wave_height': max(0, predicted_wave_height),
            'predicted_wave_period': max(0, predicted_wave_period),
            'rouge_wave_probability': rouge_probability,
            'risk_level': risk_level,
            'confidence_score': 0.7,  # Rule-based predictions have moderate confidence
            'analysis_notes': analysis_notes
        }
    
    def _create_wave_description(self, wave_params: Dict[str, Any]) -> str:
        """Create a text description of wave conditions"""
        return f"""
        Wave conditions at {wave_params.get('latitude', 'N/A')}°N, {wave_params.get('longitude', 'N/A')}°E:
        - Wave height: {wave_params.get('wave_height', 'N/A')} meters
        - Wave period: {wave_params.get('wave_period', 'N/A')} seconds
        - Wind speed: {wave_params.get('wind_speed', 'N/A')} m/s
        - Wind direction: {wave_params.get('wind_direction', 'N/A')}°
        - Timestamp: {wave_params.get('timestamp', 'N/A')}
        """
    
    def _parse_model_output(self, generated_text: str, wave_params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the model's text output to extract predictions"""
        # This is a simplified parser - in practice, you'd want more sophisticated parsing
        try:
            # Extract numerical values from the generated text
            import re
            
            # Look for wave height predictions
            height_match = re.search(r'wave height.*?(\d+\.?\d*)', generated_text.lower())
            predicted_height = float(height_match.group(1)) if height_match else wave_params.get('wave_height', 0)
            
            # Look for risk assessments
            if 'high risk' in generated_text.lower() or 'dangerous' in generated_text.lower():
                risk_level = 'High'
                rouge_probability = 0.8
            elif 'medium risk' in generated_text.lower() or 'caution' in generated_text.lower():
                risk_level = 'Medium'
                rouge_probability = 0.5
            else:
                risk_level = 'Low'
                rouge_probability = 0.2
            
            return {
                'predicted_wave_height': predicted_height,
                'predicted_wave_period': wave_params.get('wave_period', 0),
                'rouge_wave_probability': rouge_probability,
                'risk_level': risk_level,
                'confidence_score': 0.8,
                'analysis_notes': generated_text[:200] + '...' if len(generated_text) > 200 else generated_text
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing model output: {str(e)}")
            return self._rule_based_prediction(wave_params)
    
    def _parse_model_outputs(self, outputs, wave_params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse model outputs for different model types"""
        try:
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                # For classification models
                if logits.dim() == 2:
                    probabilities = torch.softmax(logits, dim=-1)
                    rouge_probability = probabilities[0][1].item() if logits.shape[1] > 1 else 0.5
                else:
                    rouge_probability = 0.5
            else:
                rouge_probability = 0.5
            
            # Determine risk level based on probability
            if rouge_probability > 0.7:
                risk_level = 'High'
            elif rouge_probability > 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            return {
                'predicted_wave_height': wave_params.get('wave_height', 0),
                'predicted_wave_period': wave_params.get('wave_period', 0),
                'rouge_wave_probability': rouge_probability,
                'risk_level': risk_level,
                'confidence_score': 0.8,
                'analysis_notes': f'Model output analysis with {rouge_probability:.2f} rouge wave probability'
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing model outputs: {str(e)}")
            return self._rule_based_prediction(wave_params)
    
    def _generate_analysis_notes(self, wave_params: Dict[str, Any], rouge_probability: float, risk_level: str) -> str:
        """Generate human-readable analysis notes"""
        notes = []
        
        wave_height = wave_params.get('wave_height', 0)
        wave_period = wave_params.get('wave_period', 0)
        wind_speed = wave_params.get('wind_speed', 0)
        
        if rouge_probability > 0.7:
            notes.append("HIGH RISK: Conditions favorable for rouge wave formation")
        elif rouge_probability > 0.4:
            notes.append("MODERATE RISK: Monitor conditions for potential rouge waves")
        else:
            notes.append("LOW RISK: Normal wave conditions expected")
        
        if wave_height > 6:
            notes.append(f"Large waves observed ({wave_height:.1f}m) - exercise caution")
        
        if wind_speed > 20:
            notes.append(f"High wind conditions ({wind_speed:.1f} m/s) may amplify wave activity")
        
        if wave_period < 6:
            notes.append("Short wave periods indicate choppy conditions")
        
        return "; ".join(notes)
    
    def _create_fallback_prediction(self, wave_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback prediction when all else fails"""
        return {
            'timestamp': wave_params.get('timestamp', 'Unknown'),
            'latitude': wave_params.get('latitude', 0),
            'longitude': wave_params.get('longitude', 0),
            'input_wave_height': wave_params.get('wave_height', 0),
            'input_wave_period': wave_params.get('wave_period', 0),
            'input_wind_speed': wave_params.get('wind_speed', 0),
            'input_wind_direction': wave_params.get('wind_direction', 0),
            'predicted_wave_height': wave_params.get('wave_height', 0),
            'predicted_wave_period': wave_params.get('wave_period', 0),
            'rouge_wave_probability': 0.1,
            'risk_level': 'Low',
            'confidence_score': 0.3,
            'analysis_notes': 'Fallback prediction - model analysis unavailable',
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def _generate_fallback_predictions(self, wave_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate fallback predictions for the entire dataset"""
        self.logger.info("Generating fallback predictions for entire dataset")
        fallback_predictions = []
        
        for _, row in wave_data.iterrows():
            fallback_pred = self._create_fallback_prediction(row.to_dict())
            fallback_predictions.append(fallback_pred)
        
        return fallback_predictions
    
    def save_predictions(self, predictions: List[Dict[str, Any]], output_dir: str):
        """Save predictions to files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as CSV
            df = pd.DataFrame(predictions)
            csv_path = os.path.join(output_dir, 'wave_predictions.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Predictions saved to CSV: {csv_path}")
            
            # Save as JSON
            json_path = os.path.join(output_dir, 'wave_predictions.json')
            with open(json_path, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            self.logger.info(f"Predictions saved to JSON: {json_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
    
    def generate_report(self, input_data: pd.DataFrame, predictions: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive analysis report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("ROUGE WAVE ANALYSIS REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Input Data Samples: {len(input_data)}")
            report.append(f"Predictions Generated: {len(predictions)}")
            report.append("")
            
            # Summary statistics
            if predictions:
                rouge_probs = [p.get('rouge_wave_probability', 0) for p in predictions]
                risk_levels = [p.get('risk_level', 'Unknown') for p in predictions]
                confidence_scores = [p.get('confidence_score', 0) for p in predictions]
                
                report.append("PREDICTION SUMMARY:")
                report.append(f"- Average Rouge Wave Probability: {np.mean(rouge_probs):.3f}")
                report.append(f"- High Risk Predictions: {risk_levels.count('High')}")
                report.append(f"- Medium Risk Predictions: {risk_levels.count('Medium')}")
                report.append(f"- Low Risk Predictions: {risk_levels.count('Low')}")
                report.append(f"- Average Confidence Score: {np.mean(confidence_scores):.3f}")
                report.append("")
            
            # Input data summary
            if not input_data.empty:
                report.append("INPUT DATA SUMMARY:")
                report.append(f"- Wave Height Range: {input_data['wave_height'].min():.2f} - {input_data['wave_height'].max():.2f} m")
                report.append(f"- Wave Period Range: {input_data['wave_period'].min():.2f} - {input_data['wave_period'].max():.2f} s")
                report.append(f"- Wind Speed Range: {input_data['wind_speed'].min():.2f} - {input_data['wind_speed'].max():.2f} m/s")
                report.append(f"- Geographic Coverage: {input_data['latitude'].min():.2f}° to {input_data['latitude'].max():.2f}° N, "
                           f"{input_data['longitude'].min():.2f}° to {input_data['longitude'].max():.2f}° E")
                report.append("")
            
            # Recommendations
            report.append("RECOMMENDATIONS:")
            if predictions:
                high_risk_count = sum(1 for p in predictions if p.get('risk_level') == 'High')
                if high_risk_count > 0:
                    report.append(f"- ⚠️  {high_risk_count} high-risk conditions detected - immediate attention required")
                else:
                    report.append("- ✅ No high-risk conditions detected")
                
                avg_rouge_prob = np.mean(rouge_probs)
                if avg_rouge_prob > 0.5:
                    report.append("- 🚨 Elevated rouge wave risk - monitor conditions closely")
                elif avg_rouge_prob > 0.3:
                    report.append("- ⚠️  Moderate rouge wave risk - exercise caution")
                else:
                    report.append("- ✅ Low rouge wave risk - normal operations")
            
            report.append("")
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}" 