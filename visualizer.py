"""
Visualization module for rouge wave analysis results
Creates charts and plots for wave data and predictions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class WaveVisualizer:
    """Creates visualizations for wave analysis results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.figsize = (12, 8)
        self.dpi = 300
        
        # Color schemes for different risk levels
        self.risk_colors = {
            'Low': '#2E8B57',      # Sea Green
            'Medium': '#FF8C00',    # Dark Orange
            'High': '#DC143C'       # Crimson
        }
        
        self.logger.info("Initialized WaveVisualizer")
    
    def create_wave_analysis_plots(self, wave_data: pd.DataFrame, 
                                 predictions: List[Dict[str, Any]], 
                                 output_dir: str):
        """
        Create comprehensive visualization suite for wave analysis
        
        Args:
            wave_data: Input wave data DataFrame
            predictions: List of prediction dictionaries
            output_dir: Directory to save plots
        """
        self.logger.info("Creating wave analysis visualizations")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert predictions to DataFrame for easier plotting
            pred_df = pd.DataFrame(predictions)
            
            # Create individual plots
            self._create_wave_height_distribution(wave_data, pred_df, output_dir)
            self._create_rouge_wave_probability_plot(pred_df, output_dir)
            self._create_geographic_risk_map(pred_df, output_dir)
            self._create_time_series_analysis(wave_data, pred_df, output_dir)
            self._create_risk_level_distribution(pred_df, output_dir)
            self._create_wave_characteristics_scatter(wave_data, pred_df, output_dir)
            self._create_confidence_analysis(pred_df, output_dir)
            
            # Create combined dashboard
            self._create_analysis_dashboard(wave_data, pred_df, output_dir)
            
            self.logger.info(f"All visualizations saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
    
    def _create_wave_height_distribution(self, wave_data: pd.DataFrame, 
                                       pred_df: pd.DataFrame, output_dir: str):
        """Create wave height distribution plots"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Input wave height distribution
            ax1.hist(wave_data['wave_height'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Wave Height (m)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Input Wave Height Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Predicted vs Input wave height comparison
            ax2.scatter(wave_data['wave_height'], pred_df['predicted_wave_height'], 
                       alpha=0.6, c=pred_df['rouge_wave_probability'], cmap='viridis')
            ax2.plot([0, max(wave_data['wave_height'])], [0, max(wave_data['wave_height'])], 
                    'r--', alpha=0.8, label='Perfect Prediction')
            ax2.set_xlabel('Input Wave Height (m)')
            ax2.set_ylabel('Predicted Wave Height (m)')
            ax2.set_title('Predicted vs Input Wave Heights')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            scatter = ax2.scatter(wave_data['wave_height'], pred_df['predicted_wave_height'], 
                                c=pred_df['rouge_wave_probability'], cmap='viridis')
            plt.colorbar(scatter, ax=ax2, label='Rouge Wave Probability')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'wave_height_distribution.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating wave height distribution plot: {str(e)}")
    
    def _create_rouge_wave_probability_plot(self, pred_df: pd.DataFrame, output_dir: str):
        """Create rouge wave probability analysis plots"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Rouge wave probability distribution
            ax1.hist(pred_df['rouge_wave_probability'], bins=20, alpha=0.7, 
                    color='coral', edgecolor='black')
            ax1.set_xlabel('Rouge Wave Probability')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Rouge Wave Probability Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Rouge wave probability by risk level
            risk_groups = pred_df.groupby('risk_level')['rouge_wave_probability']
            risk_means = risk_groups.mean()
            risk_counts = risk_groups.count()
            
            bars = ax2.bar(risk_means.index, risk_means.values, 
                          color=[self.risk_colors.get(risk, 'gray') for risk in risk_means.index])
            ax2.set_xlabel('Risk Level')
            ax2.set_ylabel('Average Rouge Wave Probability')
            ax2.set_title('Rouge Wave Probability by Risk Level')
            ax2.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bar, count in zip(bars, risk_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={count}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'rouge_wave_probability.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating rouge wave probability plot: {str(e)}")
    
    def _create_geographic_risk_map(self, pred_df: pd.DataFrame, output_dir: str):
        """Create geographic risk visualization"""
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create scatter plot with color coding by risk level
            for risk_level in ['Low', 'Medium', 'High']:
                risk_data = pred_df[pred_df['risk_level'] == risk_level]
                if not risk_data.empty:
                    ax.scatter(risk_data['longitude'], risk_data['latitude'], 
                             c=self.risk_colors.get(risk_level, 'gray'),
                             s=50, alpha=0.7, label=f'{risk_level} Risk',
                             edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            ax.set_title('Geographic Distribution of Wave Risk Levels')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add ocean basin labels
            ax.text(-70, 45, 'North Atlantic', fontsize=12, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'geographic_risk_map.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating geographic risk map: {str(e)}")
    
    def _create_time_series_analysis(self, wave_data: pd.DataFrame, 
                                   pred_df: pd.DataFrame, output_dir: str):
        """Create time series analysis plots"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(wave_data['timestamp']):
                wave_data['timestamp'] = pd.to_datetime(wave_data['timestamp'])
            
            # Time series of wave heights
            ax1.plot(wave_data['timestamp'], wave_data['wave_height'], 
                    alpha=0.7, label='Input Wave Height', color='blue')
            ax1.plot(wave_data['timestamp'], pred_df['predicted_wave_height'], 
                    alpha=0.7, label='Predicted Wave Height', color='red')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Wave Height (m)')
            ax1.set_title('Wave Height Time Series')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Time series of rouge wave probability
            ax2.plot(wave_data['timestamp'], pred_df['rouge_wave_probability'], 
                    alpha=0.8, color='coral', linewidth=2)
            ax2.fill_between(wave_data['timestamp'], pred_df['rouge_wave_probability'], 
                           alpha=0.3, color='coral')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Rouge Wave Probability')
            ax2.set_title('Rouge Wave Probability Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Add risk level thresholds
            ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
            ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Risk Threshold')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'time_series_analysis.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating time series analysis: {str(e)}")
    
    def _create_risk_level_distribution(self, pred_df: pd.DataFrame, output_dir: str):
        """Create risk level distribution visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Pie chart of risk levels
            risk_counts = pred_df['risk_level'].value_counts()
            colors = [self.risk_colors.get(risk, 'gray') for risk in risk_counts.index]
            
            wedges, texts, autotexts = ax1.pie(risk_counts.values, labels=risk_counts.index, 
                                              colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Distribution of Risk Levels')
            
            # Bar chart of risk levels with counts
            bars = ax2.bar(risk_counts.index, risk_counts.values, 
                          color=[self.risk_colors.get(risk, 'gray') for risk in risk_counts.index])
            ax2.set_xlabel('Risk Level')
            ax2.set_ylabel('Count')
            ax2.set_title('Risk Level Counts')
            ax2.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'risk_level_distribution.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating risk level distribution: {str(e)}")
    
    def _create_wave_characteristics_scatter(self, wave_data: pd.DataFrame, 
                                           pred_df: pd.DataFrame, output_dir: str):
        """Create scatter plots of wave characteristics"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Wave height vs wave period
            scatter1 = ax1.scatter(wave_data['wave_height'], wave_data['wave_period'], 
                                 c=pred_df['rouge_wave_probability'], cmap='viridis', alpha=0.7)
            ax1.set_xlabel('Wave Height (m)')
            ax1.set_ylabel('Wave Period (s)')
            ax1.set_title('Wave Height vs Period')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter1, ax=ax1, label='Rouge Wave Probability')
            
            # Wind speed vs wave height
            scatter2 = ax2.scatter(wave_data['wind_speed'], wave_data['wave_height'], 
                                 c=pred_df['rouge_wave_probability'], cmap='viridis', alpha=0.7)
            ax2.set_xlabel('Wind Speed (m/s)')
            ax2.set_ylabel('Wave Height (m)')
            ax2.set_title('Wind Speed vs Wave Height')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='Rouge Wave Probability')
            
            # Wave steepness vs rouge wave probability
            wave_steepness = wave_data['wave_height'] / (wave_data['wave_period'] + 1e-8)
            ax3.scatter(wave_steepness, pred_df['rouge_wave_probability'], 
                       alpha=0.7, color='purple')
            ax3.set_xlabel('Wave Steepness (H/T)')
            ax3.set_ylabel('Rouge Wave Probability')
            ax3.set_title('Wave Steepness vs Rouge Wave Probability')
            ax3.grid(True, alpha=0.3)
            
            # Confidence score vs rouge wave probability
            ax4.scatter(pred_df['confidence_score'], pred_df['rouge_wave_probability'], 
                       alpha=0.7, color='green')
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Rouge Wave Probability')
            ax4.set_title('Confidence vs Rouge Wave Probability')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'wave_characteristics_scatter.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating wave characteristics scatter: {str(e)}")
    
    def _create_confidence_analysis(self, pred_df: pd.DataFrame, output_dir: str):
        """Create confidence score analysis"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Confidence score distribution
            ax1.hist(pred_df['confidence_score'], bins=20, alpha=0.7, 
                    color='lightgreen', edgecolor='black')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Confidence Score Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Confidence by risk level
            confidence_by_risk = pred_df.groupby('risk_level')['confidence_score'].mean()
            bars = ax2.bar(confidence_by_risk.index, confidence_by_risk.values,
                          color=[self.risk_colors.get(risk, 'gray') for risk in confidence_by_risk.index])
            ax2.set_xlabel('Risk Level')
            ax2.set_ylabel('Average Confidence Score')
            ax2.set_title('Average Confidence Score by Risk Level')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating confidence analysis: {str(e)}")
    
    def _create_analysis_dashboard(self, wave_data: pd.DataFrame, 
                                 pred_df: pd.DataFrame, output_dir: str):
        """Create a comprehensive dashboard combining key visualizations"""
        try:
            fig = plt.figure(figsize=(20, 16))
            
            # Create grid layout
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # 1. Wave height distribution (top left)
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.hist(wave_data['wave_height'], bins=25, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Wave Height (m)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Wave Height Distribution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 2. Rouge wave probability over time (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            if not pd.api.types.is_datetime64_any_dtype(wave_data['timestamp']):
                wave_data['timestamp'] = pd.to_datetime(wave_data['timestamp'])
            ax2.plot(wave_data['timestamp'], pred_df['rouge_wave_probability'], 
                    alpha=0.8, color='coral', linewidth=2)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Rouge Wave Probability')
            ax2.set_title('Rouge Wave Probability Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 3. Risk level distribution (middle left)
            ax3 = fig.add_subplot(gs[1, :2])
            risk_counts = pred_df['risk_level'].value_counts()
            colors = [self.risk_colors.get(risk, 'gray') for risk in risk_counts.index]
            ax3.pie(risk_counts.values, labels=risk_counts.index, colors=colors, 
                   autopct='%1.1f%%', startangle=90)
            ax3.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
            
            # 4. Geographic risk map (middle right)
            ax4 = fig.add_subplot(gs[1, 2:])
            for risk_level in ['Low', 'Medium', 'High']:
                risk_data = pred_df[pred_df['risk_level'] == risk_level]
                if not risk_data.empty:
                    ax4.scatter(risk_data['longitude'], risk_data['latitude'], 
                              c=self.risk_colors.get(risk_level, 'gray'),
                              s=30, alpha=0.7, label=f'{risk_level} Risk')
            ax4.set_xlabel('Longitude (°E)')
            ax4.set_ylabel('Latitude (°N)')
            ax4.set_title('Geographic Risk Distribution', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Wave height vs period scatter (bottom left)
            ax5 = fig.add_subplot(gs[2, :2])
            scatter = ax5.scatter(wave_data['wave_height'], wave_data['wave_period'], 
                                c=pred_df['rouge_wave_probability'], cmap='viridis', alpha=0.7)
            ax5.set_xlabel('Wave Height (m)')
            ax5.set_ylabel('Wave Period (s)')
            ax5.set_title('Wave Height vs Period', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax5, label='Rouge Wave Probability')
            
            # 6. Confidence analysis (bottom right)
            ax6 = fig.add_subplot(gs[2, 2:])
            confidence_by_risk = pred_df.groupby('risk_level')['confidence_score'].mean()
            bars = ax6.bar(confidence_by_risk.index, confidence_by_risk.values,
                          color=[self.risk_colors.get(risk, 'gray') for risk in confidence_by_risk.index])
            ax6.set_xlabel('Risk Level')
            ax6.set_ylabel('Average Confidence Score')
            ax6.set_title('Confidence by Risk Level', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # 7. Summary statistics (bottom full width)
            ax7 = fig.add_subplot(gs[3, :])
            ax7.axis('off')
            
            # Calculate summary statistics
            total_samples = len(pred_df)
            high_risk_count = len(pred_df[pred_df['risk_level'] == 'High'])
            avg_rouge_prob = pred_df['rouge_wave_probability'].mean()
            avg_confidence = pred_df['confidence_score'].mean()
            
            summary_text = f"""
            ANALYSIS SUMMARY
            ================
            Total Samples Analyzed: {total_samples:,}
            High Risk Conditions: {high_risk_count:,} ({high_risk_count/total_samples*100:.1f}%)
            Average Rouge Wave Probability: {avg_rouge_prob:.3f}
            Average Confidence Score: {avg_confidence:.3f}
            
            Risk Level Breakdown:
            • Low Risk: {len(pred_df[pred_df['risk_level'] == 'Low']):,} samples
            • Medium Risk: {len(pred_df[pred_df['risk_level'] == 'Medium']):,} samples  
            • High Risk: {high_risk_count:,} samples
            """
            
            ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.suptitle('Rouge Wave Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
            plt.savefig(os.path.join(output_dir, 'analysis_dashboard.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating analysis dashboard: {str(e)}") 