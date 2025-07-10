#!/usr/bin/env python3
"""
Example implementation of a classification task.

This script demonstrates how an AI agent might approach a classification task
using the task system infrastructure.
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationTaskSolver:
    """Example solver for classification tasks."""
    
    def __init__(self, data_dir: Path, results_dir: Path):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load the classification dataset."""
        logger.info("Loading classification data...")
        
        # Look for data files
        data_files = list(self.data_dir.glob('*.csv'))
        
        if not data_files:
            raise FileNotFoundError("No CSV data files found")
        
        # Use the first CSV file
        data_file = data_files[0]
        logger.info(f"Loading data from: {data_file}")
        
        self.data = pd.read_csv(data_file)
        logger.info(f"Data shape: {self.data.shape}")
        
        return self.data
    
    def explore_data(self):
        """Perform basic data exploration."""
        logger.info("Exploring data...")
        
        # Basic statistics
        exploration = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'target_distribution': self.data['target'].value_counts().to_dict()
        }
        
        # Save exploration results
        with open(self.results_dir / 'data_exploration.json', 'w') as f:
            json.dump(exploration, f, indent=2, default=str)
        
        # Create visualizations
        self._create_exploration_plots()
        
        logger.info("Data exploration completed")
        return exploration
    
    def _create_exploration_plots(self):
        """Create data exploration visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Target distribution
        self.data['target'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        
        # Feature correlation heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = self.data[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
            axes[0, 1].set_title('Feature Correlation')
        
        # Feature distributions (first few features)
        feature_cols = [col for col in self.data.columns if col != 'target']
        if len(feature_cols) > 0:
            self.data[feature_cols[0]].hist(bins=30, ax=axes[1, 0])
            axes[1, 0].set_title(f'Distribution of {feature_cols[0]}')
        
        # Box plot for outlier detection
        if len(feature_cols) > 1:
            self.data[feature_cols[:5]].boxplot(ax=axes[1, 1])
            axes[1, 1].set_title('Feature Box Plots')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'data_exploration.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for training."""
        logger.info("Preparing data for training...")
        
        # Separate features and target
        feature_cols = [col for col in self.data.columns if col != 'target']
        X = self.data[feature_cols]
        y = self.data['target']
        
        self.feature_names = feature_cols
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Class distribution in training: {pd.Series(self.y_train).value_counts().to_dict()}")
    
    def train_model(self):
        """Train the classification model."""
        logger.info("Training classification model...")
        
        # Use Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train on scaled features
        self.model.fit(self.X_train_scaled, self.y_train)
        
        logger.info("Model training completed")
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred_train = self.model.predict(self.X_train_scaled)
        y_pred_test = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Detailed evaluation
        evaluation = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(self.y_test, y_pred_test, output_dict=True),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        
        # Save evaluation results
        with open(self.results_dir / 'model_evaluation.json', 'w') as f:
            json.dump(evaluation, f, indent=2, default=str)
        
        # Create evaluation plots
        self._create_evaluation_plots(y_pred_test)
        
        return evaluation
    
    def _create_evaluation_plots(self, y_pred_test):
        """Create evaluation visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        importance_df.plot(x='feature', y='importance', kind='barh', ax=axes[0, 1])
        axes[0, 1].set_title('Feature Importance')
        
        # Prediction confidence distribution
        y_proba = self.model.predict_proba(self.X_test_scaled)
        max_proba = np.max(y_proba, axis=1)
        
        axes[1, 0].hist(max_proba, bins=20, alpha=0.7)
        axes[1, 0].set_title('Prediction Confidence Distribution')
        axes[1, 0].set_xlabel('Max Probability')
        axes[1, 0].set_ylabel('Count')
        
        # Class-wise accuracy
        class_accuracies = []
        classes = sorted(self.y_test.unique())
        
        for cls in classes:
            mask = self.y_test == cls
            if mask.sum() > 0:
                cls_accuracy = accuracy_score(self.y_test[mask], y_pred_test[mask])
                class_accuracies.append(cls_accuracy)
            else:
                class_accuracies.append(0)
        
        axes[1, 1].bar(range(len(classes)), class_accuracies)
        axes[1, 1].set_title('Class-wise Accuracy')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_xticks(range(len(classes)))
        axes[1, 1].set_xticklabels(classes)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_predictions(self):
        """Save predictions in the required format."""
        logger.info("Saving predictions...")
        
        # Make predictions on test set
        y_pred_test = self.model.predict(self.X_test_scaled)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'true_labels': self.y_test.values,
            'predicted_labels': y_pred_test
        })
        
        # Save to CSV
        predictions_df.to_csv(self.results_dir / 'predictions.csv', index=False)
        logger.info("Predictions saved to predictions.csv")
    
    def create_model_summary(self):
        """Create a model summary report."""
        logger.info("Creating model summary...")
        
        summary = f"""Classification Model Summary
================================

Model Type: Random Forest Classifier
Number of Estimators: {self.model.n_estimators}
Max Depth: {self.model.max_depth}

Dataset Information:
- Total samples: {len(self.data)}
- Features: {len(self.feature_names)}
- Classes: {len(self.data['target'].unique())}
- Training samples: {len(self.X_train)}
- Test samples: {len(self.X_test)}

Performance Metrics:
- Training Accuracy: {accuracy_score(self.y_train, self.model.predict(self.X_train_scaled)):.4f}
- Test Accuracy: {accuracy_score(self.y_test, self.model.predict(self.X_test_scaled)):.4f}

Top 5 Most Important Features:
"""
        
        # Add feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(importance_df.head().iterrows()):
            summary += f"{i+1}. {row['feature']}: {row['importance']:.4f}\n"
        
        summary += f"""
Model Configuration:
- Random State: 42
- Feature Scaling: StandardScaler
- Cross-validation: Not performed (single train/test split)

Notes:
- Model trained on scaled features
- No hyperparameter tuning performed
- Results are deterministic due to fixed random state
"""
        
        # Save summary
        with open(self.results_dir / 'model_summary.txt', 'w') as f:
            f.write(summary)
        
        logger.info("Model summary saved to model_summary.txt")
    
    def run_complete_pipeline(self):
        """Run the complete classification pipeline."""
        logger.info("Starting classification task pipeline...")
        
        try:
            # Load and explore data
            self.load_data()
            self.explore_data()
            
            # Prepare data
            self.prepare_data()
            
            # Train model
            self.train_model()
            
            # Evaluate model
            evaluation = self.evaluate_model()
            
            # Save required outputs
            self.save_predictions()
            self.create_model_summary()
            
            logger.info("Classification task completed successfully!")
            logger.info(f"Final test accuracy: {evaluation['test_accuracy']:.4f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in classification pipeline: {e}")
            raise


def main():
    """Main function for running the classification example."""
    # Set up paths
    data_dir = Path('/home/data')
    results_dir = Path('/home/experiments/results')
    
    # Alternative paths for local testing
    if not data_dir.exists():
        data_dir = Path('./data')
    if not results_dir.exists():
        results_dir = Path('./results')
    
    # Create solver and run pipeline
    solver = ClassificationTaskSolver(data_dir, results_dir)
    evaluation = solver.run_complete_pipeline()
    
    print(f"Task completed with test accuracy: {evaluation['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()