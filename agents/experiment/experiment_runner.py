#!/usr/bin/env python3
"""
Experiment Runner for AI Agents

This script provides a flexible framework for AI agents to run various kinds of experiments
and tasks. It supports different task types and provides utilities for data handling,
model training, evaluation, and result submission.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner class that orchestrates different types of tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_dir = Path(config['experiment_dir'])
        self.data_dir = Path(config['data_dir'])
        self.code_dir = Path(config['code_dir'])
        self.logs_dir = Path(config['logs_dir'])
        self.submission_dir = Path(config['submission_dir'])
        self.experiment_name = config['experiment_name']
        self.task_type = config['task_type']
        self.time_limit = config['time_limit']
        
        # Create workspace directories
        self.workspace_dir = self.experiment_dir / 'workspace'
        self.models_dir = self.experiment_dir / 'models'
        self.results_dir = self.experiment_dir / 'results'
        self.notebooks_dir = self.experiment_dir / 'notebooks'
        self.scripts_dir = self.experiment_dir / 'scripts'
        self.configs_dir = self.experiment_dir / 'configs'
        
        # Ensure directories exist
        for dir_path in [self.workspace_dir, self.models_dir, self.results_dir, 
                        self.notebooks_dir, self.scripts_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load task configuration if available
        self.task_config = self._load_task_config()
        
        logger.info(f"Initialized ExperimentRunner for task type: {self.task_type}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
        logger.info(f"Time limit: {self.time_limit} seconds")
    
    def _load_task_config(self) -> Dict[str, Any]:
        """Load task-specific configuration."""
        task_config_path = self.experiment_dir / 'task_config.json'
        if task_config_path.exists():
            with open(task_config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_experiment_metadata(self):
        """Save experiment metadata and configuration."""
        metadata = {
            'experiment_name': self.experiment_name,
            'task_type': self.task_type,
            'start_time': datetime.now().isoformat(),
            'time_limit': self.time_limit,
            'config': self.config,
            'task_config': self.task_config,
            'directories': {
                'experiment_dir': str(self.experiment_dir),
                'data_dir': str(self.data_dir),
                'code_dir': str(self.code_dir),
                'logs_dir': str(self.logs_dir),
                'submission_dir': str(self.submission_dir)
            }
        }
        
        metadata_path = self.results_dir / 'experiment_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved experiment metadata to {metadata_path}")
    
    def run_data_analysis_task(self):
        """Run data analysis and exploration task."""
        logger.info("Starting data analysis task...")
        
        # Create analysis notebook template
        notebook_content = self._create_data_analysis_notebook()
        notebook_path = self.notebooks_dir / 'data_analysis.ipynb'
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        # Create analysis script
        script_content = self._create_data_analysis_script()
        script_path = self.scripts_dir / 'analyze_data.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created data analysis templates at {notebook_path} and {script_path}")
        
        # Execute basic data exploration
        try:
            self._execute_basic_data_exploration()
        except Exception as e:
            logger.error(f"Error in data exploration: {e}")
            traceback.print_exc()
    
    def run_machine_learning_task(self):
        """Run machine learning model development task."""
        logger.info("Starting machine learning task...")
        
        # Create ML pipeline template
        pipeline_content = self._create_ml_pipeline_script()
        pipeline_path = self.scripts_dir / 'ml_pipeline.py'
        with open(pipeline_path, 'w') as f:
            f.write(pipeline_content)
        
        # Create model training notebook
        notebook_content = self._create_ml_notebook()
        notebook_path = self.notebooks_dir / 'model_training.ipynb'
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        logger.info(f"Created ML templates at {pipeline_path} and {notebook_path}")
        
        # Execute basic ML pipeline
        try:
            self._execute_basic_ml_pipeline()
        except Exception as e:
            logger.error(f"Error in ML pipeline: {e}")
            traceback.print_exc()
    
    def run_deep_learning_task(self):
        """Run deep learning model development task."""
        logger.info("Starting deep learning task...")
        
        # Create DL training script
        script_content = self._create_dl_training_script()
        script_path = self.scripts_dir / 'train_model.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Create experiment tracking notebook
        notebook_content = self._create_dl_notebook()
        notebook_path = self.notebooks_dir / 'deep_learning_experiments.ipynb'
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        logger.info(f"Created DL templates at {script_path} and {notebook_path}")
    
    def run_general_task(self):
        """Run general-purpose experimentation task."""
        logger.info("Starting general experimentation task...")
        
        # Create general experiment template
        script_content = self._create_general_experiment_script()
        script_path = self.scripts_dir / 'experiment.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Create exploration notebook
        notebook_content = self._create_general_notebook()
        notebook_path = self.notebooks_dir / 'exploration.ipynb'
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        logger.info(f"Created general templates at {script_path} and {notebook_path}")
        
        # Execute basic exploration
        try:
            self._execute_general_exploration()
        except Exception as e:
            logger.error(f"Error in general exploration: {e}")
            traceback.print_exc()
    
    def _execute_basic_data_exploration(self):
        """Execute basic data exploration."""
        logger.info("Executing basic data exploration...")
        
        # Look for data files
        data_files = []
        for ext in ['*.csv', '*.json', '*.parquet', '*.xlsx', '*.txt']:
            data_files.extend(list(self.data_dir.glob(ext)))
        
        if not data_files:
            logger.warning("No data files found for exploration")
            return
        
        exploration_results = {}
        
        for data_file in data_files[:5]:  # Limit to first 5 files
            try:
                logger.info(f"Exploring {data_file.name}...")
                
                if data_file.suffix == '.csv':
                    df = pd.read_csv(data_file)
                    exploration_results[data_file.name] = {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.to_dict(),
                        'missing_values': df.isnull().sum().to_dict(),
                        'sample_data': df.head().to_dict()
                    }
                elif data_file.suffix == '.json':
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    exploration_results[data_file.name] = {
                        'type': type(data).__name__,
                        'size': len(data) if hasattr(data, '__len__') else 'unknown',
                        'sample': str(data)[:500] if isinstance(data, (dict, list)) else str(data)[:500]
                    }
                
            except Exception as e:
                logger.error(f"Error exploring {data_file.name}: {e}")
                exploration_results[data_file.name] = {'error': str(e)}
        
        # Save exploration results
        results_path = self.results_dir / 'data_exploration_results.json'
        with open(results_path, 'w') as f:
            json.dump(exploration_results, f, indent=2, default=str)
        
        logger.info(f"Saved data exploration results to {results_path}")
    
    def _execute_basic_ml_pipeline(self):
        """Execute a basic ML pipeline."""
        logger.info("Executing basic ML pipeline...")
        
        # This is a placeholder - in practice, agents would implement their own logic
        pipeline_results = {
            'status': 'template_created',
            'message': 'ML pipeline template created. Agents should implement specific logic.',
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.results_dir / 'ml_pipeline_results.json'
        with open(results_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        logger.info(f"Saved ML pipeline results to {results_path}")
    
    def _execute_general_exploration(self):
        """Execute general exploration."""
        logger.info("Executing general exploration...")
        
        # Create a summary of the environment
        environment_info = {
            'python_version': sys.version,
            'available_packages': [],
            'data_directory_contents': [str(p) for p in self.data_dir.iterdir()] if self.data_dir.exists() else [],
            'experiment_config': self.config,
            'task_config': self.task_config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to get installed packages
        try:
            import pkg_resources
            environment_info['available_packages'] = [str(d) for d in pkg_resources.working_set]
        except Exception as e:
            logger.warning(f"Could not get package list: {e}")
        
        results_path = self.results_dir / 'environment_info.json'
        with open(results_path, 'w') as f:
            json.dump(environment_info, f, indent=2)
        
        logger.info(f"Saved environment info to {results_path}")
    
    def _create_data_analysis_notebook(self) -> Dict[str, Any]:
        """Create a Jupyter notebook template for data analysis."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Data Analysis Notebook\n", "\n", "This notebook provides a template for data analysis tasks."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "from pathlib import Path\n",
                        "\n",
                        "# Set up plotting\n",
                        "plt.style.use('default')\n",
                        "sns.set_palette('husl')\n",
                        "\n",
                        "# Define paths\n",
                        "data_dir = Path('/home/data')\n",
                        "results_dir = Path('/home/experiments/results')\n",
                        "\n",
                        "print('Data analysis environment ready!')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Load and Explore Data"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Load your data here\n",
                        "# Example:\n",
                        "# df = pd.read_csv(data_dir / 'your_data.csv')\n",
                        "# print(f'Data shape: {df.shape}')\n",
                        "# df.head()"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def _create_data_analysis_script(self) -> str:
        """Create a Python script template for data analysis."""
        return '''#!/usr/bin/env python3
"""
Data Analysis Script Template

This script provides a template for automated data analysis tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_data(data_dir: Path, output_dir: Path):
    """
    Perform data analysis on files in the data directory.
    
    Args:
        data_dir: Directory containing data files
        output_dir: Directory to save analysis results
    """
    logger.info(f"Starting data analysis on {data_dir}")
    
    # Find data files
    data_files = list(data_dir.glob('*.csv'))
    
    if not data_files:
        logger.warning("No CSV files found for analysis")
        return
    
    results = {}
    
    for data_file in data_files:
        logger.info(f"Analyzing {data_file.name}")
        
        try:
            df = pd.read_csv(data_file)
            
            # Basic statistics
            analysis = {
                'file_name': data_file.name,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
            
            results[data_file.name] = analysis
            
            # Create visualizations if numeric columns exist
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'Analysis of {data_file.name}')
                
                # Distribution plot
                if len(numeric_cols) > 0:
                    df[numeric_cols[0]].hist(ax=axes[0, 0])
                    axes[0, 0].set_title(f'Distribution of {numeric_cols[0]}')
                
                # Correlation heatmap
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    sns.heatmap(corr, annot=True, ax=axes[0, 1])
                    axes[0, 1].set_title('Correlation Matrix')
                
                # Missing values plot
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    missing_data[missing_data > 0].plot(kind='bar', ax=axes[1, 0])
                    axes[1, 0].set_title('Missing Values')
                
                # Data types plot
                dtype_counts = df.dtypes.value_counts()
                dtype_counts.plot(kind='pie', ax=axes[1, 1])
                axes[1, 1].set_title('Data Types Distribution')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'{data_file.stem}_analysis.png')
                plt.close()
                
        except Exception as e:
            logger.error(f"Error analyzing {data_file.name}: {e}")
            results[data_file.name] = {'error': str(e)}
    
    # Save results
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    data_dir = Path('/home/data')
    output_dir = Path('/home/experiments/results')
    
    analyze_data(data_dir, output_dir)
'''
    
    def _create_ml_pipeline_script(self) -> str:
        """Create a machine learning pipeline script template."""
        return '''#!/usr/bin/env python3
"""
Machine Learning Pipeline Template

This script provides a template for ML model development and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from pathlib import Path
import joblib
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPipeline:
    """A flexible ML pipeline for classification and regression tasks."""
    
    def __init__(self, task_type='auto'):
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_name = None
        
    def load_data(self, data_path: Path):
        """Load data from CSV file."""
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        logger.info(f"Data shape: {self.data.shape}")
        return self.data
    
    def prepare_data(self, target_column: str, test_size=0.2, random_state=42):
        """Prepare data for training."""
        logger.info(f"Preparing data with target column: {target_column}")
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        self.feature_names = list(X.columns)
        self.target_name = target_column
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Determine task type if auto
        if self.task_type == 'auto':
            if y.dtype == 'object' or len(y.unique()) < 10:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
        
        logger.info(f"Task type: {self.task_type}")
        
        # Encode target for classification
        if self.task_type == 'classification' and y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Test set shape: {self.X_test.shape}")
    
    def train_model(self, model_type='auto'):
        """Train the model."""
        logger.info(f"Training {model_type} model for {self.task_type}")
        
        if model_type == 'auto':
            if self.task_type == 'classification':
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            if self.task_type == 'classification':
                self.model = LogisticRegression(random_state=42)
            else:
                self.model = LinearRegression()
        
        # Train model
        if model_type == 'linear':
            self.model.fit(self.X_train_scaled, self.y_train)
        else:
            self.model.fit(self.X_train, self.y_train)
        
        logger.info("Model training completed")
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        logger.info("Evaluating model performance")
        
        # Make predictions
        if hasattr(self.model, 'predict'):
            if isinstance(self.model, (LogisticRegression, LinearRegression)):
                y_pred = self.model.predict(self.X_test_scaled)
            else:
                y_pred = self.model.predict(self.X_test)
        
        results = {}
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(self.y_test, y_pred)
            results['accuracy'] = accuracy
            results['classification_report'] = classification_report(self.y_test, y_pred, output_dict=True)
            logger.info(f"Accuracy: {accuracy:.4f}")
        else:
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            results['mse'] = mse
            results['r2'] = r2
            logger.info(f"MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return results
    
    def save_model(self, model_path: Path):
        """Save the trained model."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'task_type': self.task_type
        }, model_path)
        logger.info(f"Model saved to {model_path}")

def run_ml_pipeline(data_dir: Path, output_dir: Path):
    """Run the complete ML pipeline."""
    logger.info("Starting ML pipeline")
    
    # Find CSV files
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        logger.warning("No CSV files found")
        return
    
    # Use the first CSV file
    data_file = csv_files[0]
    logger.info(f"Using data file: {data_file}")
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    try:
        # Load data
        data = pipeline.load_data(data_file)
        
        # Assume the last column is the target (this is a simple heuristic)
        target_column = data.columns[-1]
        logger.info(f"Using '{target_column}' as target column")
        
        # Prepare data
        pipeline.prepare_data(target_column)
        
        # Train model
        pipeline.train_model()
        
        # Evaluate model
        results = pipeline.evaluate_model()
        
        # Save results
        with open(output_dir / 'ml_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model
        pipeline.save_model(output_dir / 'trained_model.joblib')
        
        logger.info("ML pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {e}")
        with open(output_dir / 'ml_error.json', 'w') as f:
            json.dump({'error': str(e)}, f, indent=2)

if __name__ == "__main__":
    data_dir = Path('/home/data')
    output_dir = Path('/home/experiments/results')
    
    run_ml_pipeline(data_dir, output_dir)
'''
    
    def _create_ml_notebook(self) -> Dict[str, Any]:
        """Create a machine learning notebook template."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Machine Learning Experiment Notebook"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "from sklearn.model_selection import train_test_split\n",
                        "from sklearn.preprocessing import StandardScaler\n",
                        "from sklearn.ensemble import RandomForestClassifier\n",
                        "from sklearn.metrics import accuracy_score, classification_report\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "\n",
                        "print('ML environment ready!')"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def _create_dl_training_script(self) -> str:
        """Create a deep learning training script template."""
        return '''#!/usr/bin/env python3
"""
Deep Learning Training Script Template

This script provides a template for deep learning model training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """Simple dataset class for demonstration."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNN(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """Train the model."""
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return train_losses

def run_dl_training(data_dir: Path, output_dir: Path):
    """Run deep learning training pipeline."""
    logger.info("Starting deep learning training")
    
    # This is a template - agents should implement their specific logic
    results = {
        'status': 'template_created',
        'message': 'Deep learning template created. Implement specific training logic.',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    with open(output_dir / 'dl_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Deep learning template setup completed")

if __name__ == "__main__":
    data_dir = Path('/home/data')
    output_dir = Path('/home/experiments/results')
    
    run_dl_training(data_dir, output_dir)
'''
    
    def _create_dl_notebook(self) -> Dict[str, Any]:
        """Create a deep learning notebook template."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Deep Learning Experiment Notebook"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import torch\n",
                        "import torch.nn as nn\n",
                        "import torch.optim as optim\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "\n",
                        "print(f'PyTorch version: {torch.__version__}')\n",
                        "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def _create_general_experiment_script(self) -> str:
        """Create a general experiment script template."""
        return '''#!/usr/bin/env python3
"""
General Experiment Script Template

This script provides a flexible template for various types of experiments.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_experiment(experiment_config: dict):
    """
    Run a general experiment based on the provided configuration.
    
    Args:
        experiment_config: Dictionary containing experiment parameters
    """
    logger.info("Starting general experiment")
    logger.info(f"Experiment config: {experiment_config}")
    
    # Get environment information
    env_info = {
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'environment_variables': dict(os.environ),
        'timestamp': datetime.now().isoformat()
    }
    
    # Experiment results
    results = {
        'experiment_type': 'general',
        'status': 'completed',
        'environment_info': env_info,
        'config': experiment_config,
        'message': 'General experiment template executed successfully'
    }
    
    return results

def main():
    """Main experiment function."""
    # Load experiment configuration
    config_path = Path('/home/experiments/task_config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {'type': 'general', 'parameters': {}}
    
    # Run experiment
    results = run_experiment(config)
    
    # Save results
    output_dir = Path('/home/experiments/results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
'''
    
    def _create_general_notebook(self) -> Dict[str, Any]:
        """Create a general exploration notebook template."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# General Exploration Notebook\n", "\n", "This notebook provides a template for general experimentation and exploration."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import os\n",
                        "import sys\n",
                        "import json\n",
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "from pathlib import Path\n",
                        "\n",
                        "print('General exploration environment ready!')\n",
                        "print(f'Python version: {sys.version}')\n",
                        "print(f'Working directory: {os.getcwd()}')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Explore Available Data"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Explore data directory\n",
                        "data_dir = Path('/home/data')\n",
                        "if data_dir.exists():\n",
                        "    print('Data directory contents:')\n",
                        "    for item in data_dir.iterdir():\n",
                        "        print(f'  {item.name} ({\"directory\" if item.is_dir() else \"file\"})')\n",
                        "else:\n",
                        "    print('No data directory found')"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    def run(self):
        """Run the experiment based on task type."""
        start_time = time.time()
        
        try:
            # Save experiment metadata
            self._save_experiment_metadata()
            
            # Route to appropriate task handler
            if self.task_type == 'data_analysis':
                self.run_data_analysis_task()
            elif self.task_type == 'machine_learning':
                self.run_machine_learning_task()
            elif self.task_type == 'deep_learning':
                self.run_deep_learning_task()
            else:
                self.run_general_task()
            
            # Save completion status
            completion_info = {
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'duration_seconds': time.time() - start_time,
                'task_type': self.task_type
            }
            
            with open(self.results_dir / 'completion_status.json', 'w') as f:
                json.dump(completion_info, f, indent=2)
            
            logger.info(f"Experiment completed successfully in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            traceback.print_exc()
            
            # Save error status
            error_info = {
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': time.time() - start_time,
                'task_type': self.task_type
            }
            
            with open(self.results_dir / 'error_status.json', 'w') as f:
                json.dump(error_info, f, indent=2)


def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(description='AI Agent Experiment Runner')
    parser.add_argument('--experiment-dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--code-dir', type=str, required=True, help='Code directory')
    parser.add_argument('--logs-dir', type=str, required=True, help='Logs directory')
    parser.add_argument('--submission-dir', type=str, required=True, help='Submission directory')
    parser.add_argument('--experiment-name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--task-type', type=str, default='general', help='Task type')
    parser.add_argument('--time-limit', type=int, default=3600, help='Time limit in seconds')
    
    args = parser.parse_args()
    
    config = {
        'experiment_dir': args.experiment_dir,
        'data_dir': args.data_dir,
        'code_dir': args.code_dir,
        'logs_dir': args.logs_dir,
        'submission_dir': args.submission_dir,
        'experiment_name': args.experiment_name,
        'task_type': args.task_type,
        'time_limit': args.time_limit
    }
    
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()