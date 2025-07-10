"""
Data Generator for creating synthetic datasets for AI agent tasks.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler


class DataGenerator:
    """Generates synthetic datasets for various machine learning tasks."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_classification_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        n_informative: Optional[int] = None,
        n_redundant: Optional[int] = None,
        class_sep: float = 1.0,
        flip_y: float = 0.01
    ) -> pd.DataFrame:
        """Generate a classification dataset."""
        
        if n_informative is None:
            n_informative = max(2, n_features // 2)
        if n_redundant is None:
            n_redundant = max(0, n_features - n_informative - 2)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=self.random_state
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def generate_regression_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_informative: Optional[int] = None,
        noise: float = 0.1,
        bias: float = 0.0
    ) -> pd.DataFrame:
        """Generate a regression dataset."""
        
        if n_informative is None:
            n_informative = max(2, n_features // 2)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            bias=bias,
            random_state=self.random_state
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def generate_time_series_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 5,
        trend: bool = True,
        seasonality: bool = True,
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """Generate a time series dataset."""
        
        # Create time index
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        # Generate base time series
        t = np.arange(n_samples)
        
        data = {}
        data['date'] = dates
        
        for i in range(n_features):
            # Base signal
            signal = np.zeros(n_samples)
            
            # Add trend
            if trend:
                trend_coef = np.random.uniform(-0.01, 0.01)
                signal += trend_coef * t
            
            # Add seasonality
            if seasonality:
                seasonal_period = np.random.randint(7, 365)  # Weekly to yearly
                seasonal_amplitude = np.random.uniform(0.5, 2.0)
                signal += seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
            
            # Add noise
            noise = np.random.normal(0, noise_level, n_samples)
            signal += noise
            
            # Add random walk component
            random_walk = np.cumsum(np.random.normal(0, 0.1, n_samples))
            signal += random_walk
            
            data[f'feature_{i}'] = signal
        
        # Create target as a combination of features
        target = np.zeros(n_samples)
        for i in range(min(3, n_features)):  # Use up to 3 features for target
            weight = np.random.uniform(0.1, 1.0)
            target += weight * data[f'feature_{i}']
        
        # Add some lag effect
        if n_samples > 10:
            lag_effect = np.roll(target, 5) * 0.3
            target += lag_effect
        
        data['target'] = target
        
        return pd.DataFrame(data)
    
    def generate_clustering_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 2,
        n_centers: int = 3,
        cluster_std: float = 1.0
    ) -> pd.DataFrame:
        """Generate a clustering dataset."""
        
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=self.random_state
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = y
        
        return df
    
    def generate_text_classification_dataset(
        self,
        n_samples: int = 1000,
        n_classes: int = 3,
        vocab_size: int = 1000,
        avg_doc_length: int = 50
    ) -> pd.DataFrame:
        """Generate a synthetic text classification dataset."""
        
        # Create vocabulary
        vocab = [f'word_{i}' for i in range(vocab_size)]
        
        # Define class-specific word preferences
        class_words = {}
        words_per_class = vocab_size // n_classes
        
        for class_id in range(n_classes):
            start_idx = class_id * words_per_class
            end_idx = start_idx + words_per_class
            class_words[class_id] = vocab[start_idx:end_idx]
        
        documents = []
        labels = []
        
        for _ in range(n_samples):
            # Choose class
            class_id = np.random.randint(0, n_classes)
            labels.append(class_id)
            
            # Generate document
            doc_length = max(10, int(np.random.normal(avg_doc_length, 10)))
            
            # 70% class-specific words, 30% random words
            n_class_words = int(0.7 * doc_length)
            n_random_words = doc_length - n_class_words
            
            class_specific = np.random.choice(class_words[class_id], n_class_words)
            random_words = np.random.choice(vocab, n_random_words)
            
            doc_words = np.concatenate([class_specific, random_words])
            np.random.shuffle(doc_words)
            
            document = ' '.join(doc_words)
            documents.append(document)
        
        return pd.DataFrame({
            'text': documents,
            'label': labels
        })
    
    def generate_tabular_dataset_with_missing_values(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        missing_rate: float = 0.1,
        task_type: str = 'classification'
    ) -> pd.DataFrame:
        """Generate a tabular dataset with missing values."""
        
        if task_type == 'classification':
            df = self.generate_classification_dataset(n_samples, n_features)
        else:
            df = self.generate_regression_dataset(n_samples, n_features)
        
        # Introduce missing values
        feature_cols = [col for col in df.columns if col != 'target']
        
        for col in feature_cols:
            # Randomly select indices to make missing
            n_missing = int(missing_rate * n_samples)
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
            df.loc[missing_indices, col] = np.nan
        
        return df
    
    def generate_imbalanced_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        imbalance_ratio: float = 0.1  # ratio of minority class
    ) -> pd.DataFrame:
        """Generate an imbalanced classification dataset."""
        
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority
        
        # Generate majority class
        X_maj, _ = make_classification(
            n_samples=n_majority,
            n_features=n_features,
            n_classes=1,
            n_clusters_per_class=1,
            random_state=self.random_state
        )
        y_maj = np.zeros(n_majority)
        
        # Generate minority class (shifted distribution)
        X_min, _ = make_classification(
            n_samples=n_minority,
            n_features=n_features,
            n_classes=1,
            n_clusters_per_class=1,
            random_state=self.random_state + 1
        )
        # Shift minority class
        X_min += 2
        y_min = np.ones(n_minority)
        
        # Combine
        X = np.vstack([X_maj, X_min])
        y = np.hstack([y_maj, y_min])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y.astype(int)
        
        return df
    
    def generate_multi_modal_dataset(
        self,
        n_samples: int = 1000,
        image_size: Tuple[int, int] = (28, 28),
        n_text_features: int = 100,
        n_classes: int = 3
    ) -> Dict[str, Any]:
        """Generate a multi-modal dataset with images and text."""
        
        # Generate synthetic images (random noise for simplicity)
        images = np.random.rand(n_samples, *image_size, 1)  # Grayscale
        
        # Generate text features (bag of words representation)
        text_features = np.random.rand(n_samples, n_text_features)
        
        # Generate labels
        labels = np.random.randint(0, n_classes, n_samples)
        
        return {
            'images': images,
            'text_features': text_features,
            'labels': labels,
            'metadata': {
                'n_samples': n_samples,
                'image_shape': image_size,
                'n_text_features': n_text_features,
                'n_classes': n_classes
            }
        }