"""
Task Validator for evaluating AI agent task completion.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

from ..task_registry import Task


logger = logging.getLogger(__name__)


class TaskValidator:
    """Validates and evaluates task completion by AI agents."""
    
    def __init__(self):
        self.evaluation_functions = {
            'classification': self._evaluate_classification,
            'regression': self._evaluate_regression,
            'clustering': self._evaluate_clustering,
            'data_analysis': self._evaluate_data_analysis,
            'general': self._evaluate_general
        }
    
    def evaluate_task(self, task: Task, results_dir: Path) -> Dict[str, Any]:
        """Evaluate a completed task."""
        logger.info(f"Evaluating task: {task.name}")
        
        evaluation = {
            'task_id': task.id,
            'task_name': task.name,
            'task_type': task.task_type,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'criteria_met': {},
            'scores': {},
            'feedback': [],
            'overall_score': 0.0,
            'success': False
        }
        
        try:
            # Check if expected outputs exist
            output_check = self._check_expected_outputs(task, results_dir)
            evaluation['output_check'] = output_check
            
            # Run task-specific evaluation
            if task.task_type in self.evaluation_functions:
                specific_eval = self.evaluation_functions[task.task_type](task, results_dir)
                evaluation.update(specific_eval)
            else:
                evaluation['feedback'].append(f"No specific evaluator for task type: {task.task_type}")
            
            # Calculate overall score
            evaluation['overall_score'] = self._calculate_overall_score(evaluation)
            evaluation['success'] = evaluation['overall_score'] >= 0.6  # 60% threshold
            
        except Exception as e:
            logger.error(f"Error evaluating task {task.id}: {e}")
            evaluation['error'] = str(e)
            evaluation['success'] = False
        
        return evaluation
    
    def _check_expected_outputs(self, task: Task, results_dir: Path) -> Dict[str, Any]:
        """Check if expected output files exist."""
        output_check = {
            'files_found': [],
            'files_missing': [],
            'score': 0.0
        }
        
        expected_outputs = task.expected_outputs
        
        for expected_file in expected_outputs:
            file_path = results_dir / expected_file
            if file_path.exists():
                output_check['files_found'].append(expected_file)
            else:
                output_check['files_missing'].append(expected_file)
        
        if expected_outputs:
            output_check['score'] = len(output_check['files_found']) / len(expected_outputs)
        else:
            output_check['score'] = 1.0  # No specific outputs required
        
        return output_check
    
    def _evaluate_classification(self, task: Task, results_dir: Path) -> Dict[str, Any]:
        """Evaluate a classification task."""
        evaluation = {
            'criteria_met': {},
            'scores': {},
            'feedback': []
        }
        
        # Look for predictions file
        predictions_file = results_dir / 'predictions.csv'
        if not predictions_file.exists():
            evaluation['feedback'].append("No predictions.csv file found")
            return evaluation
        
        try:
            predictions_df = pd.read_csv(predictions_file)
            
            # Check if required columns exist
            required_cols = ['true_labels', 'predicted_labels']
            missing_cols = [col for col in required_cols if col not in predictions_df.columns]
            
            if missing_cols:
                evaluation['feedback'].append(f"Missing columns in predictions: {missing_cols}")
                return evaluation
            
            # Calculate metrics
            y_true = predictions_df['true_labels']
            y_pred = predictions_df['predicted_labels']
            
            accuracy = accuracy_score(y_true, y_pred)
            evaluation['scores']['accuracy'] = accuracy
            
            # Check accuracy threshold
            accuracy_threshold = task.evaluation_criteria.get('min_accuracy', 0.5)
            evaluation['criteria_met']['accuracy'] = accuracy >= accuracy_threshold
            
            if accuracy >= accuracy_threshold:
                evaluation['feedback'].append(f"Accuracy {accuracy:.3f} meets threshold {accuracy_threshold}")
            else:
                evaluation['feedback'].append(f"Accuracy {accuracy:.3f} below threshold {accuracy_threshold}")
            
            # Additional metrics
            try:
                report = classification_report(y_true, y_pred, output_dict=True)
                evaluation['scores']['classification_report'] = report
            except Exception as e:
                evaluation['feedback'].append(f"Could not generate classification report: {e}")
            
        except Exception as e:
            evaluation['feedback'].append(f"Error evaluating classification: {e}")
        
        return evaluation
    
    def _evaluate_regression(self, task: Task, results_dir: Path) -> Dict[str, Any]:
        """Evaluate a regression task."""
        evaluation = {
            'criteria_met': {},
            'scores': {},
            'feedback': []
        }
        
        # Look for predictions file
        predictions_file = results_dir / 'predictions.csv'
        if not predictions_file.exists():
            evaluation['feedback'].append("No predictions.csv file found")
            return evaluation
        
        try:
            predictions_df = pd.read_csv(predictions_file)
            
            # Check if required columns exist
            required_cols = ['true_values', 'predicted_values']
            missing_cols = [col for col in required_cols if col not in predictions_df.columns]
            
            if missing_cols:
                evaluation['feedback'].append(f"Missing columns in predictions: {missing_cols}")
                return evaluation
            
            # Calculate metrics
            y_true = predictions_df['true_values']
            y_pred = predictions_df['predicted_values']
            
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            evaluation['scores']['mse'] = mse
            evaluation['scores']['r2'] = r2
            evaluation['scores']['rmse'] = np.sqrt(mse)
            
            # Check R² threshold
            r2_threshold = task.evaluation_criteria.get('min_r2', 0.3)
            evaluation['criteria_met']['r2'] = r2 >= r2_threshold
            
            if r2 >= r2_threshold:
                evaluation['feedback'].append(f"R² {r2:.3f} meets threshold {r2_threshold}")
            else:
                evaluation['feedback'].append(f"R² {r2:.3f} below threshold {r2_threshold}")
            
        except Exception as e:
            evaluation['feedback'].append(f"Error evaluating regression: {e}")
        
        return evaluation
    
    def _evaluate_clustering(self, task: Task, results_dir: Path) -> Dict[str, Any]:
        """Evaluate a clustering task."""
        evaluation = {
            'criteria_met': {},
            'scores': {},
            'feedback': []
        }
        
        # Look for cluster assignments
        clusters_file = results_dir / 'cluster_assignments.csv'
        if not clusters_file.exists():
            evaluation['feedback'].append("No cluster_assignments.csv file found")
            return evaluation
        
        try:
            clusters_df = pd.read_csv(clusters_file)
            
            if 'cluster' not in clusters_df.columns:
                evaluation['feedback'].append("No 'cluster' column found in assignments")
                return evaluation
            
            clusters = clusters_df['cluster']
            n_clusters = len(clusters.unique())
            
            evaluation['scores']['n_clusters_found'] = n_clusters
            
            # Check if number of clusters is reasonable
            expected_clusters = task.evaluation_criteria.get('expected_clusters')
            if expected_clusters:
                cluster_diff = abs(n_clusters - expected_clusters)
                evaluation['criteria_met']['cluster_count'] = cluster_diff <= 1
                evaluation['feedback'].append(f"Found {n_clusters} clusters, expected {expected_clusters}")
            
            # Check cluster size distribution
            cluster_sizes = clusters.value_counts()
            min_cluster_size = cluster_sizes.min()
            max_cluster_size = cluster_sizes.max()
            
            evaluation['scores']['min_cluster_size'] = min_cluster_size
            evaluation['scores']['max_cluster_size'] = max_cluster_size
            evaluation['scores']['cluster_size_ratio'] = max_cluster_size / min_cluster_size if min_cluster_size > 0 else float('inf')
            
        except Exception as e:
            evaluation['feedback'].append(f"Error evaluating clustering: {e}")
        
        return evaluation
    
    def _evaluate_data_analysis(self, task: Task, results_dir: Path) -> Dict[str, Any]:
        """Evaluate a data analysis task."""
        evaluation = {
            'criteria_met': {},
            'scores': {},
            'feedback': []
        }
        
        # Check for analysis report
        report_files = ['analysis_report.txt', 'analysis_report.md', 'analysis_results.json']
        report_found = False
        
        for report_file in report_files:
            if (results_dir / report_file).exists():
                report_found = True
                evaluation['feedback'].append(f"Found analysis report: {report_file}")
                break
        
        evaluation['criteria_met']['report_exists'] = report_found
        
        if not report_found:
            evaluation['feedback'].append("No analysis report found")
        
        # Check for visualizations
        viz_files = list(results_dir.glob('*.png')) + list(results_dir.glob('*.jpg')) + list(results_dir.glob('*.svg'))
        evaluation['scores']['n_visualizations'] = len(viz_files)
        evaluation['criteria_met']['has_visualizations'] = len(viz_files) > 0
        
        if viz_files:
            evaluation['feedback'].append(f"Found {len(viz_files)} visualization files")
        
        # Check for summary statistics
        stats_file = results_dir / 'summary_statistics.json'
        if stats_file.exists():
            evaluation['criteria_met']['has_statistics'] = True
            evaluation['feedback'].append("Found summary statistics")
        else:
            evaluation['criteria_met']['has_statistics'] = False
        
        return evaluation
    
    def _evaluate_general(self, task: Task, results_dir: Path) -> Dict[str, Any]:
        """Evaluate a general task."""
        evaluation = {
            'criteria_met': {},
            'scores': {},
            'feedback': []
        }
        
        # Count output files
        output_files = list(results_dir.glob('*'))
        evaluation['scores']['n_output_files'] = len(output_files)
        
        # Check for code files
        code_files = list(results_dir.glob('*.py')) + list(results_dir.glob('*.ipynb'))
        evaluation['scores']['n_code_files'] = len(code_files)
        evaluation['criteria_met']['has_code'] = len(code_files) > 0
        
        # Check for documentation
        doc_files = list(results_dir.glob('*.md')) + list(results_dir.glob('*.txt'))
        evaluation['scores']['n_doc_files'] = len(doc_files)
        evaluation['criteria_met']['has_documentation'] = len(doc_files) > 0
        
        evaluation['feedback'].append(f"Generated {len(output_files)} output files")
        
        return evaluation
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall score based on criteria and scores."""
        score = 0.0
        total_weight = 0.0
        
        # Output check score (weight: 0.3)
        if 'output_check' in evaluation:
            score += 0.3 * evaluation['output_check']['score']
            total_weight += 0.3
        
        # Criteria met score (weight: 0.4)
        criteria_met = evaluation.get('criteria_met', {})
        if criteria_met:
            criteria_score = sum(criteria_met.values()) / len(criteria_met)
            score += 0.4 * criteria_score
            total_weight += 0.4
        
        # Specific scores (weight: 0.3)
        scores = evaluation.get('scores', {})
        if scores:
            # Normalize specific scores based on task type
            specific_score = self._normalize_specific_scores(scores, evaluation.get('task_type'))
            score += 0.3 * specific_score
            total_weight += 0.3
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _normalize_specific_scores(self, scores: Dict[str, Any], task_type: str) -> float:
        """Normalize task-specific scores to 0-1 range."""
        if task_type == 'classification':
            return scores.get('accuracy', 0.0)
        elif task_type == 'regression':
            r2 = scores.get('r2', 0.0)
            return max(0.0, r2)  # R² can be negative
        elif task_type == 'clustering':
            # Simple heuristic based on cluster count reasonableness
            n_clusters = scores.get('n_clusters_found', 0)
            if 2 <= n_clusters <= 10:
                return 1.0
            else:
                return 0.5
        else:
            # For general tasks, use file count as a proxy
            n_files = scores.get('n_output_files', 0)
            return min(1.0, n_files / 5.0)  # Normalize to max 5 files