#!/usr/bin/env python3
"""
Task Management CLI

Command-line interface for managing AI agent tasks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .task_registry import Task, registry
from .task_runner import TaskRunner
from .utils.data_generator import DataGenerator


def list_tasks(task_type: Optional[str] = None, difficulty: Optional[str] = None):
    """List available tasks."""
    tasks = registry.list_tasks(task_type=task_type, difficulty=difficulty)
    
    if not tasks:
        print("No tasks found matching criteria.")
        return
    
    print(f"Found {len(tasks)} tasks:")
    print("-" * 80)
    
    for task in tasks:
        print(f"ID: {task.id}")
        print(f"Name: {task.name}")
        print(f"Type: {task.task_type}")
        print(f"Difficulty: {task.difficulty}")
        print(f"Estimated Time: {task.estimated_time}s ({task.estimated_time//60}m)")
        print(f"Description: {task.description}")
        print("-" * 80)


def show_task(task_id: str):
    """Show detailed information about a task."""
    task = registry.get_task(task_id)
    
    if not task:
        print(f"Task '{task_id}' not found.")
        return
    
    print(f"Task: {task.name}")
    print(f"ID: {task.id}")
    print(f"Type: {task.task_type}")
    print(f"Difficulty: {task.difficulty}")
    print(f"Estimated Time: {task.estimated_time}s ({task.estimated_time//60}m)")
    print(f"Created: {task.created_at}")
    print()
    print("Description:")
    print(task.description)
    print()
    print("Instructions:")
    print(task.instructions)
    print()
    print("Data Requirements:")
    print(json.dumps(task.data_requirements, indent=2))
    print()
    print("Evaluation Criteria:")
    print(json.dumps(task.evaluation_criteria, indent=2))
    print()
    print("Expected Outputs:")
    for output in task.expected_outputs:
        print(f"  - {output}")
    print()
    print("Metadata:")
    print(json.dumps(task.metadata, indent=2))


def create_task_config(task_id: str, output_path: str, **kwargs):
    """Create a task configuration file."""
    output_file = Path(output_path)
    
    try:
        config = registry.create_task_config(task_id, output_file, **kwargs)
        print(f"Task configuration created: {output_file}")
        print("Configuration:")
        print(json.dumps(config, indent=2))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def generate_sample_data(task_id: str, output_dir: str):
    """Generate sample data for a task."""
    task = registry.get_task(task_id)
    
    if not task:
        print(f"Task '{task_id}' not found.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_requirements = task.data_requirements
    
    if not data_requirements.get('generate_synthetic', False):
        print("Task does not support synthetic data generation.")
        return
    
    generator = DataGenerator()
    
    try:
        if data_requirements.get('dataset_type') == 'classification':
            df = generator.generate_classification_dataset(
                n_samples=data_requirements.get('n_samples', 1000),
                n_features=data_requirements.get('n_features', 10),
                n_classes=data_requirements.get('n_classes', 2)
            )
            output_file = output_path / 'classification_data.csv'
            df.to_csv(output_file, index=False)
            print(f"Generated classification dataset: {output_file}")
        
        elif data_requirements.get('dataset_type') == 'regression':
            df = generator.generate_regression_dataset(
                n_samples=data_requirements.get('n_samples', 1000),
                n_features=data_requirements.get('n_features', 10)
            )
            output_file = output_path / 'regression_data.csv'
            df.to_csv(output_file, index=False)
            print(f"Generated regression dataset: {output_file}")
        
        elif data_requirements.get('dataset_type') == 'time_series':
            df = generator.generate_time_series_dataset(
                n_samples=data_requirements.get('n_samples', 1000),
                n_features=data_requirements.get('n_features', 5)
            )
            output_file = output_path / 'time_series_data.csv'
            df.to_csv(output_file, index=False)
            print(f"Generated time series dataset: {output_file}")
        
        else:
            print(f"Unsupported dataset type: {data_requirements.get('dataset_type')}")
    
    except Exception as e:
        print(f"Error generating data: {e}")


def validate_task_results(task_id: str, results_dir: str):
    """Validate task results."""
    task = registry.get_task(task_id)
    
    if not task:
        print(f"Task '{task_id}' not found.")
        return
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return
    
    from .utils.task_validator import TaskValidator
    
    validator = TaskValidator()
    evaluation = validator.evaluate_task(task, results_path)
    
    print(f"Task Evaluation Results for: {task.name}")
    print("=" * 60)
    print(f"Overall Score: {evaluation['overall_score']:.2f}")
    print(f"Success: {'✓' if evaluation['success'] else '✗'}")
    print()
    
    if 'output_check' in evaluation:
        output_check = evaluation['output_check']
        print(f"Output Files Score: {output_check['score']:.2f}")
        if output_check['files_found']:
            print("Files Found:")
            for file in output_check['files_found']:
                print(f"  ✓ {file}")
        if output_check['files_missing']:
            print("Files Missing:")
            for file in output_check['files_missing']:
                print(f"  ✗ {file}")
        print()
    
    if evaluation['criteria_met']:
        print("Criteria Met:")
        for criterion, met in evaluation['criteria_met'].items():
            status = "✓" if met else "✗"
            print(f"  {status} {criterion}")
        print()
    
    if evaluation['scores']:
        print("Scores:")
        for metric, score in evaluation['scores'].items():
            if isinstance(score, (int, float)):
                print(f"  {metric}: {score:.3f}")
            else:
                print(f"  {metric}: {score}")
        print()
    
    if evaluation['feedback']:
        print("Feedback:")
        for feedback in evaluation['feedback']:
            print(f"  • {feedback}")


def create_custom_task():
    """Interactive task creation."""
    print("Creating a new custom task...")
    print("=" * 40)
    
    task_id = input("Task ID: ").strip()
    name = input("Task Name: ").strip()
    description = input("Description: ").strip()
    task_type = input("Task Type (classification/regression/data_analysis/general): ").strip()
    difficulty = input("Difficulty (easy/medium/hard): ").strip()
    estimated_time = int(input("Estimated Time (seconds): ").strip())
    
    print("\nData Requirements:")
    generate_synthetic = input("Generate synthetic data? (y/n): ").strip().lower() == 'y'
    
    data_requirements = {"generate_synthetic": generate_synthetic}
    
    if generate_synthetic:
        dataset_type = input("Dataset type: ").strip()
        n_samples = int(input("Number of samples: ").strip())
        n_features = int(input("Number of features: ").strip())
        
        data_requirements.update({
            "dataset_type": dataset_type,
            "n_samples": n_samples,
            "n_features": n_features
        })
        
        if dataset_type == "classification":
            n_classes = int(input("Number of classes: ").strip())
            data_requirements["n_classes"] = n_classes
    
    print("\nEvaluation Criteria:")
    evaluation_criteria = {}
    
    if task_type == "classification":
        min_accuracy = float(input("Minimum accuracy (0-1): ").strip())
        evaluation_criteria["min_accuracy"] = min_accuracy
    elif task_type == "regression":
        min_r2 = float(input("Minimum R² score: ").strip())
        evaluation_criteria["min_r2"] = min_r2
    
    instructions = input("\nTask Instructions: ").strip()
    
    print("\nExpected Outputs (comma-separated):")
    expected_outputs = [x.strip() for x in input().split(',') if x.strip()]
    
    # Create task
    task = Task(
        id=task_id,
        name=name,
        description=description,
        task_type=task_type,
        difficulty=difficulty,
        estimated_time=estimated_time,
        data_requirements=data_requirements,
        evaluation_criteria=evaluation_criteria,
        instructions=instructions,
        expected_outputs=expected_outputs,
        metadata={"custom": True}
    )
    
    # Save task
    registry.save_task(task)
    print(f"\nTask '{task_id}' created and saved successfully!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AI Agent Task Management CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List tasks
    list_parser = subparsers.add_parser('list', help='List available tasks')
    list_parser.add_argument('--type', help='Filter by task type')
    list_parser.add_argument('--difficulty', help='Filter by difficulty')
    
    # Show task
    show_parser = subparsers.add_parser('show', help='Show task details')
    show_parser.add_argument('task_id', help='Task ID to show')
    
    # Create config
    config_parser = subparsers.add_parser('config', help='Create task configuration')
    config_parser.add_argument('task_id', help='Task ID')
    config_parser.add_argument('output', help='Output configuration file path')
    
    # Generate data
    data_parser = subparsers.add_parser('generate-data', help='Generate sample data for task')
    data_parser.add_argument('task_id', help='Task ID')
    data_parser.add_argument('output_dir', help='Output directory for data')
    
    # Validate results
    validate_parser = subparsers.add_parser('validate', help='Validate task results')
    validate_parser.add_argument('task_id', help='Task ID')
    validate_parser.add_argument('results_dir', help='Results directory to validate')
    
    # Create custom task
    subparsers.add_parser('create', help='Create a custom task interactively')
    
    # List task types
    subparsers.add_parser('types', help='List available task types')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_tasks(task_type=args.type, difficulty=args.difficulty)
    elif args.command == 'show':
        show_task(args.task_id)
    elif args.command == 'config':
        create_task_config(args.task_id, args.output)
    elif args.command == 'generate-data':
        generate_sample_data(args.task_id, args.output_dir)
    elif args.command == 'validate':
        validate_task_results(args.task_id, args.results_dir)
    elif args.command == 'create':
        create_custom_task()
    elif args.command == 'types':
        types = registry.get_task_types()
        print("Available task types:")
        for task_type in sorted(types):
            print(f"  - {task_type}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()