"""
Stability testing runner for benchmarking tasks.

Runs a task multiple times to measure model consistency and performance stability.
"""

import sys
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .experiment import Experiment
from .config import Config


def extract_task_name(issue_body: str) -> str:
    """Extract task name from GitHub issue body."""
    for line in issue_body.split('\n'):
        if line.startswith('task:'):
            return line.split('task:')[1].strip()
    raise ValueError("No task specified in issue body (expected 'task: <task_name>')")


def extract_model(issue_body: str) -> str:
    """Extract model name from GitHub issue body."""
    for line in issue_body.split('\n'):
        if line.startswith('model:'):
            return line.split('model:')[1].strip()
    raise ValueError("No model specified in issue body (expected 'model: <model_name>')")


def extract_num_runs(issue_body: str, default: int = 10) -> int:
    """Extract number of runs from GitHub issue body."""
    for line in issue_body.split('\n'):
        if line.startswith('num_runs:') or line.startswith('runs:'):
            try:
                return int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                return default
    return default


def run_stability_test(
    task_name: str,
    model: str,
    num_runs: int = 10,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Run task multiple times and compute stability statistics.

    Args:
        task_name: Name of the task to run
        model: Model identifier
        num_runs: Number of times to run the task (default: 10)
        temperature: Model temperature (default: 0.0)

    Returns:
        Dict with stability metrics (mean, std, CI for each metric)
    """
    print(f"\n{'='*70}")
    print(f"Stability Test: {task_name}")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Runs: {num_runs}")
    print(f"Temperature: {temperature}")
    print(f"Started: {datetime.now().isoformat()}\n")

    results_per_run = []

    for run_idx in range(num_runs):
        print(f"\n{'─'*70}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'─'*70}")

        try:
            # Create experiment for this run
            experiment = Experiment(
                model=model,
                task_names=[task_name],
                temperature=temperature
            )

            # Run experiment
            run_results = experiment.run(update_other_experiments=False)

            # Extract task results
            if task_name in run_results:
                task_result = run_results[task_name]
                results_per_run.append(task_result)
                print(f"\n✓ Run {run_idx + 1} completed")

                # Show quick metrics
                overall = task_result.get('overall_metrics', {})
                print(f"  Score: {overall.get('average_score', 0):.3f}")
                if 'average_f1' in overall:
                    print(f"  F1: {overall.get('average_f1', 0):.3f}")
                    print(f"  Precision: {overall.get('average_precision', 0):.3f}")
                    print(f"  Recall: {overall.get('average_recall', 0):.3f}")
            else:
                print(f"\n✗ Run {run_idx + 1} failed: Task {task_name} not in results")

        except Exception as e:
            print(f"\n✗ Run {run_idx + 1} failed with error: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate stability metrics
    if not results_per_run:
        raise ValueError(f"No successful runs completed for task {task_name}")

    stability_summary = aggregate_stability_metrics(
        results_per_run,
        task_name,
        model,
        num_runs
    )

    print(f"\n{'='*70}")
    print(f"Stability Test Complete!")
    print(f"{'='*70}")
    print(f"Completed runs: {len(results_per_run)}/{num_runs}")
    print(f"Ended: {datetime.now().isoformat()}\n")

    return stability_summary


def aggregate_stability_metrics(
    results: List[Dict[str, Any]],
    task_name: str,
    model: str,
    num_runs: int
) -> Dict[str, Any]:
    """
    Compute mean, std, and 95% CI for all metrics.

    Args:
        results: List of task results from multiple runs
        task_name: Task name
        model: Model identifier
        num_runs: Number of runs attempted

    Returns:
        Dict with aggregated stability metrics
    """
    metrics_to_aggregate = [
        'average_score',
        'average_f1',
        'average_precision',
        'average_recall',
        'average_confidence'
    ]

    aggregated = {
        'task_name': task_name,
        'model': model,
        'num_runs_attempted': num_runs,
        'num_runs_completed': len(results),
        'timestamp': datetime.now().isoformat()
    }

    for metric in metrics_to_aggregate:
        values = []
        for r in results:
            overall = r.get('overall_metrics', {})
            if metric in overall:
                val = overall[metric]
                if val is not None:
                    values.append(val)

        if values:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0

            # 95% confidence interval (z=1.96 for 95% CI)
            ci_margin = 1.96 * (std_val / (len(values) ** 0.5))

            metric_short = metric.replace('average_', '')
            aggregated[metric_short] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': mean_val - ci_margin,
                'ci_upper': mean_val + ci_margin,
                'min': min(values),
                'max': max(values),
                'median': statistics.median(values),
                'values': values  # Store all values for detailed analysis
            }

    return aggregated


def save_stability_summary(summary: Dict[str, Any], output_dir: Path):
    """Save stability summary to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main summary
    summary_path = output_dir / "stability_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nStability summary saved to: {summary_path}")

    # Also save timestamped version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name_safe = summary['task_name'].replace('/', '_')
    timestamped_path = output_dir / f"stability_{task_name_safe}_{timestamp}.json"
    with open(timestamped_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Timestamped copy saved to: {timestamped_path}")


def print_stability_summary(summary: Dict[str, Any]):
    """Print formatted stability summary."""
    print(f"\n{'='*70}")
    print(f"STABILITY SUMMARY")
    print(f"{'='*70}")
    print(f"Task: {summary['task_name']}")
    print(f"Model: {summary['model']}")
    print(f"Completed: {summary['num_runs_completed']}/{summary['num_runs_attempted']} runs")
    print(f"\nMetrics (Mean ± Std Dev):\n")

    for metric in ['score', 'f1', 'precision', 'recall', 'confidence']:
        if metric in summary:
            data = summary[metric]
            print(f"  {metric.upper()}:")
            print(f"    Mean: {data['mean']:.4f}")
            print(f"    Std Dev: {data['std']:.4f}")
            print(f"    95% CI: [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]")
            print(f"    Range: [{data['min']:.4f}, {data['max']:.4f}]")
            print(f"    Median: {data['median']:.4f}\n")

    print(f"{'='*70}\n")


def main():
    """Main entry point for stability testing."""
    if len(sys.argv) < 3:
        print("Usage: python -m src.stability_runner '<issue_body>' '<issue_number>'")
        print("\nExpected issue body format:")
        print("  task: <task_name>")
        print("  model: <model_name>")
        print("  num_runs: <number> (optional, default: 10)")
        sys.exit(1)

    issue_body = sys.argv[1]
    issue_number = sys.argv[2]

    try:
        # Extract parameters from issue body
        task_name = extract_task_name(issue_body)
        model = extract_model(issue_body)
        num_runs = extract_num_runs(issue_body)

        # Run stability test
        summary = run_stability_test(task_name, model, num_runs)

        # Save results
        results_dir = Path("docs/results")
        save_stability_summary(summary, results_dir)

        # Print summary
        print_stability_summary(summary)

        print(f"✅ Stability test completed successfully!")
        print(f"   Issue #{issue_number} can be closed.")

    except Exception as e:
        print(f"\n❌ Stability test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
