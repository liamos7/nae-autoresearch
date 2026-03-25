"""
evaluate.py — READ-ONLY evaluation harness for NAE autoresearch
================================================================
DO NOT MODIFY THIS FILE. The agent only modifies train.py.

This script:
1. Runs train.py with a fixed time budget
2. Reads the metrics.json output
3. Computes a single scalar score for comparison
4. Logs the result

The primary metric is a composite score that rewards:
  - High AUC on anomaly detection (most important)
  - Training stability (no collapse, no NaN)
  - Reasonable energy separation between inliers and outliers

Score = val_auc * stability_multiplier

Where stability_multiplier is:
  1.0 if training was stable
  0.0 if training collapsed or produced NaN
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime


def compute_score(metrics: dict) -> float:
    """
    Compute a single scalar score from training metrics.
    Higher is better. Range: [0.0, 1.0]
    
    The score is designed so that:
    - A collapsed run scores 0.0
    - A stable run scores val_auc (typically 0.5 - 1.0)
    - AUC is the primary metric since this is an anomaly detection task
    """
    if metrics.get('collapsed', True):
        return 0.0
    
    if not metrics.get('energy_stable', False):
        return 0.0
    
    auc = metrics.get('best_val_auc', 0.0)
    
    # Bonus for completing all epochs without issues
    epochs_completed = metrics.get('epochs_completed', 0)
    epochs_target = 50  # matches EPOCHS in train.py default
    completion_bonus = min(epochs_completed / max(epochs_target, 1), 1.0)
    
    # Small bonus for energy separation (neg > pos means model learned contrast)
    pos_e = metrics.get('final_pos_energy', 0)
    neg_e = metrics.get('final_neg_energy', 0)
    if isinstance(pos_e, (int, float)) and isinstance(neg_e, (int, float)):
        separation = 1.0 if neg_e > pos_e else 0.95
    else:
        separation = 0.9
    
    score = auc * completion_bonus * separation
    return round(score, 6)


def run_experiment(args):
    """Run a single training experiment and return the score."""
    
    output_dir = os.path.join(args.log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "train.py",
        "--dataset", args.dataset,
        "--holdout-class", str(args.holdout_class),
        "--data-root", args.data_root,
        "--output-dir", output_dir,
    ]
    if args.pretrained_path:
        cmd += ["--pretrained-path", args.pretrained_path]
    
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start = time.time()
    
    try:
        # Run with time budget
        result = subprocess.run(
            cmd,
            timeout=args.time_budget,
            capture_output=not args.verbose,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"EXPERIMENT FAILED (return code {result.returncode})")
            if not args.verbose and result.stderr:
                print(f"STDERR:\n{result.stderr[-2000:]}")  # last 2000 chars
            return 0.0, {"error": "nonzero return code", "returncode": result.returncode}
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"EXPERIMENT TIMED OUT after {elapsed:.0f}s")
        # Still try to read metrics if they were written
    except Exception as e:
        print(f"EXPERIMENT ERROR: {e}")
        return 0.0, {"error": str(e)}
    
    # Read metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print("No metrics.json found — experiment may have crashed before writing results.")
        return 0.0, {"error": "no metrics.json"}
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    score = compute_score(metrics)
    metrics['score'] = score
    metrics['elapsed'] = elapsed
    
    # Save enriched metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT RESULT")
    print(f"  Score:        {score:.6f}")
    print(f"  Best AUC:     {metrics.get('best_val_auc', 'N/A')}")
    print(f"  Collapsed:    {metrics.get('collapsed', 'N/A')}")
    print(f"  Stable:       {metrics.get('energy_stable', 'N/A')}")
    print(f"  Epochs:       {metrics.get('epochs_completed', 'N/A')}")
    print(f"  Time:         {elapsed:.0f}s")
    print(f"{'='*60}\n")
    
    return score, metrics


def load_best_score(log_dir):
    """Load the best score from previous runs."""
    history_path = os.path.join(log_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        if history:
            return max(h['score'] for h in history)
    return 0.0


def save_to_history(log_dir, entry):
    """Append an experiment result to the history log."""
    history_path = os.path.join(log_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []
    history.append(entry)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAE Autoresearch Evaluation Harness (DO NOT MODIFY)")
    parser.add_argument("--dataset", type=str, default="CICADA",
                        choices=["MNIST", "FMNIST", "CIFAR10", "CICADA"])
    parser.add_argument("--holdout-class", type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help="Which class(es) to use as anomaly. Single int or comma-separated (e.g. '1,2')")
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path to Phase 1 AE weights")
    parser.add_argument("--data-root", type=str, default="/scratch/network/lo8603/thesis/fast-ad/data/h5_files/")
    parser.add_argument("--log-dir", type=str, default="./autoresearch_logs",
                        help="Directory to store experiment logs and history")
    parser.add_argument("--time-budget", type=int, default=600,
                        help="Maximum wall-clock seconds per experiment (default: 600 = 10 min)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show training output in real-time")
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    score, metrics = run_experiment(args)
    
    save_to_history(args.log_dir, {
        'timestamp': datetime.now().isoformat(),
        'score': score,
        'best_val_auc': metrics.get('best_val_auc'),
        'collapsed': metrics.get('collapsed'),
        'epochs_completed': metrics.get('epochs_completed'),
    })
    
    best_so_far = load_best_score(args.log_dir)
    if score > best_so_far:
        print(f"*** NEW BEST SCORE: {score:.6f} (previous best: {best_so_far:.6f}) ***")
    else:
        print(f"Score {score:.6f} did not beat best {best_so_far:.6f}")