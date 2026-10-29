"""CLI interface for PepScan 1.0."""

import argparse
import json
from pepscan.predict import PepScanPredictor
from pepscan.tasks import TASK_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="PepScan 1.0: Multi-task peptide bioactivity prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tasks:
  {', '.join(TASK_NAMES)}

Examples:
  pepscan KWKLFKKIGAVLKVL
  pepscan KWKLFKKIGAVLKVL --tasks AMP Toxicity
  pepscan KWKLFKKIGAVLKVL --tasks all --json
  pepscan --file sequences.txt
        """
    )
    parser.add_argument("sequence", nargs="?",
                        help="Peptide sequence (single-letter AA codes)")
    parser.add_argument("--file", "-f",
                        help="File with one sequence per line")
    parser.add_argument("--tasks", "-t", nargs="+", default=["all"],
                        help="Tasks to predict (default: all)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--weights", "-w",
                        help="Path to local model weights")
    parser.add_argument("--device", "-d",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    if not args.sequence and not args.file:
        parser.print_help()
        return

    pred = PepScanPredictor(weights_path=args.weights, device=args.device)
    tasks = args.tasks[0] if args.tasks == ["all"] else args.tasks

    if args.file:
        with open(args.file) as f:
            sequences = [line.strip() for line in f
                         if line.strip() and not line.startswith("#")]
        results = pred.predict_batch(sequences, tasks=tasks)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for seq, res in zip(sequences, results):
                print(f"\n> {seq}")
                for task, info in res.items():
                    marker = "+" if info["prediction"] == "positive" else "-"
                    print(f"  {task:20s}  {info['probability']:.4f}  {marker}")
    else:
        result = pred.predict(args.sequence, tasks=tasks)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nSequence: {args.sequence}")
            for task, info in result.items():
                marker = "+" if info["prediction"] == "positive" else "-"
                print(f"  {task:20s}  {info['probability']:.4f}  {marker}")


if __name__ == "__main__":
    main()
