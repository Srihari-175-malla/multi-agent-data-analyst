#!/usr/bin/env python
"""
CLI entry point: load a dataset and run the full multi-agent analysis workflow.

Usage:
    python scripts/run_analysis.py --data data/sample/sales.csv \
        --question "Why did sales decrease last month?"
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import DataLoader  # noqa: E402
from src.orchestration.workflow import AnalysisWorkflow  # noqa: E402
from src.utils_logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to a CSV/Excel file")
    parser.add_argument("--question", required=True, help="Analytical question to answer")
    args = parser.parse_args()

    dataset = DataLoader.load(args.data)
    workflow = AnalysisWorkflow()
    result = workflow.run(dataset.dataset_id, args.question)

    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(result.report)
    print("\nCharts:", result.chart_paths)
    print(f"\nRevision rounds used: {result.revision_rounds_used}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
