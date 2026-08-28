# -*- coding: utf-8 -*-
"""
CLI entrypoint for the unified pipeline.
"""

import argparse
import json

from .router import ReportRequest
from .pipeline import ReportPipeline


def _parse_args():
    parser = argparse.ArgumentParser(description="SOA report pipeline")
    parser.add_argument("--report-type", choices=["policy_report", "spotlight_report", "trend_pack", "validation_diagnostic"])
    parser.add_argument("--use-test-data", action="store_true", help="Use deterministic test sample")
    parser.add_argument("--input-json", type=str, help="Inline JSON for input record")
    parser.add_argument("--input-file", type=str, help="Path to JSON file for input record")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--top-n-spotlight", type=int, default=5, help="Top N anomalies for spotlight report")
    return parser.parse_args()


def _load_input(args) -> dict | None:
    if args.input_json:
        return json.loads(args.input_json)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    args = _parse_args()
    input_record = _load_input(args)

    request = ReportRequest(
        report_type=args.report_type,
        input_record=input_record,
        use_test_data=args.use_test_data,
        run_validation=True,
        top_n_spotlight=args.top_n_spotlight,
    )

    pipeline = ReportPipeline(output_dir=args.output_dir)
    result = pipeline.run(request)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
