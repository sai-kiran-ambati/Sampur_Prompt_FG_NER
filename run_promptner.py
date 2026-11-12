"""Run hosted PromptNER inference and evaluation on SampurNER Telugu."""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from promptner.evaluation import evaluate


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=os.getenv("SAMPURNER_TELUGU_DIR", "sampurner_telugu_saved"))
    parser.add_argument("--provider", choices=("openai", "sarvam"), default=os.getenv("PROMPTNER_PROVIDER", "sarvam"))
    parser.add_argument("--model", default=None, help="Model override; defaults are provider-specific")
    parser.add_argument("--limit", type=int, default=500, help="Number of test sentences (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent hosted requests (default: 4)")
    parser.add_argument(
        "--method",
        choices=("two-stage", "one-stage"),
        default="two-stage",
        help="NER inference method (default: two-stage)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true", help="Retry failed API requests in a matching output file")
    args = parser.parse_args()
    output = args.output or f"outputs/{args.method}_{args.provider}_eval.jsonl"

    result = evaluate(
        dataset_dir=args.dataset_dir,
        provider=args.provider,
        model=args.model,
        limit=args.limit,
        seed=args.seed,
        output=output,
        workers=args.workers,
        resume=args.resume,
        method=args.method,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
