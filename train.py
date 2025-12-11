from __future__ import annotations
import argparse
import os
from pathlib import Path
from src.train_tf import train

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--model", default="hybrid", choices=["lstm", "hybrid"])
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--outputsize", default="full", choices=["full", "compact"])
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args()

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise SystemExit("Missing ALPHAVANTAGE_API_KEY env var.")

    meta = train(
        symbol=args.symbol,
        api_key=api_key,
        model_type=args.model,
        window_size=args.window_size,
        epochs=args.epochs,
        output_dir=Path(args.artifacts),
        outputsize=args.outputsize,
    )

    print(f"\nSaved to: {Path(args.artifacts) / args.symbol.upper()}")
    print(f"Validation RMSE: {meta[metrics][val_rmse]:.6f}")
    print(f"Next-day predicted close: {meta[next_day_predicted_close]:.4f}")

if __name__ == "__main__":
    main()
