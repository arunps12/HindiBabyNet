from __future__ import annotations

import argparse

from hindibabynet_pipeline.workflow.model_evaluation import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate automatic speaker labels against manual annotations.")
    parser.add_argument("--participant-id", default=None)
    args = parser.parse_args()

    outputs = evaluate_models(participant_id=args.participant_id)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()