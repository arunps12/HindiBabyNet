from __future__ import annotations

import argparse

from hindibabynet_pipeline.workflow.textgrid_generation import generate_textgrids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TextGrid files from HindiBabyNet classification outputs.")
    parser.add_argument("--backend", choices=["xgb", "vtc"], required=True)
    parser.add_argument("--participant-id", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--child-label", choices=["key_child", "child"], default="key_child")
    args = parser.parse_args()

    outputs = generate_textgrids(
        backend=args.backend,
        participant_id=args.participant_id,
        input_file=args.input_file,
        limit=args.limit,
        child_label=args.child_label,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()