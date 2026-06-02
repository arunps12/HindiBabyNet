from __future__ import annotations

import argparse

from hindibabynet_pipeline.workflow.manual_annotation import annotate_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually annotate speaker-class segments.")
    parser.add_argument("--backend", choices=["xgb", "vtc"], required=True)
    parser.add_argument("--participant-id", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--n-segments", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--child-label", choices=["key_child", "child"], default="key_child")
    args = parser.parse_args()

    outputs = annotate_segments(
        backend=args.backend,
        participant_id=args.participant_id,
        input_file=args.input_file,
        n_segments=args.n_segments,
        random_seed=args.random_seed,
        child_label=args.child_label,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()