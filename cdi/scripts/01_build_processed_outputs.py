from __future__ import annotations

from hindibabynet_cdi.pipeline import run_pipeline


def main() -> None:
    outputs = run_pipeline()
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()