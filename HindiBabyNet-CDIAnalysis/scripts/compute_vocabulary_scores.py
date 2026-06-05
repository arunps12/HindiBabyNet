"""Compute participant-level and category-level CDI vocabulary scores."""

from hindibabynet_cdi.config import load_config
from hindibabynet_cdi.scoring import build_scoring_outputs


def main() -> None:
    config = load_config()
    outputs = build_scoring_outputs(config=config)
    config.paths.processed_data.mkdir(parents=True, exist_ok=True)
    outputs["participant_metadata"].to_csv(config.paths.processed_data / "participant_info.csv", index=False)
    outputs["cdi_participant_scores"].to_csv(config.paths.processed_data / "vocabulary_scores.csv", index=False)
    outputs["cdi_category_scores"].to_csv(config.paths.processed_data / "category_scores.csv", index=False)
    outputs["cdi_master_dataset"].to_csv(config.paths.processed_data / "master_dataset.csv", index=False)
    print(f"Wrote {config.paths.processed_data / 'participant_info.csv'}")
    print(f"Wrote {config.paths.processed_data / 'vocabulary_scores.csv'}")
    print(f"Wrote {config.paths.processed_data / 'category_scores.csv'}")
    print(f"Wrote {config.paths.processed_data / 'master_dataset.csv'}")


if __name__ == "__main__":
    main()