from hindibabynet_cdi.config import load_config
from hindibabynet_cdi.io import build_form_detection_report, load_detected_forms, summarize_response_vocabularies


def main() -> None:
    config = load_config()
    output_dir = config.paths.outputs / "tables" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_report = build_form_detection_report(config)
    detection_path = output_dir / "raw_form_detection_report.csv"
    detection_report.to_csv(detection_path, index=False, encoding="utf-8-sig")

    loaded_forms = load_detected_forms(config)
    response_report = summarize_response_vocabularies(loaded_forms)
    response_path = output_dir / "raw_response_vocabularies.csv"
    response_report.to_csv(response_path, index=False, encoding="utf-8-sig")

    detected_roles = sorted({loaded.role for loaded in loaded_forms})
    required_roles = {"consent", "eligibility", "background", "cdi_8_18", "cdi_19_36", "contact"}
    missing_roles = sorted(required_roles - set(detected_roles))
    if missing_roles:
        raise SystemExit(f"Missing required form roles: {', '.join(missing_roles)}")

    duplicate_roles = detection_report["detected_role"].value_counts()
    ambiguous = detection_report[detection_report["is_ambiguous"]]

    print(f"Wrote {detection_path}")
    print(f"Wrote {response_path}")
    print(f"Detected roles: {', '.join(detected_roles)}")
    print("Preferred files:")
    for loaded_form in loaded_forms:
        print(f"  {loaded_form.role}: {loaded_form.path.name}")
    if not ambiguous.empty:
        print("Ambiguous detections found:")
        print(ambiguous[["file_name", "detected_role", "matched_columns"]].to_string(index=False))
    print("Role counts:")
    print(duplicate_roles.to_string())


if __name__ == "__main__":
    main()