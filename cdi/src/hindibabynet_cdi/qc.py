from __future__ import annotations

import pandas as pd


def build_sample_characteristics(participant_analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    included = participant_analysis.loc[participant_analysis["included_final"]].copy()

    for form, frame in included.groupby("form", dropna=False):
        ages = pd.to_numeric(frame["age_months_exact"], errors="coerce")
        hindi = pd.to_numeric(frame["hindi_percentage"], errors="coerce")

        rows.extend(
            [
                {"form": form, "Variable": "N", "Value": int(len(frame))},
                {
                    "form": form,
                    "Variable": "Age, mean (SD)",
                    "Value": f"{ages.mean():.2f} ({ages.std(ddof=1):.2f})" if len(frame) > 1 else f"{ages.mean():.2f} (NA)",
                },
                {
                    "form": form,
                    "Variable": "Age, median [IQR]",
                    "Value": f"{ages.median():.2f} [{ages.quantile(0.25):.2f}, {ages.quantile(0.75):.2f}]",
                },
                {
                    "form": form,
                    "Variable": "Age range",
                    "Value": f"{ages.min():.2f}-{ages.max():.2f}",
                },
                {
                    "form": form,
                    "Variable": "Hindi exposure, mean (SD)",
                    "Value": f"{hindi.mean():.2f} ({hindi.std(ddof=1):.2f})" if len(frame) > 1 else f"{hindi.mean():.2f} (NA)",
                },
            ]
        )

        for sex in ["Female", "Male"]:
            count = int(frame["sex"].eq(sex).sum())
            percent = (count / len(frame) * 100) if len(frame) else 0
            rows.append({"form": form, "Variable": f"{sex}, n (%)", "Value": f"{count} ({percent:.1f}%)"})

        for label in ["primary_school", "high_school", "some_college", "bachelor", "master"]:
            count = int(frame["mother_education"].eq(label).sum())
            percent = (count / len(frame) * 100) if len(frame) else 0
            rows.append(
                {
                    "form": form,
                    "Variable": f"Mother's education: {label}, n (%)",
                    "Value": f"{count} ({percent:.1f}%)",
                }
            )

    return pd.DataFrame(rows)


def build_age_counts(participant_analysis: pd.DataFrame) -> pd.DataFrame:
    included = participant_analysis.loc[participant_analysis["included_final"]].copy()
    return (
        included.groupby(["form", "age_month"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["form", "age_month"])
        .reset_index(drop=True)
    )