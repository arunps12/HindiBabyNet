from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


SELECTED_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
DECILES = tuple(value / 100 for value in range(10, 100, 10))
QUARTILES = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class SmoothCurveResult:
    predictions: pd.DataFrame
    metadata: pd.DataFrame


def empirical_norm_table(
    data: pd.DataFrame,
    measure_column: str,
    age_months: list[int],
    quantiles: tuple[float, ...] = DECILES,
) -> pd.DataFrame:
    included = data.loc[data["included_final"]].copy()
    rows: list[dict[str, float | int | None]] = []

    for age_month in age_months:
        frame = included.loc[included["age_month"].eq(age_month)].copy()
        values = pd.to_numeric(frame[measure_column], errors="coerce").dropna()
        row: dict[str, float | int | None] = {
            "age_month": age_month,
            "n": int(len(values)),
            "mean": float(values.mean()) if len(values) else np.nan,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "min": float(values.min()) if len(values) else np.nan,
            "max": float(values.max()) if len(values) else np.nan,
        }
        for quantile in quantiles:
            label = int(round(quantile * 100))
            row[f"p{label}"] = float(values.quantile(quantile)) if len(values) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def grouped_empirical_table(
    data: pd.DataFrame,
    measure_column: str,
    group_column: str,
    age_months: list[int],
    quantiles: tuple[float, ...],
) -> pd.DataFrame:
    included = data.loc[data["included_final"]].copy()
    rows: list[dict[str, object]] = []

    for age_month in age_months:
        for group_value, frame in included.loc[included["age_month"].eq(age_month)].groupby(group_column, dropna=False):
            values = pd.to_numeric(frame[measure_column], errors="coerce").dropna()
            row: dict[str, object] = {
                "age_month": age_month,
                group_column: group_value,
                "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
            }
            for quantile in quantiles:
                label = int(round(quantile * 100))
                row[f"p{label}"] = float(values.quantile(quantile)) if len(values) else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def _spline_df(n_unique_ages: int) -> int:
    return max(3, min(5, n_unique_ages - 1)) if n_unique_ages > 3 else 1


def _clip_predictions(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return np.clip(values, minimum, maximum)


def _prediction_grid(age_months: list[int], observed: pd.Series) -> pd.DataFrame:
    observed_min = float(observed.min())
    observed_max = float(observed.max())
    clipped = [min(max(float(age_month), observed_min), observed_max) for age_month in age_months]
    return pd.DataFrame({"age_month": age_months, "age_months_exact": clipped})


def fit_mean_curve(
    data: pd.DataFrame,
    measure_column: str,
    age_months: list[int],
    maximum_score: int,
    label: str,
) -> SmoothCurveResult:
    included = data.loc[data["included_final"]].copy()
    model_data = included[["age_months_exact", measure_column]].dropna().copy()
    n_unique_ages = model_data["age_months_exact"].nunique()
    metadata_rows: list[dict[str, object]] = []

    if len(model_data) < 6 or n_unique_ages < 4:
        empirical = empirical_norm_table(data, measure_column, age_months, quantiles=(0.50,))[["age_month", "p50"]].rename(
            columns={"p50": "predicted"}
        )
        empirical["curve_type"] = label
        metadata_rows.append(
            {
                "curve": label,
                "model_type": "empirical_median_fallback",
                "formula": "empirical median by age month",
                "n_obs": int(len(model_data)),
            }
        )
        return SmoothCurveResult(predictions=empirical, metadata=pd.DataFrame(metadata_rows))

    spline_df = _spline_df(n_unique_ages)
    design = patsy.dmatrix(f"bs(age_months_exact, df={spline_df}, degree=3, include_intercept=False)", model_data)
    model = sm.OLS(model_data[measure_column], design).fit()
    prediction_grid = _prediction_grid(age_months, model_data["age_months_exact"])
    prediction_design = patsy.build_design_matrices([design.design_info], prediction_grid)[0]
    predicted = model.predict(prediction_design)

    predictions = pd.DataFrame(
        {
            "age_month": prediction_grid["age_month"],
            "predicted": _clip_predictions(np.asarray(predicted), 0, maximum_score),
            "curve_type": label,
        }
    )
    metadata_rows.append(
        {
            "curve": label,
            "model_type": "ols_bspline",
            "formula": f"score ~ bs(age_months_exact, df={spline_df}, degree=3)",
            "n_obs": int(len(model_data)),
            "r_squared": float(model.rsquared),
        }
    )
    return SmoothCurveResult(predictions=predictions, metadata=pd.DataFrame(metadata_rows))


def fit_quantile_curves(
    data: pd.DataFrame,
    measure_column: str,
    age_months: list[int],
    quantiles: tuple[float, ...],
    maximum_score: int,
    label_prefix: str,
) -> SmoothCurveResult:
    included = data.loc[data["included_final"]].copy()
    model_data = included[["age_months_exact", measure_column]].dropna().copy()
    n_unique_ages = model_data["age_months_exact"].nunique()
    metadata_rows: list[dict[str, object]] = []

    if len(model_data) < 20 or n_unique_ages < 5:
        empirical = empirical_norm_table(data, measure_column, age_months, quantiles=quantiles)
        frames: list[pd.DataFrame] = []
        for quantile in quantiles:
            label = int(round(quantile * 100))
            frame = empirical[["age_month", f"p{label}"]].rename(columns={f"p{label}": "predicted"})
            frame["curve_type"] = f"{label_prefix}_p{label}"
            frame["quantile"] = quantile
            frames.append(frame)
            metadata_rows.append(
                {
                    "curve": f"{label_prefix}_p{label}",
                    "model_type": "empirical_quantile_fallback",
                    "formula": f"empirical quantile {quantile}",
                    "n_obs": int(len(model_data)),
                }
            )
        return SmoothCurveResult(predictions=pd.concat(frames, ignore_index=True), metadata=pd.DataFrame(metadata_rows))

    spline_df = _spline_df(n_unique_ages)
    design = patsy.dmatrix(f"bs(age_months_exact, df={spline_df}, degree=3, include_intercept=False)", model_data)
    prediction_grid = _prediction_grid(age_months, model_data["age_months_exact"])
    prediction_design = patsy.build_design_matrices([design.design_info], prediction_grid)[0]

    predictions: list[pd.DataFrame] = []
    previous_values: np.ndarray | None = None
    for quantile in sorted(quantiles):
        label = int(round(quantile * 100))
        try:
            model = sm.QuantReg(model_data[measure_column], design).fit(q=quantile, max_iter=5000)
            predicted = np.asarray(model.predict(prediction_design))
            model_type = "quantreg_bspline"
            formula = f"score ~ bs(age_months_exact, df={spline_df}, degree=3), q={quantile}"
        except Exception:
            empirical = empirical_norm_table(data, measure_column, age_months, quantiles=(quantile,))
            predicted = empirical[f"p{label}"].to_numpy()
            model_type = "empirical_quantile_fallback"
            formula = f"empirical quantile {quantile}"

        predicted = _clip_predictions(predicted, 0, maximum_score)
        if previous_values is not None:
            predicted = np.maximum(predicted, previous_values)
        previous_values = predicted

        frame = pd.DataFrame(
            {
                    "age_month": prediction_grid["age_month"],
                "predicted": predicted,
                "curve_type": f"{label_prefix}_p{label}",
                "quantile": quantile,
            }
        )
        predictions.append(frame)
        metadata_rows.append(
            {
                "curve": f"{label_prefix}_p{label}",
                "model_type": model_type,
                "formula": formula,
                "n_obs": int(len(model_data)),
            }
        )

    return SmoothCurveResult(predictions=pd.concat(predictions, ignore_index=True), metadata=pd.DataFrame(metadata_rows))


def fit_group_mean_curves(
    data: pd.DataFrame,
    measure_column: str,
    group_column: str,
    age_months: list[int],
    maximum_score: int,
    min_group_n: int = 8,
) -> SmoothCurveResult:
    predictions: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []

    included = data.loc[data["included_final"]].copy()
    for group_value, frame in included.groupby(group_column, dropna=False):
        clean = frame[["age_months_exact", measure_column]].dropna().copy()
        label = str(group_value)
        if len(clean) < min_group_n:
            empirical = empirical_norm_table(frame.assign(included_final=True), measure_column, age_months, quantiles=(0.50,))
            pred = empirical[["age_month", "p50"]].rename(columns={"p50": "predicted"})
            pred["group"] = label
            predictions.append(pred)
            metadata.append(
                {
                    "curve": label,
                    "model_type": "empirical_group_fallback",
                    "formula": "empirical median by age month",
                    "n_obs": int(len(clean)),
                }
            )
            continue

        result = fit_mean_curve(frame.assign(included_final=True), measure_column, age_months, maximum_score, label)
        pred = result.predictions[["age_month", "predicted"]].copy()
        pred["group"] = label
        predictions.append(pred)
        metadata.extend(result.metadata.assign(group=label).to_dict(orient="records"))

    return SmoothCurveResult(predictions=pd.concat(predictions, ignore_index=True), metadata=pd.DataFrame(metadata))


def fit_item_probability_curves(
    item_long: pd.DataFrame,
    response_column: str,
    age_months: list[int],
    min_item_n: int = 20,
    min_positive: int = 3,
) -> SmoothCurveResult:
    included = item_long.loc[item_long["included_final"] & item_long["cdi_is_complete_enough"]].copy()
    predictions: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []

    for item_id, frame in included.groupby("item_id", dropna=False):
        clean = frame[["age_months_exact", response_column, "word", "word_english"]].dropna().copy()
        n_obs = len(clean)
        positives = int(pd.to_numeric(clean[response_column], errors="coerce").sum())
        if n_obs < min_item_n or positives < min_positive or positives == n_obs:
            empirical = (
                frame.groupby("age_month", dropna=False)[response_column]
                .agg(["mean", "size"])
                .reset_index()
                .rename(columns={"mean": "predicted", "size": "n"})
            )
            empirical = empirical[empirical["age_month"].isin(age_months)]
            empirical["item_id"] = item_id
            empirical["word"] = clean["word"].iloc[0] if n_obs else frame["word"].iloc[0]
            empirical["word_english"] = clean["word_english"].iloc[0] if n_obs else frame["word_english"].iloc[0]
            empirical["insufficient_data"] = True
            predictions.append(empirical[["age_month", "predicted", "item_id", "word", "word_english", "insufficient_data"]])
            metadata.append(
                {
                    "curve": item_id,
                    "model_type": "empirical_item_fallback",
                    "formula": f"empirical mean of {response_column}",
                    "n_obs": n_obs,
                    "n_positive": positives,
                }
            )
            continue

        spline_df = _spline_df(clean["age_months_exact"].nunique())
        design = patsy.dmatrix(f"bs(age_months_exact, df={spline_df}, degree=3, include_intercept=False)", clean)
        model = sm.GLM(clean[response_column], design, family=sm.families.Binomial()).fit()
        prediction_grid = _prediction_grid(age_months, clean["age_months_exact"])
        prediction_design = patsy.build_design_matrices([design.design_info], prediction_grid)[0]
        predicted = np.asarray(model.predict(prediction_design))
        predictions.append(
            pd.DataFrame(
                {
                    "age_month": prediction_grid["age_month"],
                    "predicted": _clip_predictions(predicted, 0, 1),
                    "item_id": item_id,
                    "word": clean["word"].iloc[0],
                    "word_english": clean["word_english"].iloc[0],
                    "insufficient_data": False,
                }
            )
        )
        metadata.append(
            {
                "curve": item_id,
                "model_type": "binomial_glm_bspline",
                "formula": f"{response_column} ~ bs(age_months_exact, df={spline_df}, degree=3)",
                "n_obs": n_obs,
                "n_positive": positives,
            }
        )

    return SmoothCurveResult(predictions=pd.concat(predictions, ignore_index=True), metadata=pd.DataFrame(metadata))


def category_proportion_long(item_long: pd.DataFrame, response_column: str) -> pd.DataFrame:
    included = item_long.loc[item_long["included_final"] & item_long["cdi_is_complete_enough"]].copy()
    included[response_column] = pd.to_numeric(included[response_column], errors="coerce")
    return (
        included.groupby(["participant_id", "form", "category", "category_english", "age_month", "age_months_exact"], dropna=False)[response_column]
        .mean()
        .reset_index(name="category_proportion")
    )