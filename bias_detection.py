import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd


def _require_aif360() -> None:
    try:
        import aif360  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: aif360.\n\n"
            "Install it, then re-run:\n"
            "  pip install aif360\n\n"
            f"Original error: {e}"
        )


def _encode_categorical(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("Unknown").str.strip()
    s = s.mask(s.eq("") | s.str.lower().eq("nan") | s.str.lower().eq("none"), "Unknown")
    cat = pd.Categorical(s)
    return pd.Series(cat.codes.astype(int), index=series.index), [str(x) for x in cat.categories.tolist()]


def _age_to_group(age_series: pd.Series, threshold: float = 40) -> pd.Series:
    # privileged (1) if age >= threshold, else unprivileged (0)
    age = pd.to_numeric(age_series, errors="coerce")
    if age.notna().sum() == 0:
        return pd.Series([0] * len(age_series), index=age_series.index, dtype=int)
    return (age.fillna(age.median()) >= threshold).astype(int)


def _binary_label(series: pd.Series) -> pd.Series:
    # Ensures 0/1 labels. Supports common yes/no style strings.
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce").fillna(0)
        return (s > 0).astype(int)

    s = series.astype("string").fillna("").str.strip().str.lower()
    positive = {"1", "true", "t", "yes", "y", "hired", "selected", "accept", "accepted"}
    return s.isin(positive).astype(int)


def _most_frequent_value(series: pd.Series) -> Any:
    vc = series.value_counts(dropna=True)
    return vc.index[0] if len(vc) else None


def main(
    input_csv: str = "cleaned_data.csv",
    output_json: str = "results.json",
    label_col: str = "hired",
    gender_col: str = "gender",
    race_col: str = "race",
    education_col: str = "education",
    age_col: str = "age",
    age_privileged_threshold: float = 40,
) -> None:
    _require_aif360()
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric

    df = pd.read_csv(input_csv)

    missing = [c for c in [label_col, gender_col, race_col, education_col, age_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s) in {input_csv}: {missing}")

    # Keep original protected attributes to choose privileged groups sensibly.
    gender_raw = df[gender_col].astype("string").fillna("Unknown").str.strip().replace("", "Unknown")
    race_raw = df[race_col].astype("string").fillna("Unknown").str.strip().replace("", "Unknown")

    # 1) Encode gender, race, education into numbers (kept as *_enc).
    df["gender_enc"], gender_categories = _encode_categorical(df[gender_col])
    df["race_enc"], race_categories = _encode_categorical(df[race_col])
    df["education_enc"], education_categories = _encode_categorical(df[education_col])

    # Age as a binary protected attribute for AIF360 metrics.
    df["age_group"] = _age_to_group(df[age_col], threshold=age_privileged_threshold)

    # Ensure binary label.
    df[label_col] = _binary_label(df[label_col])

    # Choose privileged classes:
    # - gender/race: most frequent group in dataset (common default when no domain choice provided)
    # - age_group: 1 means age >= threshold
    gender_priv_raw = _most_frequent_value(gender_raw)
    race_priv_raw = _most_frequent_value(race_raw)

    gender_priv_code = int(df.loc[gender_raw.eq(gender_priv_raw), "gender_enc"].iloc[0]) if gender_priv_raw is not None else 0
    race_priv_code = int(df.loc[race_raw.eq(race_priv_raw), "race_enc"].iloc[0]) if race_priv_raw is not None else 0

    # AIF360 requires all values to be numeric: drop any remaining non-numeric columns (e.g., name).
    df_num = df.copy()
    df_num = df_num.select_dtypes(include=["number"]).copy()

    gender_attr = "gender_enc"
    race_attr = "race_enc"

    dataset = BinaryLabelDataset(
        df=df_num,
        label_names=[label_col],
        protected_attribute_names=[gender_attr, race_attr, "age_group"],
        favorable_label=1,
        unfavorable_label=0,
    )

    def metric_for(ds: BinaryLabelDataset, attr: str, privileged: List[Any], unprivileged: List[Any]) -> Dict[str, float]:
        metric = BinaryLabelDatasetMetric(
            ds,
            unprivileged_groups=[{attr: unprivileged[0]}],
            privileged_groups=[{attr: privileged[0]}],
        )
        return {
            "mean_difference": float(metric.mean_difference()),
            "statistical_parity_difference": float(metric.statistical_parity_difference()),
            "disparate_impact": float(metric.disparate_impact()),
            "base_rate_unprivileged": float(metric.base_rate(privileged=False)),
            "base_rate_privileged": float(metric.base_rate(privileged=True)),
        }

    # Unprivileged class for gender/race: pick a different value if possible, else same.
    gender_unpriv_raw = next((v for v in gender_raw.dropna().unique().tolist() if v != gender_priv_raw), gender_priv_raw)
    race_unpriv_raw = next((v for v in race_raw.dropna().unique().tolist() if v != race_priv_raw), race_priv_raw)

    gender_unpriv_code = int(df.loc[gender_raw.eq(gender_unpriv_raw), "gender_enc"].iloc[0]) if gender_unpriv_raw is not None else gender_priv_code
    race_unpriv_code = int(df.loc[race_raw.eq(race_unpriv_raw), "race_enc"].iloc[0]) if race_unpriv_raw is not None else race_priv_code

    # 2) Run BinaryLabelDatasetMetric for gender, race, and age.
    bias_before = {
        "gender": {
            "attr": gender_attr,
            "privileged_value": gender_priv_code,
            "unprivileged_value": gender_unpriv_code,
            "raw_privileged_value": None if gender_priv_raw is None else str(gender_priv_raw),
            "raw_unprivileged_value": None if gender_unpriv_raw is None else str(gender_unpriv_raw),
            "metrics": metric_for(dataset, gender_attr, [gender_priv_code], [gender_unpriv_code]),
        },
        "race": {
            "attr": race_attr,
            "privileged_value": race_priv_code,
            "unprivileged_value": race_unpriv_code,
            "raw_privileged_value": None if race_priv_raw is None else str(race_priv_raw),
            "raw_unprivileged_value": None if race_unpriv_raw is None else str(race_unpriv_raw),
            "metrics": metric_for(dataset, race_attr, [race_priv_code], [race_unpriv_code]),
        },
        "age": {
            "attr": "age_group",
            "privileged_value": 1,
            "unprivileged_value": 0,
            "threshold": age_privileged_threshold,
            "metrics": metric_for(dataset, "age_group", [1], [0]),
        },
    }

    # Mitigation: compute reweighed instance weights (Reweighing) using all protected attributes.
    rw = Reweighing(
        unprivileged_groups=[{gender_attr: gender_unpriv_code, race_attr: race_unpriv_code, "age_group": 0}],
        privileged_groups=[{gender_attr: gender_priv_code, race_attr: race_priv_code, "age_group": 1}],
    )
    dataset_rw = rw.fit_transform(dataset)

    bias_after = {
        "gender": {
            "attr": gender_attr,
            "privileged_value": gender_priv_code,
            "unprivileged_value": gender_unpriv_code,
            "raw_privileged_value": None if gender_priv_raw is None else str(gender_priv_raw),
            "raw_unprivileged_value": None if gender_unpriv_raw is None else str(gender_unpriv_raw),
            "metrics": metric_for(dataset_rw, gender_attr, [gender_priv_code], [gender_unpriv_code]),
        },
        "race": {
            "attr": race_attr,
            "privileged_value": race_priv_code,
            "unprivileged_value": race_unpriv_code,
            "raw_privileged_value": None if race_priv_raw is None else str(race_priv_raw),
            "raw_unprivileged_value": None if race_unpriv_raw is None else str(race_unpriv_raw),
            "metrics": metric_for(dataset_rw, race_attr, [race_priv_code], [race_unpriv_code]),
        },
        "age": {
            "attr": "age_group",
            "privileged_value": 1,
            "unprivileged_value": 0,
            "threshold": age_privileged_threshold,
            "metrics": metric_for(dataset_rw, "age_group", [1], [0]),
        },
    }

    # 3) Export bias scores and mitigated weights to results.json
    results: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": input_csv,
        "label_column": label_col,
        "encoded_columns": ["gender_enc", "race_enc", "education_enc"],
        "categories": {
            "gender_enc": gender_categories,
            "race_enc": race_categories,
            "education_enc": education_categories,
        },
        "protected_attributes": [gender_attr, race_attr, "age_group"],
        "privileged_groups_used": {
            gender_attr: gender_priv_code,
            race_attr: race_priv_code,
            "age_group": 1,
        },
        "unprivileged_groups_used": {
            gender_attr: gender_unpriv_code,
            race_attr: race_unpriv_code,
            "age_group": 0,
        },
        "bias_scores": {
            "before": bias_before,
            "after_reweighing": bias_after,
        },
        "mitigated_weights": {
            "method": "aif360.preprocessing.Reweighing",
            "weights": [float(w) for w in dataset_rw.instance_weights.tolist()],
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote bias metrics + mitigated weights to {output_json}")


if __name__ == "__main__":
    main()

