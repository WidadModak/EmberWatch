from pathlib import Path
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/raw/merged_fire_events.csv")
BASE_OUTPUT_PATH = Path("data/processed/region_month_base.csv")
VARIANTS_DIR = Path("data/variants")

VARIANT_SPECS = {
    "risk_total_burned_area": {
        "source_col": "total_burned_area",
        "output_path": VARIANTS_DIR / "risk_total_burned_area.csv",
        "family": "burden_based",
        "description": "Total burned area in a region-month",
    },
    "risk_fire_count": {
        "source_col": "fire_count",
        "output_path": VARIANTS_DIR / "risk_fire_count.csv",
        "family": "frequency_based",
        "description": "Number of fires in a region-month",
    },
    "risk_average_fire_size": {
        "source_col": "average_fire_size",
        "output_path": VARIANTS_DIR / "risk_average_fire_size.csv",
        "family": "burden_based",
        "description": "Mean fire size within a region-month",
    },
    "risk_worst_case_fire_size": {
        "source_col": "worst_case_fire_size",
        "output_path": VARIANTS_DIR / "risk_worst_case_fire_size.csv",
        "family": "burden_based",
        "description": "Largest single fire within a region-month",
    },
}

RISK_LABELS = ["Low", "Moderate", "High", "Extreme"]


# -----------------------------
# Core processing
# -----------------------------
def build_region_month_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per region-month-season with aggregated fire outcomes
    and aggregated environmental predictors.
    """
    df = df.copy()

    if "RISK_LABEL" in df.columns:
        df = df.drop(columns=["RISK_LABEL"])

    region_month = (
        df.groupby(["Region", "Year", "Month", "Season"], as_index=False)
        .agg(
            total_burned_area=("SIZE_HA", "sum"),
            fire_count=("FIRE_NO", "count"),
            average_fire_size=("SIZE_HA", "mean"),
            worst_case_fire_size=("SIZE_HA", "max"),
            temperature_c=("Temperature (°C)", "mean"),
            humidity_pct=("Humidity (%)", "mean"),
            rainfall_mm=("Rainfall (mm)", "mean"),
            wind_speed_kmh=("Wind Speed (km/h)", "mean"),
        )
        .sort_values(["Region", "Year", "Month"])
        .reset_index(drop=True)
    )

    return region_month


def assign_quantile_risk(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    labels: list[str] = RISK_LABELS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create quartile-based risk labels from a numeric source column.
    """
    out = df.copy()

    out[target_col], bin_edges = pd.qcut(
        out[source_col],
        q=4,
        labels=labels,
        retbins=True,
        duplicates="drop",
    )

    actual_bins = out[target_col].nunique(dropna=True)
    if actual_bins != 4:
        raise ValueError(
            f"Expected 4 bins for '{source_col}', but formed {actual_bins}. "
            "This usually means too many repeated values for qcut."
        )

    return out, pd.Series(bin_edges, name=f"{source_col}_cutoffs")


def save_base_dataset(region_month: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    region_month.to_csv(output_path, index=False)
    print(f"\nSaved base region-month dataset to: {output_path}")


def save_target_variant(
    base_df: pd.DataFrame,
    target_name: str,
    spec: dict,
) -> None:
    """
    Create and save one target-specific dataset.
    """
    variant_df, cutoffs = assign_quantile_risk(
        df=base_df,
        source_col=spec["source_col"],
        target_col=target_name,
    )

    output_path = spec["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variant_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"Target: {target_name}")
    print(f"Family: {spec['family']}")
    print(f"Definition: {spec['description']}")
    print(f"Source column: {spec['source_col']}")
    print("\nQuantile cutoffs:")
    print(cutoffs)
    print("\nClass distribution:")
    print(variant_df[target_name].value_counts().sort_index())
    print(f"\nSaved variant to: {output_path}")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)
    print("\nOriginal columns:")
    print(df.columns.tolist())

    region_month = build_region_month_base(df)

    print("\nRegion-month shape:", region_month.shape)
    print("\nRegion-month columns:")
    print(region_month.columns.tolist())

    print("\nHead of region-month dataset:")
    print(region_month.head())

    print("\nMissing values in region-month dataset:")
    print(region_month.isnull().sum())

    save_base_dataset(region_month, BASE_OUTPUT_PATH)

    for target_name, spec in VARIANT_SPECS.items():
        save_target_variant(region_month, target_name, spec)


if __name__ == "__main__":
    main()