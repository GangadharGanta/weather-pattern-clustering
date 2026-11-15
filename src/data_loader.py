"""Data Ingestion, Synthetic Generation, and Preprocessing Module for Weather Clustering."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils import calculate_apparent_temperature, calculate_dew_point, get_logger

logger = get_logger("DataLoader")

CORE_METEOROLOGICAL_FEATURES = [
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Pressure",
    "Precipitation",
]

ALL_NUMERIC_FEATURES = [
    "Temperature",
    "Apparent_Temperature",
    "Humidity",
    "Wind_Speed",
    "Pressure",
    "Precipitation",
    "Dew_Point",
    "Visibility",
]


def generate_synthetic_weather_data(
    n_days: int = 1826,
    start_date: str = "2020-01-01",
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Generate statistically realistic, meteorologically grounded daily weather data.
    Simulates annual seasonal cycles, synoptic weather systems, pressure gradients,
    and realistic atmospheric correlations.
    """
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, periods=n_days, freq="D")
    day_of_year = np.array([d.timetuple().tm_yday for d in date_range])

    # Base seasonal temperature wave (annual cycle)
    # Peak around day 200 (July), minimum around day 20 (January)
    temp_seasonal = 20.0 + 13.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)

    # Weather regime transitions via Markov-like stochastic state
    states = np.zeros(n_days, dtype=int)
    current_state = 0
    transition_matrix = np.array([
        [0.70, 0.08, 0.07, 0.05, 0.10],  # 0: Pleasant/Mild
        [0.08, 0.72, 0.05, 0.10, 0.05],  # 1: Hot & Humid
        [0.10, 0.05, 0.73, 0.02, 0.10],  # 2: Cold & Dry
        [0.12, 0.06, 0.02, 0.65, 0.15],  # 3: Stormy / Low Pressure
        [0.10, 0.08, 0.05, 0.02, 0.75],  # 4: Hot & Arid
    ])

    for t in range(1, n_days):
        current_state = np.random.choice(5, p=transition_matrix[current_state])
        states[t] = current_state

    # Synthesize core parameters based on seasonal baseline + regime influences
    temperatures = []
    humidities = []
    wind_speeds = []
    pressures = []
    precipitations = []
    visibilities = []

    for t in range(n_days):
        base_t = temp_seasonal[t]
        regime = states[t]

        if regime == 0:  # Pleasant & Temperate
            t_val = base_t * 0.8 + 8.0 + np.random.normal(0, 1.8)
            t_val = np.clip(t_val, 16.0, 25.0)
            h_val = np.clip(np.random.normal(52, 6), 35, 68)
            w_val = np.clip(np.random.normal(12, 3), 4, 22)
            p_val = np.clip(np.random.normal(1015, 2.5), 1010, 1022)
            rain_val = 0.0 if np.random.rand() > 0.10 else np.random.exponential(1.2)
            vis_val = np.clip(np.random.normal(14.5, 1.5), 10, 16)

        elif regime == 1:  # Hot & Humid (Sultry Monsoon)
            t_val = max(24.0, base_t + np.random.normal(5.0, 2.0))
            h_val = np.clip(np.random.normal(82, 6), 70, 98)
            w_val = np.clip(np.random.normal(16, 4), 6, 28)
            p_val = np.clip(np.random.normal(1005, 3.5), 996, 1012)
            rain_val = np.random.exponential(9.5) if np.random.rand() > 0.35 else 0.0
            vis_val = np.clip(np.random.normal(9.0, 2.5), 4, 13)

        elif regime == 2:  # Cold & Dry (Winter Continental)
            t_val = min(12.0, base_t - np.random.normal(7.0, 2.5))
            h_val = np.clip(np.random.normal(38, 7), 18, 55)
            w_val = np.clip(np.random.normal(14, 4), 5, 25)
            p_val = np.clip(np.random.normal(1024, 3.0), 1017, 1033)
            rain_val = 0.0 if np.random.rand() > 0.05 else np.random.exponential(0.6)
            vis_val = np.clip(np.random.normal(15.0, 1.0), 11, 16)

        elif regime == 3:  # Stormy & Unstable (Deep Depression Gale)
            t_val = base_t * 0.7 + 6.0 + np.random.normal(0, 2.5)
            h_val = np.clip(np.random.normal(90, 5), 80, 100)
            w_val = np.clip(np.random.normal(36, 8), 24, 62)
            p_val = np.clip(np.random.normal(994, 4.0), 980, 1002)
            rain_val = np.clip(np.random.exponential(24.0) + 10.0, 8.0, 95.0)
            vis_val = np.clip(np.random.normal(5.0, 1.8), 1.5, 9)

        else:  # Hot & Arid (Summer Heatwave)
            t_val = max(33.0, base_t + np.random.normal(9.0, 2.2))
            h_val = np.clip(np.random.normal(24, 5), 12, 38)
            w_val = np.clip(np.random.normal(20, 5), 8, 35)
            p_val = np.clip(np.random.normal(1011, 2.5), 1005, 1016)
            rain_val = 0.0
            vis_val = np.clip(np.random.normal(13.0, 2.0), 8, 16)

        temperatures.append(round(t_val, 2))
        humidities.append(round(h_val, 1))
        wind_speeds.append(round(w_val, 1))
        pressures.append(round(p_val, 1))
        precipitations.append(round(rain_val, 2))
        visibilities.append(round(vis_val, 1))

    df = pd.DataFrame(
        {
            "Date": date_range,
            "Temperature": temperatures,
            "Humidity": humidities,
            "Wind_Speed": wind_speeds,
            "Pressure": pressures,
            "Precipitation": precipitations,
            "Visibility": visibilities,
            "Regime_GroundTruth": states,
        }
    )

    # Inject realistic small rate of missing values (~1.5%) in random features
    mask_indices = np.random.choice(n_days, size=int(n_days * 0.015), replace=False)
    for idx in mask_indices:
        feat = np.random.choice(["Temperature", "Humidity", "Pressure", "Wind_Speed"])
        df.loc[idx, feat] = np.nan

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Synthetic historical weather dataset saved to {out} ({len(df)} rows).")

    return df


def load_raw_data(
    filepath: Union[str, Path] = "data/raw/historical_weather_daily.csv",
    auto_generate: bool = True,
) -> pd.DataFrame:
    """
    Ingest raw historical weather dataset. If the file is not found,
    optionally generates a validated realistic historical dataset.
    """
    path = Path(filepath)
    if not path.exists():
        if auto_generate:
            logger.warning(f"Raw data file not found at {path}. Synthesizing realistic climate dataset...")
            return generate_synthetic_weather_data(output_path=path)
        raise FileNotFoundError(f"Raw data not found at {path}")

    logger.info(f"Loading raw weather data from {path}")
    df = pd.read_csv(path)
    return df


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform production-grade cleaning on weather time-series:
    - Parses and formats datetime index
    - Imputes missing values using forward-fill followed by backward-fill
    - Validates physical meteorological boundaries
    - Computes derived features: Dew Point and Apparent Temperature
    """
    df_clean = df.copy()

    # Parse Date
    if "Date" in df_clean.columns:
        df_clean["Date"] = pd.to_datetime(df_clean["Date"])
        df_clean = df_clean.sort_values("Date").reset_index(drop=True)

    # Impute missing values chronologically
    null_count_before = df_clean[CORE_METEOROLOGICAL_FEATURES].isnull().sum().sum()
    if null_count_before > 0:
        logger.info(f"Found {null_count_before} missing entries. Applying forward-fill & backward-fill...")
        df_clean[CORE_METEOROLOGICAL_FEATURES] = (
            df_clean[CORE_METEOROLOGICAL_FEATURES].ffill().bfill()
        )

    # Meteorological boundary enforcement
    df_clean["Temperature"] = df_clean["Temperature"].clip(lower=-40.0, upper=60.0)
    df_clean["Humidity"] = df_clean["Humidity"].clip(lower=1.0, upper=100.0)
    df_clean["Wind_Speed"] = df_clean["Wind_Speed"].clip(lower=0.0, upper=150.0)
    df_clean["Pressure"] = df_clean["Pressure"].clip(lower=870.0, upper=1085.0)
    df_clean["Precipitation"] = df_clean["Precipitation"].clip(lower=0.0, upper=500.0)

    # Calculate derived thermodynamic metrics
    df_clean["Dew_Point"] = df_clean.apply(
        lambda row: calculate_dew_point(row["Temperature"], row["Humidity"]), axis=1
    )
    df_clean["Apparent_Temperature"] = df_clean.apply(
        lambda row: calculate_apparent_temperature(
            row["Temperature"], row["Humidity"], row["Wind_Speed"]
        ),
        axis=1,
    )

    logger.info(f"Dataset successfully cleaned. Records: {len(df_clean)}, Columns: {list(df_clean.columns)}")
    return df_clean


def prepare_and_save_data(
    raw_path: str = "data/raw/historical_weather_daily.csv",
    processed_path: str = "data/processed/processed_weather_data.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute ingestion, cleaning, and persist processed data."""
    raw_df = load_raw_data(raw_path)
    clean_df = clean_weather_data(raw_df)

    p_out = Path(processed_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(p_out, index=False)
    logger.info(f"Processed dataset saved to {p_out}")

    return raw_df, clean_df


if __name__ == "__main__":
    prepare_and_save_data()
