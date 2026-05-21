from __future__ import annotations

import logging
import os
import urllib.request
import zipfile

import mydatasets
import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import scipy.stats as sp_stats
import sklearn.compose as skl_compose
import sklearn.preprocessing as skl_preproc
import tensordict as thd
import torch as th

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)


# NOTE Define paths
dataset_dir = os.path.join(
    mydatasets.common.get_datasets_files_root_dir(), "uci", "bar-crawl"
)
data_zip_path = os.path.join(dataset_dir, "data.zip")
data_dir = os.path.join(dataset_dir, "data")
data_url = "https://archive.ics.uci.edu/static/public/515/bar+crawl+detecting+heavy+drinking.zip"
accel_path = os.path.join(data_dir, "all_accelerometer_data_pids_13.csv")

# NOTE Following the study setting (Killian et al., 2019):
#   "The study decomposed each time series into 10 second windows and
#    performed binary classification to predict if windows corresponded
#    to an intoxicated participant (TAC >= 0.08) or sober participant
#    (TAC < 0.08)."
#
# From Table 2 of the paper, features per 10-second window include:
#   - Time-domain: mean, std, median, zero crossing rate, max, min,
#     max_abs, min_abs, skewness, kurtosis, RMS  (per axis)
#   - Frequency-domain: spectral entropy, spectral centroid,
#     spectral spread, spectral flux, spectral rolloff, max frequency,
#     average power, spectral peak ratio  (per axis)
#
# We implement the core features from Table 2 for each of the 3 axes.

WINDOW_SIZE_MS = 10_000  # 10 seconds in milliseconds
TAC_THRESHOLD = 0.08  # legal limit in g/dl
SAMPLING_RATE = 40  # Hz


def _zero_crossing_rate(signal: np.ndarray) -> float:
    """Number of times the signal changes sign, normalized by length."""
    if len(signal) < 2:
        return 0.0
    return float(np.sum(np.diff(np.sign(signal)) != 0)) / len(signal)


def _spectral_features(signal: np.ndarray, fs: float) -> dict[str, float]:
    """Compute frequency-domain features from a signal using Welch's method."""
    if len(signal) < 4:
        return {
            "spectral_entropy": 0.0,
            "spectral_centroid": 0.0,
            "spectral_spread": 0.0,
            "spectral_rolloff": 0.0,
            "max_frequency": 0.0,
            "average_power": 0.0,
            "spectral_peak_ratio": 0.0,
        }

    freqs, psd = sp_signal.welch(signal, fs=fs, nperseg=min(len(signal), 256))

    # Avoid division by zero
    total_power = np.sum(psd)
    if total_power == 0:
        return {
            "spectral_entropy": 0.0,
            "spectral_centroid": 0.0,
            "spectral_spread": 0.0,
            "spectral_rolloff": 0.0,
            "max_frequency": 0.0,
            "average_power": 0.0,
            "spectral_peak_ratio": 0.0,
        }

    # Normalized PSD (probability distribution)
    psd_norm = psd / total_power

    # Spectral entropy
    spectral_entropy = float(sp_stats.entropy(psd_norm + 1e-12))

    # Spectral centroid (weighted mean of frequencies)
    spectral_centroid = float(np.sum(freqs * psd_norm))

    # Spectral spread (variance about centroid)
    spectral_spread = float(
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
    )

    # Spectral rolloff (frequency below which 90% of energy is contained)
    cumulative_energy = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumulative_energy, 0.90 * total_power)
    spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Max frequency (frequency with highest power)
    max_frequency = float(freqs[np.argmax(psd)])

    # Average power
    average_power = float(total_power / len(psd))

    # Spectral peak ratio (ratio of largest to second-largest peak)
    sorted_psd = np.sort(psd)[::-1]
    if len(sorted_psd) >= 2 and sorted_psd[1] > 0:
        spectral_peak_ratio = float(sorted_psd[0] / sorted_psd[1])
    else:
        spectral_peak_ratio = 0.0

    return {
        "spectral_entropy": spectral_entropy,
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "spectral_rolloff": spectral_rolloff,
        "max_frequency": max_frequency,
        "average_power": average_power,
        "spectral_peak_ratio": spectral_peak_ratio,
    }


def compute_window_features(window_data: pd.DataFrame) -> dict[str, float]:
    """Compute features for a single 10-second window of accelerometer data.

    Features follow Table 2 of Killian et al. (2019):
      - Time-domain per axis (x, y, z): mean, std, median, zero_crossing_rate,
        max, min, max_abs, min_abs, skewness, kurtosis, RMS
      - Frequency-domain per axis: spectral_entropy, spectral_centroid,
        spectral_spread, spectral_rolloff, max_frequency, average_power,
        spectral_peak_ratio
    """
    features: dict[str, float] = {}

    for axis in ["x", "y", "z"]:
        vals = window_data[axis].to_numpy()

        # Time-domain features
        features[f"{axis}_mean"] = float(np.mean(vals))
        features[f"{axis}_std"] = float(np.std(vals))
        features[f"{axis}_median"] = float(np.median(vals))
        features[f"{axis}_zcr"] = _zero_crossing_rate(vals)
        features[f"{axis}_max"] = float(np.max(vals))
        features[f"{axis}_min"] = float(np.min(vals))
        features[f"{axis}_max_abs"] = float(np.max(np.abs(vals)))
        features[f"{axis}_min_abs"] = float(np.min(np.abs(vals)))
        # NOTE: skewness and kurtosis return NaN for constant signals
        # (zero variance). A constant signal has no skew or excess
        # kurtosis, so we replace NaN with 0.0.
        _skew = float(sp_stats.skew(vals))
        features[f"{axis}_skewness"] = 0.0 if np.isnan(_skew) else _skew
        _kurt = float(sp_stats.kurtosis(vals))
        features[f"{axis}_kurtosis"] = 0.0 if np.isnan(_kurt) else _kurt
        features[f"{axis}_rms"] = float(np.sqrt(np.mean(vals**2)))

        # Frequency-domain features
        spec_feats = _spectral_features(vals, SAMPLING_RATE)
        for feat_name, feat_val in spec_feats.items():
            features[f"{axis}_{feat_name}"] = feat_val

    return features


def main() -> None:
    logger.info("Preparing Bar Crawl dataset")
    if not os.path.exists(data_zip_path):
        os.makedirs(dataset_dir, exist_ok=True)
        urllib.request.urlretrieve(data_url, data_zip_path)

    if not os.path.exists(accel_path):
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(data_zip_path) as archive:
            archive.extractall(data_dir)

        nested_zip_path = os.path.join(data_dir, "data.zip")
        if os.path.exists(nested_zip_path):
            with zipfile.ZipFile(nested_zip_path) as archive:
                archive.extractall(data_dir)

    # NOTE Load main accelerometer data (14M rows)
    # Columns: time (unix ms), pid, x, y, z
    accel_df = pd.read_csv(
        accel_path,
        dtype={
            "time": np.int64,
            "pid": str,
            "x": np.float64,
            "y": np.float64,
            "z": np.float64,
        },
    )

    # NOTE Load phone types
    phone_df = pd.read_csv(os.path.join(data_dir, "phone_types.csv"))

    # NOTE Load all clean TAC files and combine
    tac_dir = os.path.join(data_dir, "clean_tac")
    tac_dfs = []

    for filename in sorted(os.listdir(tac_dir)):
        if filename.endswith("_clean_TAC.csv"):
            pid = filename.replace("_clean_TAC.csv", "")
            tac_data = pd.read_csv(
                os.path.join(tac_dir, filename),
                dtype={"timestamp": np.int64, "TAC_Reading": np.float64},
            )
            tac_data["pid"] = pid
            # Convert timestamp from seconds to milliseconds to match accelerometer
            tac_data["time"] = tac_data["timestamp"] * 1000
            tac_data = tac_data[["time", "pid", "TAC_Reading"]]
            tac_dfs.append(tac_data)

    tac_df = pd.concat(tac_dfs, ignore_index=True)

    window_records = []

    for pid in sorted(accel_df["pid"].unique()):
        # -- subset and sort by time
        accel_pid = (
            accel_df[accel_df["pid"] == pid].sort_values("time").reset_index(drop=True)
        )
        tac_pid = (
            tac_df[tac_df["pid"] == pid].sort_values("time").reset_index(drop=True)
        )

        # -- merge TAC onto accelerometer (backward: most recent TAC reading)
        merged_pid = pd.merge_asof(
            accel_pid,
            tac_pid[["time", "TAC_Reading"]],
            on="time",
            direction="backward",
        )
        # Fill NaN TAC (before first TAC measurement) with 0.0 (sober)
        merged_pid["TAC_Reading"] = merged_pid["TAC_Reading"].fillna(0.0)

        # -- assign each row to a 10-second window
        t0 = merged_pid["time"].iloc[0]
        merged_pid["window"] = (merged_pid["time"] - t0) // WINDOW_SIZE_MS

        # -- compute features per window
        for window_id, window_data in merged_pid.groupby("window"):
            if len(window_data) < 2:
                continue  # skip windows with too few samples

            features = compute_window_features(window_data)
            features["TAC_Reading"] = float(window_data["TAC_Reading"].mean())
            features["pid"] = pid
            features["n_samples"] = len(window_data)
            window_records.append(features)

    windows_df = pd.DataFrame(window_records)

    # NOTE Merge phone type onto window-level data
    windows_df = windows_df.merge(phone_df, on="pid", how="left")

    # NOTE Define feature and target columns
    # Time-domain features per axis (11 each × 3 axes = 33)
    # Frequency-domain features per axis (7 each × 3 axes = 21)
    # Total: 54 accelerometer features + 1 phonetype = 55 features
    _axes = ["x", "y", "z"]
    _time_domain_suffixes = [
        "mean",
        "std",
        "median",
        "zcr",
        "max",
        "min",
        "max_abs",
        "min_abs",
        "skewness",
        "kurtosis",
        "rms",
    ]
    _freq_domain_suffixes = [
        "spectral_entropy",
        "spectral_centroid",
        "spectral_spread",
        "spectral_rolloff",
        "max_frequency",
        "average_power",
        "spectral_peak_ratio",
    ]

    _accel_features = []
    for axis in _axes:
        for suffix in _time_domain_suffixes + _freq_domain_suffixes:
            _accel_features.append(f"{axis}_{suffix}")

    feature_names = _accel_features + ["phonetype"]
    target_names = ["heavy_drinking"]

    # NOTE Create binary target: 1 if TAC >= 0.08, else 0
    windows_df["heavy_drinking"] = (windows_df["TAC_Reading"] >= TAC_THRESHOLD).astype(
        np.int64
    )

    # NOTE Prepare feature and target DataFrames
    xs_df: pd.DataFrame = windows_df[feature_names].copy()
    ys_df: pd.DataFrame = windows_df[target_names].copy()

    # NOTE Classify feature types for preprocessing
    _numeric_features = _accel_features
    _categorical_features = ["phonetype"]

    # NOTE Create feature preprocessor using sklearn transformers
    _MISSING_SENTINEL: int = -10

    _ordinal_kwargs = dict(
        handle_unknown="use_encoded_value",
        unknown_value=_MISSING_SENTINEL,
        encoded_missing_value=_MISSING_SENTINEL,
        dtype=np.int64,
    )

    feature_preprocessor = skl_compose.ColumnTransformer(
        transformers=[
            (
                "numeric_passthrough",
                "passthrough",
                _numeric_features,
            ),
            (
                "phonetype_ordinal",
                skl_preproc.OrdinalEncoder(
                    categories=[["Android", "iPhone"]], **_ordinal_kwargs
                ),
                _categorical_features,
            ),
        ],
        remainder="drop",
    )

    # NOTE ensure feature names are preserved in output
    feature_preprocessor.set_output(transform="pandas")
    patient_xs: pd.DataFrame = feature_preprocessor.fit_transform(xs_df)
    # NOTE strip transformer prefix from column names
    patient_xs = patient_xs.rename(columns=lambda x: str.split(x, "__", 1)[-1])

    # NOTE Encode target: heavy_drinking -> ordinal {0: sober, 1: intoxicated}
    target_preprocessor = skl_preproc.OrdinalEncoder(
        categories=[[0, 1]], dtype=np.int64
    )
    target_preprocessor.set_output(transform="pandas")
    patient_ys: pd.DataFrame = target_preprocessor.fit_transform(ys_df[target_names])

    data_thd: thd.TensorDict = thd.make_tensordict(
        {
            "xs": th.as_tensor(patient_xs.to_numpy(), dtype=th.float32),
            "ys": th.as_tensor(patient_ys.to_numpy(), dtype=th.int64),
        },
        auto_batch_size=True,
    )
    raw_data_d = {
        "feature_preprocessor": feature_preprocessor,
        "target_preprocessor": target_preprocessor,
        "patient_xs_df": patient_xs,
        "patient_ys_df": patient_ys,
    }

    os.makedirs(
        os.path.join(mydatasets.common.get_datasets_files_root_dir(), "uci"),
        exist_ok=True,
    )
    output_path_tensor = os.path.join(
        mydatasets.common.get_datasets_files_root_dir(),
        "uci",
        "bar-crawl.pt",
    )
    output_path_raw = os.path.join(
        mydatasets.common.get_datasets_files_root_dir(),
        "uci",
        "raw-bar-crawl.pkl",
    )
    th.save(data_thd, output_path_tensor)
    th.save(raw_data_d, output_path_raw)

    logger.info("Bar Crawl xs shape: %s", tuple(data_thd["xs"].shape))
    logger.info("Bar Crawl ys shape: %s", tuple(data_thd["ys"].shape))
    logger.info("Exported Bar Crawl TensorDict to %s", output_path_tensor)
    logger.info("Exported Bar Crawl metadata to %s", output_path_raw)


if __name__ == "__main__":
    main()
