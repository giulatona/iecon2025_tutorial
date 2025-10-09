"""
Time Series Preprocessing Module for Energy Consumption Forecasting

Comprehensive preprocessing pipeline for household power consumption data designed
for deep learning forecasting models using TensorFlow/Keras.

Main Functions:
- load_and_preprocess_data(): Complete preprocessing pipeline
- clean_data(): Data cleaning and datetime handling
- normalize_features(): Feature scaling with inverse transform support
- create_sliding_windows(): TensorFlow dataset creation for time series

Features:
- Missing value imputation using seasonal patterns
- Configurable downsampling and feature engineering
- Chronological train/validation/test splitting
- Optimized sliding window creation for neural networks

Example:
    >>> results = load_and_preprocess_data('data/power_consumption.txt')
    >>> train_ds = results['datasets']['train']
    >>> scaler = results['scaler_params']

Author: Giuseppe La Tona
"""

from typing import Any, Optional, Union

import holidays
import keras
import numpy as np
import pandas as pd
import tensorflow as tf


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the household power consumption dataset.

    Parameters:
    df (pd.DataFrame): Raw dataset

    Returns:
    pd.DataFrame: Cleaned dataset with datetime index
    """
    df_clean = df.copy()

    # Combine Date and Time columns
    df_clean['DateTime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Time'],
                                          format='%d/%m/%Y %H:%M:%S')

    # Replace '?' with NaN and convert to numeric
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Set DateTime as index
    df_clean.set_index('DateTime', inplace=True)

    # Drop original Date and Time columns
    df_clean.drop(['Date', 'Time'], axis=1, inplace=True)

    return df_clean


def downsample_time_series(df: pd.DataFrame,
                           freq: str = '15T',
                           aggregation_method: str = 'mean') -> pd.DataFrame:
    """
    Downsample time series data to a lower frequency.

    Parameters:
    df (pd.DataFrame): Input dataframe with datetime index
    freq (str): Target frequency (e.g., '15T' for 15 minutes, '1H' for 1 hour)
    aggregation_method (str): Method to aggregate values ('mean', 'sum', 'max', 'min')

    Returns:
    pd.DataFrame: Downsampled dataframe
    dict: Downsampling information
    """

    # Apply downsampling based on aggregation method
    if aggregation_method == 'mean':
        df_downsampled = df.resample(freq).mean()
    elif aggregation_method == 'sum':
        df_downsampled = df.resample(freq).sum()
    elif aggregation_method == 'max':
        df_downsampled = df.resample(freq).max()
    elif aggregation_method == 'min':
        df_downsampled = df.resample(freq).min()
    else:
        raise ValueError(
            f"Unsupported aggregation method: {aggregation_method}")

    # Remove rows with all NaN values (can occur at the boundaries)
    df_downsampled = df_downsampled.dropna(how='all')

    return df_downsampled


def fill_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in time series data using seasonal patterns.

    This function identifies NaN intervals and fills them using:
    1. Previous years' data at the same time period (preferred)
    2. Previous week's data if yearly data is not available

    Parameters:
    dataset (pd.DataFrame): Input dataframe with datetime index and missing values

    Returns:
    pd.DataFrame: Dataframe with filled missing values
    """
    nan_intervals = []
    during_interval = False
    interval_start_index = None
    for dt, val in dataset.Global_active_power.items():
        if during_interval:
            if not np.isnan(val):
                timedelta = dt - interval_start_index
                if (timedelta) > pd.Timedelta(1, 'm'):
                    nan_intervals.append((interval_start_index, dt, timedelta))
                during_interval = False
        elif np.isnan(val):
            interval_start_index = dt
            during_interval = True

    def get_previous_years(timestamp, index):
        timestamp_list = []
        year_count = 1
        while (timestamp - pd.Timedelta(year_count * 365, 'd')) in index:
            timestamp_list.append(
                timestamp - pd.Timedelta(year_count * 365, 'd'))
            year_count = year_count + 1
        return timestamp_list

    def get_previous_week(timestamp, index):
        if (timestamp - pd.Timedelta(7, 'd')) in index:
            return timestamp - pd.Timedelta(7, 'd')
        else:
            return None

    nan_intervals = pd.DataFrame(nan_intervals, columns=[
                                 'start', 'stop', 'duration'])

    new_df = dataset.copy()
    for start, stop, _ in nan_intervals.itertuples(index=False):
        start_previous = get_previous_years(start, dataset.index)
        if start_previous:
            stop_previous = get_previous_years(stop, dataset.index)
            values = np.array([dataset.loc[t1: t2, :]
                              for t1, t2 in zip(start_previous, stop_previous)])
            new_df.loc[start: stop, :] = np.mean(values, axis=0)
        else:
            start_previous = get_previous_week(start, dataset.index)
            if start_previous:
                stop_previous = get_previous_week(stop, dataset.index)
                new_df.loc[start: stop,
                           :] = dataset.loc[start_previous: stop_previous, :].values

    return new_df


def normalize_features(df: pd.DataFrame,
                       method: str = 'minmax',
                       columns: Optional[list[str]] = None,
                       scaler_params: Optional[dict[str,
                                                    dict[str, Any]]] = None
                       ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Normalize features using standardization or min-max scaling.

    Parameters:
    df (pd.DataFrame): Input dataframe
    method (str): 'standardize' or 'minmax'
    columns (list): Columns to normalize (None for all numeric)
    scaler_params (dict): Pre-computed scaler parameters (for transform mode)

    Returns:
    pd.DataFrame: Normalized dataframe
    dict: Scaler parameters for inverse transformation
    """
    df_norm = df.copy()

    # Determine columns to normalize
    if columns is None:
        # Select only numeric columns
        numeric_columns = df_norm.select_dtypes(
            include=[np.number]).columns.tolist()
        # Exclude datetime-related columns that shouldn't be normalized
        exclude_patterns = ['year', '_sin', '_cos', 'is_',
                            'days_', 'holiday_cluster', 'holidays_in_week']
        columns = [col for col in numeric_columns
                   if not any(pattern in col.lower() for pattern in exclude_patterns)]

    # Initialize scaler parameters if not provided
    if scaler_params is None:
        scaler_params = {}
        fit_mode = True
    else:
        fit_mode = False

    for col in columns:
        if col not in df_norm.columns:
            continue

        if method == 'standardize':
            if fit_mode:
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                scaler_params[col] = {'mean': mean_val,
                                      'std': std_val, 'method': 'standardize'}
            else:
                mean_val = scaler_params[col]['mean']
                std_val = scaler_params[col]['std']

            # Z-score normalization: (x - mean) / std
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
            else:
                df_norm[col] = 0  # Handle constant columns

        elif method == 'minmax':
            if fit_mode:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                scaler_params[col] = {'min': min_val,
                                      'max': max_val, 'method': 'minmax'}
            else:
                min_val = scaler_params[col]['min']
                max_val = scaler_params[col]['max']

            # Min-max normalization to [-1, 1]: 2 * (x - min) / (max - min) - 1
            range_val = max_val - min_val
            if range_val > 0:
                df_norm[col] = 2 * (df_norm[col] - min_val) / range_val - 1
            else:
                df_norm[col] = 0  # Handle constant columns

    return df_norm, scaler_params


def inverse_normalize_features(df_norm: Union[pd.DataFrame, pd.Series],
                               scaler_params: dict[str, dict[str, Any]],
                               columns: Optional[list[str]] = None
                               ) -> Union[pd.DataFrame, pd.Series]:
    """
    Inverse transform normalized features back to original scale.

    This function can handle cases where df_norm contains only a subset of the original
    columns, making it suitable for inverse transforming forecasting results.

    Parameters:
    df_norm (pd.DataFrame or pd.Series): Normalized dataframe/series or target predictions
    scaler_params (dict): Scaler parameters from normalize_features
    columns (list): Specific columns to inverse transform (None for all available)

    Returns:
    pd.DataFrame or pd.Series: Features in original scale (same type as input)
    """
    # Handle Series input (common for single target predictions)
    if isinstance(df_norm, pd.Series):
        series_name = df_norm.name
        if series_name is None:
            # If series has no name, try to infer from columns parameter
            if columns and len(columns) == 1:
                series_name = columns[0]
            else:
                raise ValueError(
                    "Series input must have a name or specify columns parameter with single column")

        # Check if we have scaler parameters for this series
        if series_name not in scaler_params:
            raise ValueError(
                f"No scaler parameters found for column '{series_name}'")

        params = scaler_params[series_name]
        method = params['method']

        if method == 'standardize':
            # Inverse z-score: x * std + mean
            result = df_norm * params['std'] + params['mean']
        elif method == 'minmax':
            # Inverse min-max from [-1, 1]: (x + 1) * (max - min) / 2 + min
            range_val = params['max'] - params['min']
            result = (df_norm + 1) * range_val / 2 + params['min']
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return result

    # Handle DataFrame input
    df_orig = df_norm.copy()

    # Determine which columns to process
    if columns is None:
        # Process all columns that exist in both df_norm and scaler_params
        columns_to_process = [
            col for col in df_orig.columns if col in scaler_params]
    else:
        # Process only specified columns that exist in both
        columns_to_process = [
            col for col in columns if col in df_orig.columns and col in scaler_params]

    # Warn about missing columns
    missing_in_data = [col for col in scaler_params.keys()
                       if col not in df_orig.columns]
    missing_in_params = [col for col in df_orig.columns if col in df_orig.select_dtypes(
        include=[np.number]).columns and col not in scaler_params]

    if missing_in_data and columns is None:
        print(
            f"Note: Columns {missing_in_data} have scaler parameters but are not in the data")
    if missing_in_params:
        print(
            f"Warning: Numeric columns {missing_in_params} in data but no scaler parameters found")

    # Apply inverse transformation
    for col in columns_to_process:
        params = scaler_params[col]
        method = params['method']

        if method == 'standardize':
            # Inverse z-score: x * std + mean
            df_orig[col] = df_orig[col] * params['std'] + params['mean']

        elif method == 'minmax':
            # Inverse min-max from [-1, 1]: (x + 1) * (max - min) / 2 + min
            range_val = params['max'] - params['min']
            df_orig[col] = (df_orig[col] + 1) * range_val / 2 + params['min']
        else:
            raise ValueError(
                f"Unknown normalization method for column {col}: {method}")

    return df_orig


def split_time_series_by_ratios(df: pd.DataFrame,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15
                                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Split time series data chronologically using ratio-based approach.

    Parameters:
    df (pd.DataFrame): Input dataframe with datetime index
    train_ratio (float): Proportion for training set (default: 0.7)
    val_ratio (float): Proportion for validation set (default: 0.15)  
    test_ratio (float): Proportion for test set (default: 0.15)

    Returns:
    tuple: (train_df, val_df, test_df) chronologically split dataframes
    dict: Split information with dates and sizes
    """
    # Validate ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Ensure data is sorted by datetime index
    df_sorted = df.sort_index()
    total_samples = len(df_sorted)

    # Calculate split indices
    train_end_idx = int(total_samples * train_ratio)
    val_end_idx = int(total_samples * (train_ratio + val_ratio))

    # Split the data chronologically
    train_df = df_sorted.iloc[:train_end_idx].copy()
    val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
    test_df = df_sorted.iloc[val_end_idx:].copy()

    # Create split information
    split_info = {
        'total_samples': total_samples,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_period': (train_df.index.min(), train_df.index.max()),
        'val_period': (val_df.index.min(), val_df.index.max()),
        'test_period': (test_df.index.min(), test_df.index.max()),
        'ratios_used': (train_ratio, val_ratio, test_ratio)
    }

    return train_df, val_df, test_df, split_info


def add_holiday_features(df: pd.DataFrame,
                         country: str = 'France',
                         include_holiday_names: bool = False) -> pd.DataFrame:
    """
    Add holiday indicators for single-day holidays.

    Simplified version focusing on basic holiday detection and immediate effects.
    This tutorial version demonstrates core concepts without complex multi-day patterns.

    Parameters:
    df (pd.DataFrame): Input dataframe with datetime index
    country (str): Country for holiday calendar
    include_holiday_names (bool): Whether to include holiday_name column for analysis

    Returns:
    pd.DataFrame: Dataframe with holiday features
    """
    df_holidays = df.copy()

    # Get country-specific holidays for all years in the dataset
    years = range(df.index.year.min(), df.index.year.max() + 1)
    country_holidays = holidays.country_holidays(country, years=years)

    # Create date column for holiday lookup (without time component)
    dates = df.index.date

    # Basic holiday indicator
    df_holidays['is_holiday'] = [date in country_holidays for date in dates]

    # Holiday name (for analysis, can be dropped later)
    if include_holiday_names:
        df_holidays['holiday_name'] = [
            country_holidays.get(date, '') for date in dates
        ]

    # Weekend indicator (useful baseline feature)
    df_holidays['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Weekend-holiday interaction
    df_holidays['is_weekend_holiday'] = (
        df_holidays['is_holiday'] &
        (df.index.dayofweek >= 5)  # Saturday or Sunday
    ).astype(int)

    return df_holidays


def create_fourier_features(dataset: pd.DataFrame,
                            num_fourier_terms: int = 3
                            ) -> tuple[pd.DataFrame, list[str]]:
    """
    Create Fourier terms for daily and yearly cycles.

    Fourier features capture periodic patterns more effectively than simple
    cyclical encoding by using multiple harmonics.

    Parameters:
    dataset (pd.DataFrame): Input dataframe with datetime index
    num_fourier_terms (int): Number of Fourier terms to generate

    Returns:
    pd.DataFrame: Dataframe with added Fourier features
    list: List of column names that were added
    """
    dataset = dataset.copy()
    used_columns = []

    # Convert datetime index to timestamp in seconds
    timestamp_s = dataset.index.astype('int64') // 10**9

    # Define time periods in seconds
    day = 24*60*60
    year = (365.2425)*day

    for k in range(1, num_fourier_terms + 1):
        # Daily Fourier terms
        prefix = 'Day'
        col_name = prefix + f" sin{k}"
        dataset[col_name] = np.sin(timestamp_s * (2 * k * np.pi / day))
        used_columns.append(col_name)

        col_name = prefix + f" cos{k}"
        dataset[col_name] = np.cos(timestamp_s * (2 * k * np.pi / day))
        used_columns.append(col_name)

        # Yearly Fourier terms
        prefix = 'Year'
        col_name = prefix + f" sin{k}"
        dataset[col_name] = np.sin(timestamp_s * (2 * k * np.pi / year))
        used_columns.append(col_name)

        col_name = prefix + f" cos{k}"
        dataset[col_name] = np.cos(timestamp_s * (2 * k * np.pi / year))
        used_columns.append(col_name)

    return dataset, used_columns


# Import TensorFlow and Keras utilities for time series

def create_sliding_windows(data: Union[pd.DataFrame, np.ndarray],
                           window_size: int,
                           forecast_horizon: int = 1,
                           step_size: int = 1,
                           batch_size: int = 32,
                           shuffle: bool = False,
                           target_columns: Optional[list[int]] = None) -> tf.data.Dataset:
    """
    Create sliding windows using Keras timeseries utilities.

    This is the recommended approach for Keras/TensorFlow time series workflows.
    Uses keras.utils.timeseries_dataset_from_array for optimal performance.

    Parameters:
    data (array-like): Time series data
    window_size (int): Number of time steps in input window
    forecast_horizon (int): Number of time steps to predict
    step_size (int): Step size between windows (sampling_rate)
    batch_size (int): Batch size for the dataset
    shuffle (bool): Whether to shuffle the windows. IMPORTANT: This shuffles the 
                   windows AFTER they are created as (input, target) pairs, so the 
                   temporal order within each window is preserved. Each neural network 
                   example maintains its correct time sequence, but the order in which 
                   these examples are presented during training is randomized. This is 
                   crucial for training stability while preserving time series structure.
    target_columns (list or None): Column indices to predict. If None, predicts all columns.
                                  For univariate forecasting with exogenous variables,
                                  specify only the target column index (e.g., [0])

    Returns:
    tf.data.Dataset: Keras dataset ready for model training
    """
    # Convert to numpy array if needed
    if hasattr(data, 'values'):
        data = data.values
    data = np.array(data)

    # Ensure data is float32 for TensorFlow
    data = data.astype(np.float32)

    # Handle 1D data (ensure it's 2D)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Create the windowed dataset using Keras utility
    # We'll create the full sequences and split them later
    dataset = keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,  # We'll create targets from the same data
        sequence_length=window_size + forecast_horizon,
        sequence_stride=step_size,
        shuffle=shuffle,
        batch_size=batch_size
    )

    # Split each sequence into input (X) and target (y)
    def split_window(sequence):
        # Input: first 'window_size' steps, all features
        # Shape: (batch, window_size, all_features)
        inputs = sequence[:, :window_size]

        # Target: next 'forecast_horizon' steps
        # Shape: (batch, forecast_horizon, all_features)
        targets_full = sequence[:, window_size:]

        # Apply target column selection if specified
        if target_columns is not None:
            # Use Keras ops for backend-agnostic indexing
            # Shape: (batch, forecast_horizon, len(target_columns))
            targets = keras.ops.take(targets_full, target_columns, axis=-1)
        else:
            targets = targets_full

        return inputs, targets

    # Apply the windowing split
    dataset = dataset.map(split_window, num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize dataset performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_and_preprocess_data(
    data_path: str,
    downsample_freq: str = '15T',
    normalization_method: str = 'minmax',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    window_size: int = 24,
    forecast_horizon: int = 1,
    batch_size: int = 32,
    shuffle_training: bool = True,
    target_columns: Optional[list[int]] = None,
    use_time_features: bool = True,
    use_holiday_features: bool = False,
    country: str = 'France',
    num_fourier_terms: int = 3,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Complete preprocessing pipeline for household power consumption data.

    This function performs the entire preprocessing workflow:
    1. Load and clean the data
    2. Handle missing values
    3. Add time-based features (optional)
    4. Add holiday features (optional)
    5. Downsample to target frequency
    6. Normalize features
    7. Split into train/validation/test sets
    8. Create sliding windows for time series modeling

    Parameters:
    -----------
    data_path : str
        Path to the household power consumption dataset
    downsample_freq : str, default '15T'
        Target frequency for downsampling (e.g., '15T', '1H', '30T')
    normalization_method : str, default 'minmax'
        Method for feature normalization ('minmax' or 'standardize')
    train_ratio : float, default 0.7
        Proportion of data for training
    val_ratio : float, default 0.15
        Proportion of data for validation
    test_ratio : float, default 0.15
        Proportion of data for testing
    window_size : int, default 24
        Number of time steps in input window
    forecast_horizon : int, default 1
        Number of time steps to predict
    batch_size : int, default 32
        Batch size for TensorFlow datasets
    shuffle_training : bool, default True
        Whether to shuffle training data
    target_columns : List[int], optional
        Column indices to predict (None for all columns)
    use_time_features : bool, default True
        Whether to add cyclical time features
    use_holiday_features : bool, default False
        Whether to add holiday indicators
    country : str, default 'France'
        Country for holiday calendar
    num_fourier_terms : int, default 3
        Number of Fourier terms for cyclical features
    verbose : bool, default True
        Whether to print progress information

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all preprocessing results:
        - 'datasets': {'train': tf.data.Dataset, 'val': tf.data.Dataset, 'test': tf.data.Dataset}
        - 'raw_data': {'train': pd.DataFrame, 'val': pd.DataFrame, 'test': pd.DataFrame}
        - 'scaler_params': Dict with normalization parameters
        - 'feature_names': List of feature column names
        - 'preprocessing_info': Dict with preprocessing metadata
    """
    if verbose:
        print("Starting preprocessing pipeline...")

    # Step 1: Load and clean data
    try:
        # Assuming the data is in the standard format
        df_raw = pd.read_csv(data_path, sep=';', low_memory=False)
    except Exception as e:
        raise FileNotFoundError(f"Could not load data from {data_path}: {e}")

    # Clean the data
    df_clean = clean_data(df_raw)
    if verbose:
        print("✅ Data loaded and cleaned")

    # Step 2: Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    if missing_before > 0:
        df_clean = fill_missing_values(df_clean)
        if verbose:
            print("✅ Missing values filled")
    elif verbose:
        print("✅ No missing values found")

    # Step 3: Add time-based features
    if use_time_features:
        df_clean, fourier_columns = create_fourier_features(
            df_clean, num_fourier_terms)
        if verbose:
            print("✅ Time features added")

    # Step 4: Add holiday features (optional)
    if use_holiday_features:
        try:
            df_clean = add_holiday_features(df_clean, country=country)
            if verbose:
                print("✅ Holiday features added")
        except Exception as e:
            if verbose:
                print(f"⚠️ Could not add holiday features: {e}")

    # Step 5: Downsample data
    df_downsampled = downsample_time_series(df_clean, freq=downsample_freq)
    if verbose:
        print("✅ Data downsampled")

    # Step 6: Split data chronologically
    train_df, val_df, test_df, split_info = split_time_series_by_ratios(
        df_downsampled, train_ratio, val_ratio, test_ratio
    )
    if verbose:
        print("✅ Data split completed")

    # Step 7: Normalize features (fit on training data only)
    # Fit normalization on training data
    train_normalized, scaler_params = normalize_features(
        train_df, method=normalization_method
    )

    # Apply normalization to validation and test data
    val_normalized, _ = normalize_features(
        val_df, method=normalization_method, scaler_params=scaler_params
    )
    test_normalized, _ = normalize_features(
        test_df, method=normalization_method, scaler_params=scaler_params
    )

    if verbose:
        print("✅ Features normalized")

    # Step 8: Create sliding windows
    # Create datasets for training, validation, and testing
    train_dataset = create_sliding_windows(
        train_normalized, window_size, forecast_horizon,
        batch_size=batch_size, shuffle=shuffle_training, target_columns=target_columns
    )

    val_dataset = create_sliding_windows(
        val_normalized, window_size, forecast_horizon,
        batch_size=batch_size, shuffle=False, target_columns=target_columns
    )

    test_dataset = create_sliding_windows(
        test_normalized, window_size, forecast_horizon,
        batch_size=batch_size, shuffle=False, target_columns=target_columns
    )

    if verbose:
        print("✅ Sliding windows created")

    # Prepare feature names
    feature_names = list(df_downsampled.columns)

    # Prepare preprocessing metadata
    preprocessing_info = {
        'original_samples': len(df_raw),
        'clean_samples': len(df_clean),
        'final_samples': len(df_downsampled),
        'missing_values_filled': missing_before,
        'downsample_frequency': downsample_freq,
        'normalization_method': normalization_method,
        'split_info': split_info,
        'window_size': window_size,
        'forecast_horizon': forecast_horizon,
        'target_columns': target_columns,
        'time_features_added': use_time_features,
        'holiday_features_added': use_holiday_features,
        'feature_count': len(feature_names),
        'date_range': {
            'start': str(df_clean.index.min()),
            'end': str(df_clean.index.max())
        }
    }

    if verbose:
        print("✅ Preprocessing pipeline completed!")

    return {
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        },
        'raw_data': {
            'train': train_normalized,
            'val': val_normalized,
            'test': test_normalized
        },
        'scaler_params': scaler_params,
        'feature_names': feature_names,
        'preprocessing_info': preprocessing_info
    }


def get_preprocessing_summary(preprocessing_results: dict[str, Any]) -> None:
    """
    Print a comprehensive summary of preprocessing results.

    Parameters:
    -----------
    preprocessing_results : Dict[str, Any]
        Results dictionary from load_and_preprocess_data()
    """
    info = preprocessing_results['preprocessing_info']

    print("\n" + "="*60)
    print("📊 PREPROCESSING SUMMARY")
    print("="*60)

    print(f"📈 Data Flow:")
    print(f"  • Original samples: {info['original_samples']:,}")
    print(f"  • After cleaning: {info['clean_samples']:,}")
    print(f"  • After downsampling: {info['final_samples']:,}")
    print(f"  • Missing values filled: {info['missing_values_filled']:,}")

    print(f"\n🔧 Processing Settings:")
    print(f"  • Downsampling frequency: {info['downsample_frequency']}")
    print(f"  • Normalization method: {info['normalization_method']}")
    print(f"  • Window size: {info['window_size']}")
    print(f"  • Forecast horizon: {info['forecast_horizon']}")
    print(f"  • Time features: {'✅' if info['time_features_added'] else '❌'}")
    print(
        f"  • Holiday features: {'✅' if info['holiday_features_added'] else '❌'}")

    print(f"\n📅 Time Period:")
    print(f"  • Start: {info['date_range']['start']}")
    print(f"  • End: {info['date_range']['end']}")

    split_info = info['split_info']
    print(f"\n✂️ Data Split:")
    print(
        f"  • Train: {split_info['train_samples']:,} samples ({split_info['train_samples']/split_info['total_samples']*100:.1f}%)")
    print(
        f"  • Val: {split_info['val_samples']:,} samples ({split_info['val_samples']/split_info['total_samples']*100:.1f}%)")
    print(
        f"  • Test: {split_info['test_samples']:,} samples ({split_info['test_samples']/split_info['total_samples']*100:.1f}%)")

    print(f"\n🎯 Features:")
    print(f"  • Total features: {info['feature_count']}")
    print(
        f"  • Target columns: {info['target_columns'] if info['target_columns'] else 'All columns'}")

    print(f"\n📋 Available outputs:")
    print(f"  • TensorFlow datasets: train, val, test")
    print(f"  • Normalized DataFrames: train, val, test")
    print(f"  • Scaler parameters for inverse transform")
    print(f"  • Feature names list")

    print("="*60)
