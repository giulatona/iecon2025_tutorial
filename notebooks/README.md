# IECON 2025 Tutorial Notebooks

This directory contains Jupyter notebooks for the hands-on part of Tutorial #4 at IECON 2025: "DL-based forecasting of energy related time series: tuning, evaluation, and reproducibility".

## Running the Notebooks

All notebooks can be run in:
- **Local Environment**: Using the DevContainer or Poetry environment (see main README)
- **Google Colab**: Upload notebooks directly or use "Open in Colab" badges

## Notebook Overview

### Setup and Verification

#### `setup_check.ipynb`
**Purpose**: Environment verification  
**Description**: Checks that all required dependencies (TensorFlow, Keras, NumPy, Pandas, etc.) are correctly installed and reports version information.

**When to use**: Run this first to ensure your environment is properly configured before starting the tutorial notebooks.

---

### Tutorial Sequence

Follow these notebooks in order for the complete tutorial experience:

#### 1. `01_data_exploration.ipynb`
**Purpose**: Exploratory Data Analysis  
**Description**: Introduces the UCI Household Power Consumption dataset and performs initial exploration including:
- Dataset structure and basic statistics
- Missing value analysis
- Temporal patterns and seasonality
- Visualization of consumption patterns

**Prerequisites**: None (start here after setup verification)

**Key Takeaways**: 
- Understanding the dataset characteristics
- Identifying temporal patterns in energy consumption
- Recognizing data quality issues

---

#### 2. `02_preprocessing_windowing_features.ipynb`
**Purpose**: Data Preprocessing and Feature Engineering  
**Description**: Prepares data for deep learning models through:
- Handling missing values and outliers
- Temporal resampling and aggregation
- Creating sliding windows for supervised learning
- Adding Fourier features for seasonality
- Train/validation/test splitting with temporal considerations
- Data normalization and scaling

**Prerequisites**: Understanding from notebook 01

**Key Takeaways**:
- Time series to supervised learning transformation
- Importance of temporal ordering in train/test splits
- Feature engineering for time series forecasting
- Creating TensorFlow datasets for efficient training

---

#### 3. `03_forecasting_models.ipynb`
**Purpose**: Model Training and Evaluation  
**Description**: Implements and compares various forecasting approaches:
- **Baseline Models**: Naive seasonal forecasting
- **Feedforward Neural Networks**: Multi-layer perceptron for multi-step forecasting
- **LSTM Encoder-Decoder**: Recurrent architecture for sequence-to-sequence prediction
- Model evaluation with proper time series metrics (MAE, RMSE, MSE)
- Visualization of forecasts vs. ground truth

**Prerequisites**: Preprocessed data from notebook 02

**Key Takeaways**:
- Building deep learning forecasting models with Keras
- Understanding encoder-decoder architectures
- Proper evaluation methodology for time series
- Comparing model performance and complexity trade-offs

**Key Concepts Covered**:
- Keras model building APIs (Sequential, Functional, Subclassing)
- Multi-step ahead forecasting
- Exogenous variable utilization
- Early stopping and hyperparameter tuning

---

### Additional Resources

#### `figure_preparation.ipynb`
**Purpose**: Presentation Figure Generation  
**Description**: Creates publication-quality figures for presentations including:
- Power consumption time series plots
- Solar irradiance data from PVGIS API
- Wind speed data from Open-Meteo API
- Combined subplot visualizations
- Educational time series concept figures

**Prerequisites**: None (standalone)

**Note**: This notebook is not part of the main tutorial sequence but provides examples of data visualization and working with external APIs.

---

## Dataset Information

**Source**: [UCI Machine Learning Repository - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

**Description**: Measurements of electric power consumption in one household with a one-minute sampling rate over nearly 4 years (2006-2010).

**Variables**:
- `Global_active_power`: Household global minute-averaged active power (kilowatts)
- `Global_reactive_power`: Household global minute-averaged reactive power (kilowatts)
- `Voltage`: Minute-averaged voltage (volts)
- `Global_intensity`: Household global minute-averaged current intensity (amperes)
- `Sub_metering_1`: Energy sub-metering for kitchen (watt-hours)
- `Sub_metering_2`: Energy sub-metering for laundry (watt-hours)
- `Sub_metering_3`: Energy sub-metering for climate control (watt-hours)

**Citation**: 
> Georges Hebrail (georges.hebrail '@' edf.fr), Senior Researcher, EDF R&D, Clamart, France  
> Alice Berard, TELECOM ParisTech Master of Engineering Internship at EDF R&D, Clamart, France

---

## Configuration and Hyperparameters

The tutorial notebooks use the following default configuration:

```python
SETUP_CONFIG = {
    'window_size': 36,          # 36 hours of historical data
    'forecast_horizon': 24,     # Predict next 24 hours
    'batch_size': 32,
    'max_epochs': 50,
    'validation_split': 0.2,
    'initial_learning_rate': 0.01,
    'target_columns': [0],      # Global_active_power
    'use_time_features': True,
    'use_holiday_features': False,
    'downsample_freq': '1H',
    'num_fourier_terms': 3,
}
```

Feel free to experiment with these parameters to understand their impact on model performance!

---

## Tips for Success

1. **Run notebooks in order**: Each notebook builds on concepts from previous ones
2. **Experiment freely**: Try modifying hyperparameters and configurations
3. **Check GPU availability**: Deep learning models train faster with GPU support
4. **Monitor training**: Watch loss curves to detect overfitting
5. **Compare models**: Use validation metrics to make fair comparisons

---

## Troubleshooting

**Import errors**: Ensure you've run `setup_check.ipynb` first and all dependencies are installed

**Memory issues**: Reduce `batch_size` in the configuration

**Slow training**: Consider reducing `max_epochs` or using a smaller `window_size`

**Google Colab**: If running in Colab, the notebooks will automatically install required packages from GitHub

---

## Questions or Issues?

For tutorial-related questions, contact:
- Giuseppe La Tona: giuseppe.latona@cnr.it

For code issues, please open an issue on the [GitHub repository](https://github.com/giulatona/iecon2025_tutorial).
