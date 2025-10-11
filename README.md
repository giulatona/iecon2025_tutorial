# IECON 2025 Tutorial

Code for hands-on part of **Tutorial #4** at IECON2025: "DL-based forecasting of energy related time series: tuning, evaluation, and reproducibility"

**Lecturers**: Giuseppe La Tona (giuseppe.latona@cnr.it), Christoph Bergmeir

**Conference**: [IECON 2025 - 51st Annual Conference of the IEEE Industrial Electronics Society](https://iecon2025.org/)  
**Date**: October 14-17, 2025  
**Location**: Madrid, Spain

## About the Tutorial

This tutorial provides an informative and dedicated training session on deep learning-based forecasting methods for energy-related time series. The session covers fundamentals and state-of-the-art techniques in:

- Deep learning architectures for time series forecasting
- Hyperparameter tuning strategies
- Model evaluation methodologies
- Reproducibility best practices in energy forecasting

For more information about all IECON 2025 tutorials, visit: https://iecon2025.org/tutorials/

## Notebooks

This repository includes hands-on Jupyter notebooks that can be run in multiple environments:

- **Local Development**: Using the provided DevContainer or local Poetry environment
- **Google Colab**: All notebooks are fully compatible with Google Colab for easy access without local setup

### Tutorial Notebooks

1. **[setup_check.ipynb](notebooks/setup_check.ipynb)** - Verify environment and dependencies
2. **[01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)** - Exploratory data analysis of household power consumption
3. **[02_preprocessing_windowing_features.ipynb](notebooks/02_preprocessing_windowing_features.ipynb)** - Data preprocessing, windowing, and feature engineering
4. **[03_forecasting_models.ipynb](notebooks/03_forecasting_models.ipynb)** - Deep learning models for energy forecasting (baselines, feedforward, LSTM)
5. **[figure_preparation.ipynb](notebooks/figure_preparation.ipynb)** - Generate presentation figures

For detailed notebook documentation, see [notebooks/README.md](notebooks/README.md).

## Project Setup

This project uses Python 3.12 with Poetry for dependency management and DevContainer for consistent development environment.

## Features

- Python 3.12
- Poetry for dependency management
- DevContainer support using pre-built container image
- Pre-configured with TensorFlow, Keras, NumPy, Pandas, and other ML libraries (Google Colab compatible versions)
- Scientific computing libraries: SciPy, scikit-learn, matplotlib, seaborn, plotly
- Interactive widgets support with ipywidgets
- Code formatting with Black
- Linting with Flake8 and MyPy
- Testing with pytest
- Jupyter notebook support with Google Colab compatibility

## Dependencies

### Core ML Libraries
- **TensorFlow**: 2.19.0
- **Keras**: 3.10.0  
- **NumPy**: 2.1.0
- **Pandas**: 2.2.2

### Scientific Computing & Visualization
- **SciPy**: 1.16.2
- **scikit-learn**: 1.6.1
- **matplotlib**: 3.10.0
- **seaborn**: 0.13.2
- **plotly**: 5.24.1
- **ipywidgets**: 7.7.1

*Note: These versions match those available in Google Colab as of October 2025.*

## Quick Start

### Using DevContainer (Recommended)

1. Open this repository in VS Code
2. When prompted, click "Reopen in Container" or use Command Palette: `Dev Containers: Reopen in Container`
3. The DevContainer will automatically set up the environment and install dependencies

### Local Development

1. Ensure you have Python 3.12 and Poetry installed
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Available Commands

Use Poetry directly for common tasks:

```bash
poetry install         # Install dependencies
poetry run pytest      # Run tests
poetry run black src/ tests/  # Format code
poetry run flake8 src/ tests/  # Lint code
poetry run mypy src/    # Type checking
poetry run jupyter notebook  # Start Jupyter
poetry build           # Build the package
```

## Project Structure

```
.
├── .devcontainer/         # DevContainer configuration
│   └── devcontainer.json  # Container setup using pre-built image
├── src/
│   └── iecon2025_tutorial/  # Main package (editable install)
│       ├── __init__.py
│       └── iecon2025_tutorial.py  # Version checker and setup utilities
├── tests/                 # Test files
├── notebooks/            # Jupyter notebooks
│   └── setup_check.ipynb  # Environment verification
├── pyproject.toml        # Poetry configuration
└── README.md
```

## Development

This project uses:

- **Poetry** for dependency management and packaging
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **pytest** for testing

The DevContainer uses the official Microsoft Python 3.12 container image and automatically installs the package in editable mode.

## Testing

Run tests with:
```bash
poetry run pytest
```

## Jupyter Notebooks

Start Jupyter notebook server locally:
```bash
poetry run jupyter notebook
```

The DevContainer automatically forwards port 8888 for Jupyter access.

**Google Colab**: All notebooks in this repository are designed to run seamlessly in Google Colab. Simply upload the notebook files to Colab or use the "Open in Colab" badges (coming soon) for direct access.

## Verification

Use the provided notebook `notebooks/setup_check.ipynb` to verify that all dependencies are correctly installed and working.
