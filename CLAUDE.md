# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qlib quantitative finance research repository containing modular Jupyter notebooks for building and testing quantitative trading strategies using the Microsoft Qlib framework. The notebooks demonstrate data processing, model training, backtesting, and performance evaluation for Chinese A-share market (specifically CSI300 index components).

## Repository Structure

```
qlib-experiment/
├── notebooks/
│   ├── 00_quickstart.ipynb              # Quick start example for beginners
│   ├── 01_introduction_and_config.ipynb # Project intro and basic configuration
│   ├── 02_data_module_detailed.ipynb    # Data handling and preprocessing
│   ├── 03_model_training_comprehensive.ipynb # Model training workflows
│   ├── 04_evaluation_module_detailed.ipynb   # Backtesting and evaluation
│   ├── 05_utils_helpers_advanced.ipynb       # Utility functions
│   ├── detailed_workflow.ipynb          # Complete end-to-end workflow (26MB, comprehensive)
│   ├── experiment_config.json           # Default experiment configuration
│   └── mlruns/                          # MLflow experiment tracking outputs
├── experiments/                          # Custom experiment notebooks and results
│   ├── volume_analysis_experiment.ipynb  # Volume-based analysis experiments
│   └── volume_analysis_results.csv       # Experiment output data
├── pyproject.toml                        # Project metadata and dependencies
├── uv.lock                               # Locked dependencies for reproducibility
└── README.md                             # Project documentation
```

## Key Development Commands

### Environment Setup

**Recommended: Using uv (faster, with lock file)**
```bash
# Install uv if not already installed
pip install uv

# Sync all dependencies from pyproject.toml (recommended)
uv sync

# Install optional ML models
uv pip install xgboost catboost torch

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/Mac:
source .venv/bin/activate
```

**Alternative: Using pip**
```bash
# Install Qlib and dependencies
pip install pyqlib
pip install lightgbm pandas numpy matplotlib plotly

# Optional: Install additional model libraries
pip install xgboost catboost torch
```

### Running Notebooks
```bash
# Start Jupyter notebook server
jupyter notebook

# Or use Jupyter Lab
jupyter lab

# Run a specific notebook
jupyter notebook notebooks/00_quickstart.ipynb
```

### Data Operations
```python
# Initialize Qlib (required at start of each session)
import qlib
# On Windows, data stored in: C:\Users\<username>\.qlib\qlib_data\cn_data
# On Unix/Mac, data stored in: ~/.qlib/qlib_data/cn_data
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

# Download Qlib data (if not already present, ~1GB download)
from qlib.tests.data import GetData
GetData().qlib_data(exists_skip=True)

# Check if data exists
import os
data_path = os.path.expanduser("~/.qlib/qlib_data/cn_data")
if os.path.exists(data_path):
    print("Data already downloaded")
```

### Common Workflow Pattern
```python
# 1. Set up experiment parameters
MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "experiment_name"

# 2. Initialize data handler (e.g., Alpha158)
from qlib.contrib.data.handler import Alpha158
handler = Alpha158(
    instruments=MARKET,
    start_time="2008-01-01",
    end_time="2020-08-01"
)

# 3. Create dataset
from qlib.data.dataset import DatasetH
dataset = DatasetH(
    handler,
    segments={
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2020-08-01")
    }
)

# 4. Train model
from qlib.contrib.model.gbdt import LGBModel
model = LGBModel()
model.fit(dataset)

# 5. Run backtest
from qlib.workflow import R
with R.start(experiment_name=EXP_NAME):
    # Training and evaluation code
    pass
```

## Architecture Notes

### Data Flow Pipeline
1. **Raw Data**: Downloaded from Qlib servers, stored in `~/.qlib/qlib_data/cn_data/`
2. **Data Loading**: QlibDataLoader fetches raw price/volume data and financial indicators
3. **Data Processing**: DataHandler applies preprocessing (normalization, fillna, etc.)
4. **Dataset Creation**: DatasetH organizes data into train/valid/test segments
5. **Model Training**: Models consume datasets and generate predictions
6. **Backtesting**: Strategy execution and portfolio analysis based on predictions

### Notebook Dependencies
- **All notebooks** require Qlib initialization at the start
- **02_data_module** outputs can be used by **03_model_training**
- **03_model_training** outputs (models) are used by **04_evaluation_module**
- **00_quickstart** is standalone and contains all steps in simplified form

### Key Components

- **Alpha158**: Pre-built feature engineering module with 158 technical indicators
- **LGBModel**: LightGBM-based prediction model for stock returns
- **TopkDropoutStrategy**: Portfolio selection strategy (top-k stocks with periodic rebalancing)
- **MLflow Integration**: All experiments tracked in `mlruns/` directory

### Important Concepts

- **Dynamic Universe**: Stock pool changes over time based on index composition
- **Point-In-Time Data**: Ensures no look-ahead bias in fundamental data
- **Express Engine**: Formula-based feature calculation system
- **Recorder System**: Experiment tracking and artifact management

## Notebook Learning Path

### For Beginners
Start with `00_quickstart.ipynb` for a complete but minimal example, then proceed:
1. `01_introduction_and_config.ipynb` - Setup and configuration
2. `02_data_module_detailed.ipynb` - Understanding data handling
3. `03_model_training_comprehensive.ipynb` - Model training details
4. `04_evaluation_module_detailed.ipynb` - Backtesting and evaluation
5. `05_utils_helpers_advanced.ipynb` - Advanced utilities

### For Experienced Users
- Jump directly to `detailed_workflow.ipynb` for comprehensive examples
- Use specific module notebooks as reference documentation

## Dependencies

**Core Requirements** (defined in `pyproject.toml`):
- Python 3.8+ (>=3.8 specified in pyproject.toml)
- pyqlib (Microsoft's quantitative investment library)
- pandas (>=1.3.0), numpy (>=1.21.0), matplotlib (>=3.5.0), plotly
- scikit-learn (>=1.0.0), seaborn (>=0.11.0), statsmodels (>=0.14.1)
- lightgbm (gradient boosting models)
- jupyter, jupyterlab, notebook (>=6.0.0)
- ipywidgets (>=8.0.0) for interactive visualizations

**Optional ML Models** (project.optional-dependencies):
- xgboost, catboost, torch (install with: `uv sync --extra ml-extra`)

**Development Tools**:
- ruff (linting, configured in pyproject.toml)
- Line length: 120 characters
- Target: Python 3.8

Install all dependencies:
```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

## Data Considerations

- Default data source: Chinese A-share market
- Frequency: Daily trading data
- Adjustments: Price data is pre-adjusted for splits/dividends
- Trading costs: Configurable in backtest settings (default: 0.05% open, 0.15% close)

## Default Experiment Configuration

The repository includes `notebooks/experiment_config.json` with default settings:
- Market: CSI300 (Chinese A-share market)
- Benchmark: SH000300 (CSI 300 Index)
- Training period: 2008-01-01 to 2014-12-31
- Validation period: 2015-01-01 to 2016-12-31
- Test period: 2017-01-01 to 2020-08-01

## Performance Notes

- Large notebooks (detailed_workflow.ipynb is 26MB) may take time to load
- Data download on first run requires ~1GB storage and network bandwidth
- Model training with full CSI300 universe can be memory-intensive
- Use step_len parameter carefully in TSDatasetH to avoid excessive memory usage
- MLflow experiments are tracked in `notebooks/mlruns/` directory
- Initial data loading may take 60+ seconds (as seen in 00_quickstart.ipynb)
- LightGBM training typically completes in <2 seconds with default parameters

## Common Issues and Solutions

### Data Download Issues
If data download fails, manually download from:
```python
# Alternative: Use specific region data
GetData().qlib_data(region="cn", exists_skip=True)
```

### Memory Issues with Large Datasets
```python
# Reduce memory usage by limiting stock universe
handler = Alpha158(
    instruments="csi100",  # Use CSI100 instead of CSI300
    start_time="2018-01-01",  # Shorter time period
    end_time="2020-12-31"
)
```

### Notebook Kernel Crashes
- Clear output of large notebooks before committing: `Kernel > Restart & Clear Output`
- Use `del` to explicitly free memory after large operations
- Consider splitting large notebooks into smaller chunks

### Chinese Font Display Issues
If Chinese characters don't display correctly in matplotlib plots:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
```

### Platform-Specific Notes
**Windows Users:**
- Data stored in: `C:\Users\<username>\.qlib\qlib_data\cn_data`
- Virtual environment activation: `.venv\Scripts\activate`
- Use forward slashes or raw strings for paths in Python

**Unix/Mac Users:**
- Data stored in: `~/.qlib/qlib_data/cn_data`
- Virtual environment activation: `source .venv/bin/activate`