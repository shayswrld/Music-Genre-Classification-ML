# Music Genre Classification using FMA Dataset

This project implements a music genre classification system using the Free Music Archive (FMA) dataset and PyTorch. The system extracts audio features from music tracks and trains a feed-forward neural network to predict music genres.

## Project Structure

```
├── fma/
│   ├── data_loader.py          # FMA dataset loader and preprocessor
│   ├── utils.py                # Utility functions for FMA data handling
│   ├── features.py             # Audio feature extraction utilities
│   ├── creation.py             # Dataset creation and download scripts
│   ├── example_usage.py        # Example usage of the data loader
│   ├── training.ipynb          # Main training notebook with hyperparameter tuning
│   ├── EDA.ipynb               # Exploratory Data Analysis notebook
│   └── fma_metadata/           # FMA dataset metadata files
│       ├── tracks.csv
│       ├── features.csv
│       ├── genres.csv
│       └── ...
├── requirements.txt            # Python dependencies for pip
├── environment.yml             # Conda environment specification
└── README.md                   # This file
```

## Features

- **Dataset Loading**: Loading and preprocessing of FMA dataset
- **Feature Extraction**: Support for various audio features (MFCC, spectral contrast, chroma, etc.)
- **Neural Network Training**: PyTorch-based feed-forward network with configurable architecture
- **Hyperparameter Tuning**: Grid search over dropout rates and weight decay
- **Evaluation**: Weighted F1-score evaluation for imbalanced classes
- **Visualization**: Training curves and hyperparameter optimization heatmaps

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- FMA dataset metadata files (download from [FMA website](https://freemusicarchive.org/))
- Place metadata files in `fma/fma_metadata/` directory

### Installation

#### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/Rowan-Rosenberg/Music-Genre-Classification-ML.git
cd Music-Genre-Classification-ML

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/Rowan-Rosenberg/Music-Genre-Classification-ML.git
cd Music-Genre-Classification-ML

# Create conda environment
conda env create -f environment.yml
conda activate music-genre-classification
```

### Dataset Setup

1. Download the FMA metadata from the [official FMA website](https://freemusicarchive.org/)
2. Extract the metadata files to `fma/fma_metadata/` directory
3. Ensure the following files are present:
   - `tracks.csv`
   - `features.csv`
   - `genres.csv`
   - `echonest.csv`

## Usage

### Training the Model

Open `fma/training.ipynb` in Jupyter notebook and run the cells to:

1. Load and preprocess the data
2. Define the neural network architecture
3. Train the model with hyperparameter tuning
4. Evaluate performance on validation set
5. Visualize results

### Key Components

#### Data Loader (`data_loader.py`)
- Loads FMA metadata and features
- Supports train/validation/test splits
- Handles feature selection and standardization
- Supports both single-label and multi-label classification

#### Neural Network (`training.ipynb`)
- Configurable feed-forward architecture
- Swish activation functions
- Dropout and L2 regularization
- Batch normalization support

#### Evaluation (`training.ipynb`)
- Weighted F1-score for imbalanced classes
- Accuracy metrics
- Hyperparameter optimization visualization

## Model Architecture

The default architecture consists of:
- Input layer: Audio features (MFCC, spectral contrast, etc.)
- Hidden layers: 2 layers with 2×input_dim neurons each
- Activation: Swish (SiLU)
- Output: Softmax over genre classes
- Regularization: Dropout + L2 weight decay

## Hyperparameter Tuning

The notebook performs grid search over:
- Dropout rates: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
- Weight decay: [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

Results are visualized as a heatmap of validation F1-score.