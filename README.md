# Celebal-Anaverse

This project focuses on predicting anomalies based on sensor readings using machine learning and deep learning techniques.

## Problem Statement
The goal is to evaluate the ability to:
- Work with tabular sensor data
- Perform advanced data analysis
- Build robust predictive models for anomaly detection

## Key Features
- Comprehensive data exploration and preprocessing
- Feature engineering and correlation analysis
- Implementation of both classical and advanced ML models
- Hyperparameter tuning and cross-validation
- Detailed model evaluation with appropriate metrics

## Evaluation Criteria
1. **Data Exploration & Preprocessing (20%)**
   - Handling missing values and outliers
   - Feature engineering and correlation analysis

2. **Modeling (60%)**
   - Classical models: Logistic Regression, SVM, KNN, Decision Trees
   - Advanced models: Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks
   - Model selection justification and tuning strategies

3. **Model Evaluation (20%)**
   - Metrics: Accuracy, Precision, Recall, F1 Score
   - Robustness checks: backtesting and residual analysis

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/anomaly-detection.git
cd anomaly-detection
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your dataset in the data/ folder
2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/anomaly_detection.ipynb
   ```

## Dependencies
Python 3.7+

NumPy

Pandas

Scikit-Learn

XGBoost/LightGBM/CatBoost

TensorFlow/PyTorch (optional for neural networks)

Matplotlib/Seaborn

Evaluation Metrics

## Primary evaluation based on:

F1 score

Accuracy for each class

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.
