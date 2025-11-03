# Sales-Forecasting-ML

A machine learning project for predicting future sales based on historical data using regression algorithms.

## Overview

This project implements sales forecasting using machine learning techniques to help with inventory management and revenue planning. The implementation includes data preprocessing pipelines, feature engineering for seasonal patterns, and comparison of multiple regression models (Linear Regression, Random Forest, XGBoost).

## Features

- Handles seasonal patterns and cyclical trends through engineered features
- Robust data preprocessing for missing values and outliers
- Multiple regression algorithm comparison (Linear Regression, Random Forest, XGBoost)
- Feature engineering including rolling averages, lag features, and temporal indicators
- Modular architecture for easy integration with different data sources
- Evaluation metrics: MAPE, RMSE
- Multi-product forecasting capability

## Installation

### Prerequisites

- Python (NumPy, Pandas, Scikit-learn)
- Matplotlib, Seaborn for visualization

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/TejaVamshidharReddy/Sales-Forecasting-ML.git
cd Sales-Forecasting-ML
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Prediction Model

```bash
python src/main.py
```

### Using Custom Data

Replace the sample data in `data/sample_sales_data.csv` with your own sales data. Ensure your CSV file contains the following columns:

- `date`: Date of the sale
- `product_id`: Product identifier
- `quantity`: Number of units sold
- `price`: Price per unit
- `revenue`: Total revenue (quantity × price)

## Example Input/Output

### Input (sample_sales_data.csv)
```csv
date,product_id,quantity,price,revenue
2024-01-01,P001,50,29.99,1499.50
2024-01-02,P001,45,29.99,1349.55
```

### Output (predictions)
```
Predicted Sales for Next Week:
Product P001: $1,425.30 (±5% confidence interval)
```
