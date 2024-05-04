# Dividend Policy Predictor Code 

## Introduction

My code involves three main groups of predictors:

1. **Financial Performance Metrics**
2. **Market Metrics**
3. **Corporate Governance Metrics**

For my code, i have focused on specific metrics within each group:

**Group 1: Financial Performance Metrics**
- Dividend Payout Ratio
- Return on Equity (ROE)

**Group 2: Market Metrics**
- Dividend Yield

**Group 3: Corporate Governance Metrics**
- Board Composition

In this README, i will provide an overview of the code structure.

## Code Overview

### 1. Environment Setup
To start, ensure you have your environment variables set up. The code utilizes the `dotenv` library to load environment variables from a `.env` file.

### 2. API Requests
The code makes API requests to retrieve necessary financial data. It uses the Financial Modeling Prep (FMP) API to fetch historical price data, key metrics, ratios, and executive information.

### 3. Data Processing
Once the data is retrieved, it undergoes several processing steps:
- Transformation: The data is transformed into a suitable format for analysis.
- Feature Engineering: Additional predictors are created based on the provided metrics.
- Calculations: Percentage changes and other transformations are applied to the data.

### 4. Predictors
Various predictors are computed, including dividend payout ratio, return on equity, board composition, and dividend yield. These predictors are essential for predicting future dividend policy changes.

### 5. Dataset Creation
Finally, the predictors are combined into a dataset for analysis. The target variable, indicating dividend policy changes, is also included in the dataset.
