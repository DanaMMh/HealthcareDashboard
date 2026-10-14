# Healthcare Dashboard

A Streamlit application for exploring a healthcare dataset and evaluating a baseline predictive model for patient outcomes.

## Overview

This project loads a patient dataset from `patients.csv`, previews the data, summarizes key statistics, and shows a distribution chart for a selected numeric metric. It also trains a simple logistic-regression model to estimate prediction accuracy on a held-out test set.

## Features

- Dataset preview and summary statistics
- Interactive metric selection for distribution charts
- Baseline model evaluation using a stratified train/test split
- Clean dashboard layout with Streamlit
- Model accuracy reporting with clear caveats about clinical use

## Project structure

- `app.py` — main Streamlit dashboard and model logic
- `patients.csv` — healthcare dataset used by the app

## Requirements

Install the Python dependencies with:

```bash
pip install streamlit pandas matplotlib scikit-learn
```

## Run the app

From the repository root:

```bash
streamlit run app.py
```

The dashboard will open in your default browser.

## Notes

- The app uses a baseline logistic-regression model for demonstration purposes.
- Accuracy is reported on a held-out validation set, but it should not be interpreted as clinical evidence or medical advice.
- The dataset and model are intended for exploratory analysis and learning.
