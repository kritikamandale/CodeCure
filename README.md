# Epidemic Spread Prediction & Decision Support System

An end-to-end AI-powered system for forecasting epidemic outbreaks, analyzing transmission dynamics, and generating actionable public health insights.

---

## Problem Statement

Epidemic outbreaks evolve rapidly, making it difficult for authorities to anticipate future trends and take timely action. Traditional analysis focuses on historical data but lacks predictive and decision-support capabilities.

This project addresses that gap by combining **machine learning, epidemiological modeling, and interactive visualization** to forecast outbreaks and provide actionable insights.

---

## Key Features

* **Outbreak Forecasting**
  * ARIMA (statistical baseline)
  * LSTM (deep learning model for high accuracy)

* **Biological Modeling**

  * SEIR compartmental model
  * Effective reproduction number (Rt) analysis
  * Herd immunity threshold estimation

* **Interactive Dashboard**
  * Country-wise case trends
  * 7–14 day forecasts
  * Model comparison (ARIMA vs LSTM)

* **Risk Map**

  * Global outbreak risk visualization
  * Country-level risk scoring

* **Early Warning System**

  * Threshold-based alerts (Critical / Warning / Safe)

* **What-if Simulator**

  * Policy impact simulation (lockdown, vaccination)

* **Model Explainability**

  * SHAP-based feature importance

* **Auto Report Generator**

  * Downloadable PDF summary for decision-makers

---

## Approach
### 1. Data Pipeline

* Johns Hopkins COVID-19 dataset
* Our World in Data (OWID)
* Google Mobility data

### 2. Feature Engineering

* Lag features, rolling statistics
* Growth rate, doubling time, Rt
* Mobility-based leading indicators

### 3. Modeling

* **ARIMA** → baseline time-series forecasting
* **LSTM** → captures non-linear epidemic dynamics
* **SEIR** → validates results biologically

### 4. Visualization

* Streamlit dashboard
* Plotly interactive charts
* Folium risk maps

---

## Results
* Achieved best accuracy 
* LSTM outperformed ARIMA in capturing non-linear trends
* Rt analysis successfully identified outbreak waves
* SEIR model validated epidemic behavior biologically

---

## Project Structure

```
project/
│
├── app.py
├── requirements.txt
├── data/
├── models/
├── src/
├── outputs/
└── notebooks/
```

---

## Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py
```

---

## Live Demo

👉 https://codecuree.streamlit.app

---

## Screenshots

<img width="1914" height="844" alt="image" src="https://github.com/user-attachments/assets/9906a5c3-d66f-4116-8b27-140301254fb1" />
<img width="1919" height="777" alt="image" src="https://github.com/user-attachments/assets/51266a54-916c-4848-b214-63abfe5acbe5" />
<img width="1908" height="832" alt="image" src="https://github.com/user-attachments/assets/ae644174-6d02-436d-b472-59da8dcabfa5" />



---

## Key Insights

* Rt > 1 indicates outbreak growth phase
* Mobility changes act as leading indicators
* Vaccination reduces susceptible population (SEIR validation)
* Policy interventions significantly impact future trends

---

## Future Improvements

* Real-time data integration
* More granular (state-level) predictions
* Advanced hybrid models (ARIMA + LSTM ensemble)

---

## Tech Stack

* Python, Pandas, NumPy
* TensorFlow / Keras (PyTorch)
* Statsmodels (ARIMA)
* Streamlit
* Plotly, Folium
* SHAP
---

## Team

* Kritika Mandale
* Shreya Shande

---

##License

MIT License
