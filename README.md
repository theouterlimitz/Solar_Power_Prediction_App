# Solar Power Prediction App

This repository contains the code and trained model for an interactive web application that predicts solar power generation. The app is built with Streamlit and serves a pre-trained XGBoost Regressor model.

Users can adjust sliders for various weather conditions and time-of-day features to get a live prediction of a solar power plant's AC power output in kilowatts (kW).

---

## The Machine Learning Model

The predictive model at the core of this application is an **XGBoost Regressor**.

* **Training Data:** The model was trained on a real-world dataset containing 34 days of 15-minute time-series data from two solar power plants in India.
* **Features:** It uses weather data (`Irradiation`, `Ambient Temperature`, `Module Temperature`) and time-based features (`Month`, `Hour`, `Minute`, etc.) to make its predictions.
* **Performance:** The model achieved a very high **R-squared ($R^2$) score of approximately 0.96** on the unseen test data, meaning it can explain about 96% of the variance in power output.

For a complete, in-depth analysis of the data curation, exploratory data analysis (EDA), and a comparative evaluation against a neural network, please see the full modeling project here:
* **[View the Full Modeling Project on GitHub](https://github.com/theouterlimitz/Solar_Power_Prediction)** *(Note: You may need to create this second repo and update the link)*

---

## How to Run This App Locally

To run this interactive application on your own computer, please follow these steps.

### 1. Prerequisites
* You must have Python 3 installed on your system.
* It is recommended to use a virtual environment.

### 2. Setup & run
First, clone this repository to your local machine and navigate into the directory:
```bash
git clone [https://github.com/theouterlimitz/Solar_Power_Prediction_App.git](https://github.com/theouterlimitz/Solar_Power_Prediction_App.git)
cd Solar_Power_Prediction_App

pip install -r requirements.txt

streamlit run app.py
